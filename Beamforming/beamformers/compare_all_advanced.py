"""
compare_all.py (ADVANCED)
-------------------------------------------------------
Compares Bartlett, MVDR, and MUSIC beamformers on the same
Infineon BGT60TR13C radar dataset with robustness upgrades:

. Range gating before DOA (min/max range and/or top-K bins)
. Clutter suppression (slow-time mean removal across chirps)
. Snapshot averaging across CPIs (covariance averaged over multiple CPIs)
. SNR-aware diagonal loading (data-driven loading strength)
. Forward–Backward averaging (improves MUSIC for coherent multipath)
. Automatic source-number estimation (AIC / MDL)

Outputs:
- Range–Angle maps for each beamformer
- Beampattern overlays at a selected/gated range bin
- MUSIC peak detection + estimated #sources
"""

import numpy as np
import matplotlib.pyplot as plt

from io_loader import load_infineon_recording, preprocess_frame, get_params
from array_geometry import steering_vector

from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks


# covariance helpers

def forward_backward_average(R: np.ndarray) -> np.ndarray:
    """
    Forward–Backward averaging for a ULA:
        R_fb = 0.5 * (R + J * R* * J)
    where J is the exchange (reversal) matrix and * is complex conjugate.
    """
    m = R.shape[0]
    J = np.fliplr(np.eye(m))
    return 0.5 * (R + J @ R.conj() @ J)


def diag_loading_snr_aware(
    R: np.ndarray,
    min_loading: float = 1e-6,
    max_loading: float = 1e-1
) -> tuple[np.ndarray, float, float]:
    """
    SNR-aware diagonal loading.

    Estimate noise floor from the smallest eigenvalues, then estimate
    SNR ~= (lambda_max - noise_floor) / noise_floor.

    Diagonal loading strength increases when SNR is low.

    Returns:
      R_loaded, eps_used, snr_est_linear
    """
    m = R.shape[0]
    R = 0.5 * (R + R.conj().T)

    eigvals = np.linalg.eigvalsh(R)  # ascending
    k_noise = max(1, m // 2)
    noise_floor = float(np.mean(eigvals[:k_noise]).real)
    noise_floor = max(noise_floor, 1e-12)

    lam_max = float(eigvals[-1].real)
    snr_est = max((lam_max - noise_floor) / noise_floor, 0.0)

    alpha = 1.0 / (1.0 + snr_est)  # snr=0 =>1, snr>> => ~0
    loading_factor = min_loading + (max_loading - min_loading) * alpha

    eps = loading_factor * noise_floor
    R_loaded = R + eps * np.eye(m)
    return R_loaded, float(eps), float(snr_est)


def estimate_num_sources_aic_mdl(
    R: np.ndarray,
    n_snapshots: int,
    max_sources: int | None = None
) -> dict:
    """
    Automatic source-number estimation using AIC and MDL (Wax-Kailath).
    
    Automatic source-number estimation using Akaike Information Criterion (AIC) and Minimum Description Length (MDL) following the Wax–Kailath method.

    Inputs:
      R: covariance [m x m]
      n_snapshots: snapshots used (e.g., chirps * num_cpis)
      max_sources: cap (<= m-1)

    Returns:
      {'aic_k', 'mdl_k', 'eigvals_desc'}
    """
    m = R.shape[0]
    R = 0.5 * (R + R.conj().T)
    eigvals = np.linalg.eigvalsh(R)      # ascending
    eigvals = eigvals[::-1].real         # descending
    eigvals = np.maximum(eigvals, 1e-18)

    if max_sources is None:
        max_sources = m - 1
    max_sources = int(np.clip(max_sources, 0, m - 1))

    N = max(int(n_snapshots), 1)

    aic_vals, mdl_vals = [], []

    for k in range(0, max_sources + 1):
        noise_eigs = eigvals[k:]
        p = m - k  # number of noise eigenvalues

        gm = float(np.exp(np.mean(np.log(noise_eigs))))
        am = float(np.mean(noise_eigs))
        am = max(am, 1e-18)

        ratio = max(gm / am, 1e-18)

        aic = -2 * N * p * np.log(ratio) + 2 * k * (2 * m - k)
        mdl = -N * p * np.log(ratio) + 0.5 * k * (2 * m - k) * np.log(N)

        aic_vals.append(aic)
        mdl_vals.append(mdl)

    return {"aic_k": int(np.argmin(aic_vals)),
            "mdl_k": int(np.argmin(mdl_vals)),
            "eigvals_desc": eigvals}


def covariance_from_cube(
    cube: np.ndarray,
    rbin: int,
    *,
    clutter_suppress: bool = True,
) -> tuple[np.ndarray, int]:
    """
    Builds spatial covariance R at one range bin for one CPI cube.
      cube: [range_bins, chirps, rx]
      X:    [chirps, rx]
    """
    X = cube[rbin, :, :]  # [chirps, rx]
    X = X * np.hanning(X.shape[0])[:, None]  # slow-time window

    if clutter_suppress:
        # Removes slow-time mean per RX channel (stationary clutter ~ zero Doppler)
        X = X - np.mean(X, axis=0, keepdims=True)

    R = np.cov(X, rowvar=False)  # [rx, rx]
    return R, X.shape[0]


def average_covariance_across_cpis(
    filepath: str,
    *,
    cpi_indices: list[int],
    range_bin: int,
    clutter_suppress: bool = True,
) -> tuple[np.ndarray, int]:
    """
    Snapshot averaging across CPIs:
      - loads CPI -> preprocess -> R(range_bin)
      - averages R over CPIs
    """
    R_sum = None
    total_snaps = 0

    for idx in cpi_indices:
        frame = load_infineon_recording(filepath, cpi_idx=idx)
        cube = preprocess_frame(frame)
        R, snaps = covariance_from_cube(cube, range_bin, clutter_suppress=clutter_suppress)
        R_sum = R if R_sum is None else (R_sum + R)
        total_snaps += snaps

    R_avg = R_sum / max(len(cpi_indices), 1)
    R_avg = 0.5 * (R_avg + R_avg.conj().T)
    return R_avg, total_snaps


# ===============================================================
#  Beamformer spectra
# ===============================================================
def bartlett_spectrum(R: np.ndarray, A: np.ndarray) -> np.ndarray:
    P = np.zeros(A.shape[1], dtype=float)
    for i in range(A.shape[1]):
        a = A[:, i].reshape(-1, 1)
        P[i] = float(np.real(a.conj().T @ R @ a))
    return P


def mvdr_spectrum(R: np.ndarray, A: np.ndarray, eps_floor: float = 1e-12) -> np.ndarray:
    m = R.shape[0]
    R_loaded, eps_used, snr_est = diag_loading_snr_aware(R)
    R_inv = np.linalg.inv(R_loaded)

    P = np.zeros(A.shape[1], dtype=float)
    for i in range(A.shape[1]):
        a = A[:, i].reshape(-1, 1)
        denom = float(np.real(a.conj().T @ R_inv @ a))
        P[i] = 1.0 / max(denom, eps_floor)
    return P


def music_spectrum(
    R: np.ndarray,
    A: np.ndarray,
    *,
    num_sources: int,
    fb_avg: bool = True,
    eps_floor: float = 1e-12
) -> np.ndarray:
    m = R.shape[0]

    if fb_avg:
        R = forward_backward_average(R)

    R_loaded, eps_used, snr_est = diag_loading_snr_aware(R)

    eigvals, eigvecs = np.linalg.eigh(R_loaded)
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]

    num_sources = int(np.clip(num_sources, 0, m - 1))
    E_n = eigvecs[:, num_sources:]

    P = np.zeros(A.shape[1], dtype=float)
    for i in range(A.shape[1]):
        a = A[:, i].reshape(-1, 1)
        denom = float(np.linalg.norm(E_n.conj().T @ a) ** 2)
        P[i] = 1.0 / max(denom, eps_floor)
    return P


# ===============================================================
#  Range gating helpers
# ===============================================================
def pick_range_bins_by_gate(
    cube: np.ndarray,
    params: dict,
    *,
    min_range_m: float = 0.15,
    max_range_m: float | None = None,
    top_k: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Chooses range bins for processing:
      - gate by [min_range_m, max_range_m]
      - optionally pick top_k strongest within the gate
    """
    num_ranges = cube.shape[0]
    rng = np.arange(num_ranges) * params["range_res"]

    gate = rng >= float(min_range_m)
    if max_range_m is not None:
        gate &= rng <= float(max_range_m)

    range_powers = np.sum(np.abs(cube) ** 2, axis=(1, 2)).real

    gated_bins = np.where(gate)[0]
    if gated_bins.size == 0:
        gated_bins = np.arange(num_ranges)

    if top_k and top_k > 0:
        idx_sort = np.argsort(range_powers[gated_bins])[::-1]
        sel = gated_bins[idx_sort[: min(top_k, gated_bins.size)]]
        sel = np.sort(sel)
        return sel, range_powers

    return gated_bins, range_powers


# ===============================================================
#  MAIN
# ===============================================================
if __name__ == "__main__":

    # ---------------- USER CONFIG ----------------
    cfg = {
        "start_frequency_Hz": 58_000_000_000,
        "end_frequency_Hz": 63_500_000_000,
        "sample_rate_Hz": 2e6,
        "num_samples": 64,
        "num_rx": 3,
        "tx_mask": 1,
        "tx_power_level": 31,
        "lp_cutoff_Hz": 500_000,
        "hp_cutoff_Hz": 80_000,
        "if_gain_dB": 23,
        "frame_repetition_time_s": 0.07726884633302689,
        "chirp_repetition_time_s": 0.0005911249900236726,
        "num_chirps": 64,
        "tdm_mimo": 0,
    }

    filepath = "radar.npy"

    # CPI selection for snapshot averaging:
    # If radar.npy is 4D: [N_CPI, ch, chirps, samples], set e.g. [0,1,2,3]
    cpi_indices = [0]

    # Range gating before DOA
    min_range_m = 0.20
    max_range_m = None   # e.g., 3.0
    top_k_bins = 0       # e.g., 5

    # Angle grid
    angle_grid = np.linspace(-80, 80, 721)

    # Clutter suppression (slow-time mean removal)
    clutter_suppress = True

    # MUSIC: source number mode: "MDL", "AIC", or "FIXED"
    music_source_mode = "MDL"
    music_fixed_sources = 1
    music_fb_averaging = True

    # Visualization
    smooth_sigma_db = 0.6
    dyn_range_db = 50

    # ---------------- LOAD 1 CPI FOR SHAPES + RANGE GATE ----------------
    params = get_params(cfg)
    data0 = load_infineon_recording(filepath, cpi_idx=cpi_indices[0])
    cube0 = preprocess_frame(data0)  # [range, chirps, rx]
    num_ranges, num_chirps, num_rx = cube0.shape
    print(f"[INFO] Cube shape (1 CPI): {cube0.shape}")

    # Steering matrix
    A = steering_vector(params, angle_grid, d_by_lambda=0.5)

    # Range bins
    selected_bins, range_powers = pick_range_bins_by_gate(
        cube0,
        params,
        min_range_m=min_range_m,
        max_range_m=max_range_m,
        top_k=top_k_bins,
    )
    print(f"[INFO] Range gate selected {len(selected_bins)} / {num_ranges} bins")

    rng_axis = np.arange(num_ranges) * params["range_res"]

    # Storage maps
    maps = {
        "Bartlett": np.full((num_ranges, len(angle_grid)), np.nan, dtype=float),
        "MVDR": np.full((num_ranges, len(angle_grid)), np.nan, dtype=float),
        "MUSIC": np.full((num_ranges, len(angle_grid)), np.nan, dtype=float),
    }

    # ---------------- COMPUTEs RANGE–ANGLE MAPS ----------------
    for r in selected_bins:
        R_avg, total_snaps = average_covariance_across_cpis(
            filepath,
            cpi_indices=cpi_indices,
            range_bin=int(r),
            clutter_suppress=clutter_suppress,
        )

        est = estimate_num_sources_aic_mdl(R_avg, n_snapshots=total_snaps, max_sources=num_rx - 1)
        if music_source_mode.upper() == "MDL":
            k_src = est["mdl_k"]
        elif music_source_mode.upper() == "AIC":
            k_src = est["aic_k"]
        else:
            k_src = int(music_fixed_sources)

        # With only 3 RX, k can become 0 sometimes; enforce >=1 for practical DOA.
        k_src = int(np.clip(k_src, 1, num_rx - 1))

        maps["Bartlett"][r, :] = bartlett_spectrum(R_avg, A)
        maps["MVDR"][r, :] = mvdr_spectrum(R_avg, A)
        maps["MUSIC"][r, :] = music_spectrum(R_avg, A, num_sources=k_src, fb_avg=music_fb_averaging)


    # ---------------- PLOT RANGE–ANGLE MAPS ----------------
    for name, P in maps.items():
        valid = np.isfinite(P)
        if not np.any(valid):
            print(f"[WARN] No valid bins computed for {name}. Check range gate settings.")
            continue

        # --- define crop indices ---
        rmin = int(np.nanmin(np.where(np.any(valid, axis=1))[0]))
        rmax = int(np.nanmax(np.where(np.any(valid, axis=1))[0]))

        # --- normalizes + dB on the FULL map first (global) ---
        P_max = np.nanmax(P)
        P_norm = P / (P_max + 1e-12)
        P_dB = 10 * np.log10(P_norm + 1e-12)

        # --- crops AFTER dB ---
        P_crop = P_dB[rmin:rmax+1, :]  # shape: [range_crop, angle]
        rng_crop = rng_axis[rmin:rmax+1]

        if smooth_sigma_db and smooth_sigma_db > 0:
            fill_val = np.nanmin(P_crop[np.isfinite(P_crop)])
            P_fill = np.where(np.isfinite(P_crop), P_crop, fill_val)
            P_crop = gaussian_filter(P_fill, sigma=float(smooth_sigma_db))

        plt.figure(figsize=(8, 5))
        plt.imshow(
            P_crop.T,
            aspect="auto",
            origin="lower",
            extent=[rng_crop[0], rng_crop[-1], angle_grid[0], angle_grid[-1]],
            cmap="jet",
            vmin=-dyn_range_db,
            vmax=0,
            interpolation="bicubic",
        )
        plt.colorbar(label="Power (dB)")
        plt.title(f"{name} Beamforming Range–Angle Map (advanced)")
        plt.xlabel("Range (m)")
        plt.ylabel("Angle (°)")
        plt.tight_layout()
        plt.show()


    # ---------------- CHOOSES RANGE BIN FOR BEAMPATTERN/DOA ----------------
    if selected_bins.size == 0:
        selected_bins = np.arange(num_ranges)

    strongest_bin = int(selected_bins[np.argmax(range_powers[selected_bins])])
    print(f"[INFO] Strongest gated range bin: {strongest_bin}  (~{strongest_bin*params['range_res']:.2f} m)")

    R_bin, total_snaps = average_covariance_across_cpis(
        filepath,
        cpi_indices=cpi_indices,
        range_bin=strongest_bin,
        clutter_suppress=clutter_suppress,
    )

    est = estimate_num_sources_aic_mdl(R_bin, n_snapshots=total_snaps, max_sources=num_rx - 1)
    if music_source_mode.upper() == "MDL":
        k_src = est["mdl_k"]
    elif music_source_mode.upper() == "AIC":
        k_src = est["aic_k"]
    else:
        k_src = int(music_fixed_sources)
    k_src = int(np.clip(k_src, 1, num_rx - 1))

    print(f"[INFO] Estimated #sources: AIC={est['aic_k']}  MDL={est['mdl_k']}  -> Using {k_src} ({music_source_mode})")

    # Beampattern overlay
    P_b = bartlett_spectrum(R_bin, A)
    P_m = mvdr_spectrum(R_bin, A)
    P_u = music_spectrum(R_bin, A, num_sources=k_src, fb_avg=music_fb_averaging)

    plt.figure(figsize=(9, 5))
    for label, P in [("Bartlett", P_b), ("MVDR", P_m), (f"MUSIC(k={k_src})", P_u)]:
        P_dB = 10 * np.log10(P / (np.max(P) + 1e-12))
        plt.plot(angle_grid, P_dB, linewidth=1.6, label=label)

    plt.title(f"Beampattern Comparison (Range ~ {strongest_bin*params['range_res']:.2f} m)")
    plt.xlabel("Angle (°)")
    plt.ylabel("Normalized Power (dB)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ---------------- MUSIC PEAK DETECTION ----------------
    P_u_dB = 10 * np.log10(P_u / (np.max(P_u) + 1e-12))
    rel = P_u_dB - np.max(P_u_dB)

    peak_threshold_db = -12
    min_sep_deg = 3.0

    peaks, _ = find_peaks(rel, height=peak_threshold_db)
    peak_angles = []
    for idx in peaks:
        ang = float(angle_grid[idx])
        if all(abs(ang - pa) >= min_sep_deg for pa in peak_angles):
            peak_angles.append(ang)

    if peak_angles:
        print(f"[INFO] MUSIC peaks above {peak_threshold_db} dB (deg): {np.round(peak_angles, 1)}")
    else:
        print(f"[INFO] No MUSIC peaks above {peak_threshold_db} dB at selected range bin.")
