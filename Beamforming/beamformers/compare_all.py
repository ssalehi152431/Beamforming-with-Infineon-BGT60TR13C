"""
beamformers/compare_all.py
-------------------------------------------------------
Compares Bartlett, MVDR, and MUSIC beamformers on the same
Infineon radar dataset.  Produces side-by-side range–angle maps
and beampattern overlays for direct visual comparison.
-------------------------------------------------------
"""

import numpy as np
import matplotlib.pyplot as plt
from io_loader import load_infineon_recording, preprocess_frame, get_params
from array_geometry import steering_vector
from scipy.ndimage import gaussian_filter

#  Helper functions
def bartlett_spectrum(R, A):
    P = np.zeros(A.shape[1])
    for i in range(A.shape[1]):
        a = A[:, i].reshape(-1, 1)
        P[i] = np.real(a.conj().T @ R @ a)
    return P.squeeze()


def mvdr_spectrum(R, A, eps=1e-6):
    num_rx = R.shape[0]
    R = R + eps * np.eye(num_rx)
    R_inv = np.linalg.inv(R)
    P = np.zeros(A.shape[1])
    for i in range(A.shape[1]):
        a = A[:, i].reshape(-1, 1)
        denom = np.real(a.conj().T @ R_inv @ a)
        P[i] = 1.0 / denom if denom > 0 else 0.0
    return P.squeeze()


def music_spectrum(R, A, num_sources=1, eps=1e-6):
    num_rx = R.shape[0]
    R = R + eps * np.eye(num_rx)
    eigvals, eigvecs = np.linalg.eigh(R)
    idx = np.argsort(eigvals)[::-1]
    E_n = eigvecs[:, idx[num_sources:]]
    P = np.zeros(A.shape[1])
    for i in range(A.shape[1]):
        a = A[:, i].reshape(-1, 1)
        denom = np.linalg.norm(E_n.conj().T @ a) ** 2
        P[i] = 1.0 / denom if denom > 0 else 0.0
    return P.squeeze()


#  MAIN
if __name__ == "__main__":
    # ----------- 1. Configuration -----------
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

    # ----------- Loads & preprocesses data -----------
    params = get_params(cfg)
    data = load_infineon_recording("radar.npy", cpi_idx=0)
    cube = preprocess_frame(data)  # [range, chirps, rx]
    num_ranges, num_chirps, num_rx = cube.shape
    print(f"[INFO] Data cube shape: {cube.shape}")

    # ----------- Prepares steering matrix -----------
    angle_grid = np.linspace(-80, 80, 721)
    A = steering_vector(params, angle_grid, d_by_lambda=0.5)

    # ----------- beamformers -----------
    beamformers = {
        "Bartlett": np.zeros((num_ranges, len(angle_grid))),
        "MVDR": np.zeros((num_ranges, len(angle_grid))),
        "MUSIC": np.zeros((num_ranges, len(angle_grid))),
    }

    for r in range(num_ranges):
        X = cube[r, :, :]
        X = X * np.hanning(X.shape[0])[:, None]
        R = np.cov(X, rowvar=False)

        beamformers["Bartlett"][r, :] = bartlett_spectrum(R, A)
        beamformers["MVDR"][r, :] = mvdr_spectrum(R, A)
        beamformers["MUSIC"][r, :] = music_spectrum(R, A, num_sources=1)

    rng_axis = np.arange(num_ranges) * params["range_res"]


    # -----------Normalization and plotting Range–Angle maps -----------
    for name, P in beamformers.items():
        P = P / (np.max(P) + 1e-12)
        P_dB = 10 * np.log10(P + 1e-12)
        P_dB = gaussian_filter(P_dB, sigma=0.4)

        plt.figure(figsize=(8, 5))
        im = plt.imshow(
            P_dB.T,
            aspect="auto",
            origin="lower",
            extent=[rng_axis[0], rng_axis[-1], angle_grid[0], angle_grid[-1]],
            cmap="jet",
            vmin=-50,
            vmax=0,
        )
        plt.colorbar(label="Power (dB)")
        plt.title(f"{name} Beamforming Range–Angle Map")
        plt.xlabel("Range (m)")
        plt.ylabel("Angle (°)")
        plt.tight_layout()
        plt.show()


    # --- range bin selection ---
    range_powers = np.sum(np.abs(cube) ** 2, axis=(1, 2))
    print("[DEBUG] Range bin energies:", np.round(range_powers, 2))

    # Automatically pick the strongest, but we can override:
    auto_bin = int(np.argmax(range_powers))
    range_bin = 1     # manually pick 
    print(f"[INFO] Forcing range bin {range_bin} (auto strongest was {auto_bin})")


    plt.figure(figsize=(8, 5))
    for name, func in zip(
        ["Bartlett", "MVDR", "MUSIC"],
        [bartlett_spectrum, mvdr_spectrum, music_spectrum],
    ):
        X = cube[range_bin, :, :]
        R = np.cov(X, rowvar=False)
        P = func(R, A)
        P_dB = 10 * np.log10(P / (np.max(P) + 1e-12))
        plt.plot(angle_grid, P_dB, linewidth=1.5, label=name)

    plt.title(f"Beampattern Comparison at Range Bin {range_bin}")
    plt.xlabel("Angle (°)")
    plt.ylabel("Power (dB)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


    # ===============================================================
    # MUSIC peak detection helper
    # ===============================================================
    from scipy.signal import find_peaks

    def find_music_peaks(P_dB, angle_grid, threshold_db=-12, min_separation_deg=3.0):
        """Finding peaks above relative dB threshold in MUSIC spectrum."""
        rel_power = P_dB - np.max(P_dB)
        peaks, props = find_peaks(rel_power, height=threshold_db)

        peak_angles, peak_heights = [], []
        for idx in peaks:
            ang = angle_grid[idx]
            h = rel_power[idx]
            # Skip peaks too close to each other
            if all(abs(ang - pa) >= min_separation_deg for pa in peak_angles):
                peak_angles.append(ang)
                peak_heights.append(h)
        return np.array(peak_angles), np.array(peak_heights)


    # ===============================================================
    # MUSIC analysis for the chosen range bin
    # ===============================================================
    print("\n[INFO] Running MUSIC DOA estimation...")
    X = cube[range_bin, :, :]
    R = np.cov(X, rowvar=False)

    P = music_spectrum(R, A, num_sources=2)  # allow up to 2 sources
    P_dB = 10 * np.log10(P / (np.max(P) + 1e-12))

    angles, heights = find_music_peaks(P_dB, angle_grid, threshold_db=-12, min_separation_deg=3.0)

    if len(angles) > 0:
        print(f"[INFO] MUSIC detected peaks (deg): {np.round(angles,1)}")
    else:
        print("[INFO] No significant MUSIC peaks detected above threshold.")
