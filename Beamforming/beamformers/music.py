"""
beamformers/music.py
-------------------------------------------------------
Implements the high-resolution MUSIC (Multiple Signal Classification)
beamformer for Infineon radar data.
-------------------------------------------------------
"""

import numpy as np
import matplotlib.pyplot as plt
from io_loader import load_infineon_recording, preprocess_frame, get_params
from array_geometry import steering_vector


def music_spectrum(R, A, num_sources=1, eps=1e-6):
    """
    Computes MUSIC spatial spectrum for a given covariance matrix.

    Parameters
    ----------
    R : ndarray
        Covariance matrix of shape [num_rx, num_rx]
    A : ndarray
        Steering matrix [num_rx, num_angles]
    num_sources : int
        Number of signal sources (assumed)
    eps : float
        Regularization constant for numerical stability

    Returns
    -------
    P : ndarray
        MUSIC pseudo-spectrum [num_angles]
    """
    num_rx = R.shape[0]

    # Regularizing covariance
    R = R + eps * np.eye(num_rx)

    # Eigen-decomposition
    eigvals, eigvecs = np.linalg.eigh(R)
    idx = np.argsort(eigvals)[::-1]       # sort descending
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Noise subspace
    E_n = eigvecs[:, num_sources:]

    # Computes MUSIC pseudo-spectrum
    P = np.zeros(A.shape[1])
    for i in range(A.shape[1]):
        a = A[:, i].reshape(-1, 1)
        denom = np.linalg.norm(E_n.conj().T @ a) ** 2
        P[i] = 1.0 / denom if denom > 0 else 0.0

    return P


# ===============================================================
#                 TEST SECTION: Range–Angle Map
# ===============================================================
if __name__ == "__main__":

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

    # ----------- Steering matrix -----------
    angle_grid = np.linspace(-60, 60, 241)
    A = steering_vector(params, angle_grid, d_by_lambda=0.5)

    # ----------- MUSIC for each range bin -----------
    P_range_angle = np.zeros((num_ranges, len(angle_grid)))

    for r in range(num_ranges):
        X = cube[r, :, :]                 # [chirps, rx]
        X = X * np.hanning(X.shape[0])[:, None]
        R = np.cov(X, rowvar=False)
        P = music_spectrum(R, A, num_sources=1)
        P_range_angle[r, :] = P / (np.max(P) + 1e-12)

    # ----------- Normalization and plotting -----------
    P_dB = 10 * np.log10(P_range_angle + 1e-12)
    rng_axis = np.arange(num_ranges) * params["range_res"]

    plt.figure(figsize=(10, 5))
    plt.imshow(
        P_dB.T,
        aspect="auto",
        origin="lower",
        extent=[rng_axis[0], rng_axis[-1], angle_grid[0], angle_grid[-1]],
        cmap="jet_r",
        vmin=-50,
        vmax=0,
    )
    plt.colorbar(label="Power (dB)")
    plt.xlabel("Range (m)")
    plt.ylabel("Angle (°)")
    plt.title("MUSIC Beamforming Range–Angle Map")
    plt.tight_layout()
    plt.show()

    # ----------- Auto select range bin & plotting beampattern -----------
    print("\n[INFO] Plotting MUSIC beampattern from data...")

    range_powers = np.sum(np.abs(cube) ** 2, axis=(1, 2))
    range_bin = int(np.argmax(range_powers))
    print(f"[INFO] Automatically selected range bin: {range_bin}")

    R = np.cov(cube[range_bin, :, :], rowvar=False)
    P_theta = music_spectrum(R, A, num_sources=1)
    P_dB = 10 * np.log10(P_theta / np.max(P_theta))

    plt.figure(figsize=(8, 4))
    plt.plot(angle_grid, P_dB, "r", linewidth=1)
    plt.title(f"MUSIC Measured Beampattern at Range Bin {range_bin}")
    plt.xlabel("Angle (°)")
    plt.ylabel("Power (dB)")
    plt.grid(True)
    plt.show()
