"""
beamformers/bartlett.py
-------------------------------------------------------
Implements the classical Bartlett (delay-and-sum) beamformer
for Infineon BGT60TR13C radar data.
-------------------------------------------------------
"""

import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # make io_loader visible

from io_loader import load_infineon_recording, preprocess_frame, get_params
from array_geometry import steering_vector


def bartlett_spectrum(R, A):
    """
    Compute Bartlett spatial spectrum for a given covariance matrix.

    R : ndarray [num_rx, num_rx]
        Spatial covariance matrix.
    A : ndarray [num_rx, num_angles]
        Steering matrix (each column = steering vector for that angle).

    Returns
    -------
    P : ndarray [num_angles]
        Bartlett power spectrum.
    """
    num_angles = A.shape[1]
    P = np.zeros(num_angles)
    for i in range(num_angles):
        a = A[:, i].reshape(-1, 1)
        # Classical Bartlett power = aᴴ R a
        P[i] = np.real(a.conj().T @ R @ a).item()
    return P.squeeze()


# ===============================================================
#                   MAIN TEST SECTION
# ===============================================================
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

    # ----------- 2. Load & preprocess data -----------
    params = get_params(cfg)
    data = load_infineon_recording("radar.npy", cpi_idx=0)
    cube = preprocess_frame(data)  # [range_bins, chirps, rx_antennas]
    num_ranges, num_chirps, num_rx = cube.shape
    print(f"[INFO] Data cube shape: {cube.shape}")

    # ----------- 3. Steering matrix -----------
    angle_grid = np.linspace(-60, 60, 241)  # fine angular grid
    A = steering_vector(params, angle_grid, d_by_lambda=0.5)

    # ----------- 4. Bartlett per range -----------
    P_range_angle = np.zeros((num_ranges, len(angle_grid)))
    for r in range(num_ranges):
        X = cube[r, :, :]                    # [chirps, rx]
        X = X * np.hanning(X.shape[0])[:, None]   # window across chirps
        R = np.cov(X, rowvar=False)          # spatial covariance
        P_range_angle[r, :] = bartlett_spectrum(R, A)

    # ----------- 5. Normalize and convert to dB -----------
    eps = 1e-12
    P_norm = P_range_angle / (np.max(P_range_angle) + eps)
    P_dB = 10 * np.log10(P_norm + eps)
    rng_axis = np.arange(num_ranges) * params["range_res"]

    print(f"Power range: {np.min(P_dB):.2f}  {np.max(P_dB):.2f}")

    # ----------- 6. Plot Range–Angle Map -----------
    plt.figure(figsize=(10, 5))
    plt.imshow(
        P_dB.T,
        aspect="auto",
        origin="lower",
        extent=[rng_axis[0], rng_axis[-1], angle_grid[0], angle_grid[-1]],
        cmap="jet_r",
        vmin=-80,
        vmax=0,
    )
    plt.colorbar(label="Power (dB)")
    plt.xlabel("Range (m)")
    plt.ylabel("Angle (°)")
    plt.title("Bartlett Beamforming Range–Angle Map")
    plt.tight_layout()
    plt.show()

    # ----------- 7. Auto-select strongest reflection (skip near bins) -----------
    min_range_m = 0.15  # ignore first 15 cm
    start_bin = int(np.ceil(min_range_m / params["range_res"]))
    range_powers = np.sum(np.abs(cube)**2, axis=(1, 2))
    range_bin = start_bin + int(np.argmax(range_powers[start_bin:]))

    print(f"[INFO] Auto-selected range bin: {range_bin} (~{range_bin*params['range_res']:.2f} m)")

    # ----------- 8. Beampattern for selected bin -----------
    R_sel = np.cov(cube[range_bin, :, :], rowvar=False)
    P_theta = np.array([np.real(a.conj().T @ R_sel @ a) for a in A.T])
    P_theta_dB = 10 * np.log10(P_theta / (np.max(P_theta) + eps))

    plt.figure(figsize=(8, 4))
    plt.plot(angle_grid, P_theta_dB, 'r', linewidth=1)
    plt.title(f"Measured Beampattern at Range {range_bin} (~{range_bin*params['range_res']:.2f} m)")
    plt.xlabel("Angle (°)")
    plt.ylabel("Power (dB)")
    plt.grid(True)
    plt.show()
