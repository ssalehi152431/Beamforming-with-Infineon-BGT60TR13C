"""
beamformers/mvdr.py
-------------------------------------------------------
Implements the Minimum Variance Distortionless Response (MVDR)
beamformer for Infineon BGT60TR13C radar data.
Also known as the Capon beamformer, it minimizes output noise
power while keeping unity gain in the look direction.
-------------------------------------------------------
"""

import numpy as np
import matplotlib.pyplot as plt
from io_loader import load_infineon_recording, preprocess_frame, get_params
from array_geometry import steering_vector


def mvdr_spectrum(R, A, eps=1e-6):
    """
    Computes MVDR (Capon) spatial spectrum for a given covariance matrix.

    Parameters
    ----------
    R : ndarray [num_rx, num_rx]
        Spatial covariance matrix.
    A : ndarray [num_rx, num_angles]
        Steering matrix (each column corresponds to a direction).
    eps : float
        Regularization constant for numerical stability (avoids singularities).

    Returns
    -------
    P : ndarray [num_angles]
        Computed MVDR power spectrum (higher = potential target direction).
    """

    num_angles = A.shape[1] # number of look directions
    num_rx = R.shape[0] # number of receiver elements
    P = np.zeros(num_angles) # Initializing empty power spectrum

    # Regularize covariance to prevent singularities
    R_inv = np.linalg.inv(R + eps * np.eye(num_rx)) 


    # Loop through all look directions
    for i in range(num_angles):
        a = A[:, i].reshape(-1, 1) # Extracting steering vector for current angle
        denom = np.real(a.conj().T @ R_inv @ a)  # Computing denominator aᴴ R⁻¹ a (complex quadratic form)
        denom_val = np.real(denom).item()  # Extract scalar value from 1x1 matrix
        P[i] = 1.0 / denom_val if denom_val > 0 else 0.0  # MVDR spectrum value = 1 / (aᴴ R⁻¹ a)

    return P.squeeze() # Return final spatial power spectrum


# ===============================================================
#                   TEST SECTION: Range–Angle Map
# ===============================================================
if __name__ == "__main__":

    # ----------- Configuration -----------
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
    data = load_infineon_recording("radar.npy", cpi_idx=0) # Load radar recording (.npy) and select one CPI frame
    cube = preprocess_frame(data)  # [range, chirps, rx] 
    num_ranges, num_chirps, num_rx = cube.shape # applying range FFT to obtain [range_bins, chirps, rx_antennas]
    print(f"[INFO] Data cube shape: {cube.shape}") 

    # ----------- Prepares steering matrix -----------
    angle_grid = np.linspace(-60, 60, 241)
    A = steering_vector(params, angle_grid, d_by_lambda=0.5) # Computing corresponding steering vectors for all angles

    # ----------- Computes MVDR for each range bin -----------
    P_range_angle = np.zeros((num_ranges, len(angle_grid)))  # Initializing output 2D Range–Angle power matrix

    for r in range(num_ranges):
        # Extracting data for the r-th range bin [chirps × rx]
        X = cube[r, :, :]  # [chirps, rx] 
        X = X * np.hanning(X.shape[0])[:, None]  # Apply Hanning window
        R = np.cov(X, rowvar=False)  # Compute covariance matrix across RX channels
        P_range_angle[r, :] = mvdr_spectrum(R, A) # Compute MVDR spatial spectrum for this range bin

    # ----------- Normalization and plotting -----------
    eps = 1e-12
    P_norm = P_range_angle / (np.max(P_range_angle) + eps) # normalization
    P_dB = 10 * np.log10(P_norm + eps) # to dB

    rng_axis = np.arange(num_ranges) * params["range_res"]  #range in meters

    plt.figure(figsize=(10, 5)) 
    plt.imshow( 
        P_dB.T,
        aspect="auto",
        origin="lower",
        extent=[rng_axis[0], rng_axis[-1], angle_grid[0], angle_grid[-1]],
        cmap="jet_r",
        vmin=-30,
        vmax=0,
    )
    plt.colorbar(label="Power (dB)")
    plt.xlabel("Range (m)")
    plt.ylabel("Angle (°)")
    plt.title("MVDR (Capon) Beamforming Range–Angle Map")
    plt.tight_layout()
    plt.show()


    print("\n[INFO] Plotting measured beampattern from data...")

    # --- Auto-detect strongest reflection ---
    range_powers = np.sum(np.abs(cube) ** 2, axis=(1, 2))
    range_bin = int(np.argmax(range_powers))
    print(f"[INFO] Automatically selected range bin: {range_bin} (strongest reflection)")

    R = np.cov(cube[range_bin, :, :], rowvar=False)
    P_theta = mvdr_spectrum(R, A)
    P_dB = 10 * np.log10(P_theta / np.max(P_theta))

    plt.figure(figsize=(8, 4))
    plt.plot(angle_grid, P_dB, "r", linewidth=1)
    plt.title(f"MVDR Measured Beampattern at Range Bin {range_bin}")
    plt.xlabel("Angle (°)")
    plt.ylabel("Power (dB)")
    plt.grid(True)
    plt.show()
