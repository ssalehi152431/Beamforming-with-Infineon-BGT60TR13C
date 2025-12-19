"""
beamformers/bartlett.py
-------------------------------------------------------
Implements the classical Bartlett (delay-and-sum) beamformer
for Infineon radar data.
-------------------------------------------------------
"""

import numpy as np
import matplotlib.pyplot as plt
from io_loader import load_infineon_recording, preprocess_frame, get_params
from array_geometry import steering_vector




# ===============================================================
#              bartlett_spectrum()
# ===============================================================
def bartlett_spectrum(R, A):
    """
    Compute Bartlett spatial power spectrum for a given covariance matrix.

    Parameters
    ----------
    R : ndarray
        Covariance matrix of shape [num_rx, num_rx].
        Represents the spatial correlation between receive antennas.
    A : ndarray
        Steering matrix of shape [num_rx, num_angles].
        Each column of A is the steering vector for one scan angle.

    Returns
    -------
    P : ndarray
        Power spectrum [num_angles].
        Bartlett power for each scan direction.
    """
    num_angles = A.shape[1] # Number of test angles in the scan grid
    P = np.zeros(num_angles)  # Initializing power array


    # Loop over each steering direction
    for i in range(num_angles):
        a = A[:, i].reshape(-1, 1) # Steering vector for current angle θ
        P[i] = np.real(a.conj().T @ R @ a).item() 
    return P.squeeze() # Return as 1D vector


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
    data = load_infineon_recording("radar.npy", cpi_idx=0) # Loads first CPI
    cube = preprocess_frame(data)  #range_fft- [range, chirps, rx]
    num_ranges, num_chirps, num_rx = cube.shape # extracting dimensions
    print(f"[INFO] Data cube shape: {cube.shape}") 

    # ----------- Prepares steering matrix -----------
    angle_grid = np.linspace(-60, 60, 241) # defining angle grid for azimuth scan (−60° to +60° with 0.5° spacing)
    A = steering_vector(params, angle_grid, d_by_lambda=0.5) # steering vector calculation

    # ----------- Computes Bartlett for each range bin -----------
    P_range_angle = np.zeros((num_ranges, len(angle_grid)))  # Initializing range–angle power map [range_bins × angles]


    #Loops over every range bin
    for r in range(num_ranges): 
        X = cube[r, :, :]  # extracting data for this range [chirps, rx]
        X = X * np.hanning(X.shape[0])[:, None]   # Applying Hanning window across chirps
        R = np.cov(X, rowvar=False) # Estimating RX spatial covariance matrix
        P_range_angle[r, :] = bartlett_spectrum(R, A) # Computes Bartlett spectrum for this range
        # P_range_angle[r, :] = P_range_angle[r, :] / np.max(P_range_angle[r, :])


    # ----------- Normalize and plot -----------
    eps = 1e-12 # avoiding divide by zero error
    P_norm = P_range_angle / (np.max(P_range_angle) + eps) #normalization
    P_dB = 10 * np.log10(P_norm + eps) # to dB

    rng_axis = np.arange(num_ranges) * params["range_res"] # range axis in meter
    print("Power range:", np.min(P_dB), np.max(P_dB))


    # range-Angle Map
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


    # # ----------- 6. Optional: plot beampattern for a chosen range -----------
    # chosen_range = 10  # index of range bin to visualize
    # X = cube[chosen_range, :, :]  # [chirps, rx]
    # R = np.cov(X, rowvar=False)
    # P_beam = bartlett_spectrum(R, A)
    # P_beam_dB = 10 * np.log10(P_beam / np.max(P_beam))

    # plt.figure(figsize=(6, 4))
    # plt.plot(angle_grid, P_beam_dB)
    # plt.title(f"Bartlett Beampattern at Range Bin {chosen_range}")
    # plt.xlabel("Angle (°)")
    # plt.ylabel("Normalized Power (dB)")
    # plt.grid(True)
    # plt.show()

    # # ========== ADD BELOW for data-driven beampattern ==========
    # print("\n[INFO] Plotting measured beampattern from data...")

    # # Compute steering vectors and measured covariance
    # from array_geometry import steering_vector
    # A = steering_vector(params, angle_grid)
    # range_bin = 10  # pick one range bin near the detected object
    # R = np.cov(cube[range_bin, :, :], rowvar=False)

    # # Power vs angle
    # P_theta = np.array([np.real(a.conj().T @ R @ a) for a in A.T])
    # P_dB = 10 * np.log10(P_theta / np.max(P_theta))

    # plt.figure(figsize=(8, 4))
    # plt.plot(angle_grid, P_dB, 'r', linewidth=1)
    # plt.title(f"Measured Beampattern at Range Bin {range_bin}")
    # plt.xlabel("Angle (°)")
    # plt.ylabel("Power (dB)")
    # plt.grid(True)
    # plt.show()


    # ----------- PLOTTING BEAMPATTERN AT STRONGEST RANGE BIN -----------
    print("\n[INFO] Plotting measured beampattern from data...")

    # --- Auto-detect strongest reflection ---
    range_powers = np.sum(np.abs(cube)**2, axis=(1, 2))
    range_bin = int(np.argmax(range_powers))
    print(f"[INFO] Automatically selected range bin: {range_bin} (strongest reflection)")

    # ReComputing steering vectors and measured covariance
    from array_geometry import steering_vector
    A = steering_vector(params, angle_grid)
    R = np.cov(cube[range_bin, :, :], rowvar=False)

    # Computing Bartlett power vs. angle for selected range
    P_theta = np.array([np.real(a.conj().T @ R @ a) for a in A.T])
    P_dB = 10 * np.log10(P_theta / np.max(P_theta))

    plt.figure(figsize=(8, 4))
    plt.plot(angle_grid, P_dB, 'r', linewidth=1)
    plt.title(f"Measured Beampattern at Range Bin {range_bin}")
    plt.xlabel("Angle (°)")
    plt.ylabel("Power (dB)")
    plt.grid(True)
    plt.show()

