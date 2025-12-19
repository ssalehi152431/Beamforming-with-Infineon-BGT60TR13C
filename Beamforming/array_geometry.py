"""
array_geometry.py
-------------------------------------------------------
Generates steering vectors for a uniform linear array (ULA)
-------------------------------------------------------
"""

import numpy as np
import matplotlib.pyplot as plt

# ===============================================================
#               steering_vector()
# ===============================================================
def steering_vector(params, angle_grid_deg, d_by_lambda=0.5):
    """
    Computes the steering vector matrix for a ULA (Uniform Linear Array).

    Parameters
    ----------
    params : dict
        Dictionary of radar parameters (from get_params in io_loader.py)
        Must include at least:
            - num_rx   → number of receive antennas
            - lambda   → radar wavelength
    angle_grid_deg : array-like
        List/array of angles (in degrees) over which to compute steering vectors.
        Example: np.linspace(-90, 90, 181)
    d_by_lambda : float, optional
        Spacing between adjacent antenna elements in units of wavelength (λ).
        Default = 0.5 λ → standard half-wavelength spacing to avoid grating lobes.

    Returns
    -------
    A : ndarray
        Complex steering matrix of shape [num_rx, num_angles].
        Each column A[:, i] is the steering vector for angle_grid_deg[i].
    """
    num_rx = params["num_rx"] # Number of receiving antennas (array elements)
    wavelength = params["lambda"] # Radar wavelength (from system parameters)
    d = d_by_lambda * wavelength # Physical distance (spacing) between adjacent antenna elements

    angle_grid_rad = np.deg2rad(angle_grid_deg) # Converting the input angle grid from degrees → radians for trigonometric functions
    A = np.zeros((num_rx, len(angle_grid_rad)), dtype=complex) # Initializing steering matrix with zeros (complex type)

    # -----------------------------------------------------------
    # Element indices: defines relative antenna positions.
    # For example, if num_rx = 3 → indices = [-1, 0, 1]
    # This centers the array around the midpoint (broadside = 0°)
    # -----------------------------------------------------------
    element_idx = np.arange(num_rx) - (num_rx - 1) / 2 


    # Loop through all scan angles (θ) to compute steering vectors
    for i, theta in enumerate(angle_grid_rad):
        # Computes phase shift for each element at this θ:
        # φ = (2π / λ) * d * sin(θ) * element_index
        # This represents the phase difference of the received signal
        # across antennas when a plane wave arrives at angle θ.
        phase_shift = 2 * np.pi * element_idx * d * np.sin(theta) / wavelength
        A[:, i] = np.exp(1j * phase_shift) # Each element contributes e^(jφ) → complex exponential form

    return A


# ===============================================================
#                     TEST SECTION (USING REAL DATA)
# ===============================================================
if __name__ == "__main__":
    import numpy as np
    from io_loader import get_params, load_infineon_recording

    print("[INFO] Loading real Infineon radar configuration...")

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
        "tdm_mimo": 0                       
    }

    params = get_params(cfg)

    # Loading one frame to verify consistency
    frame = load_infineon_recording("radar.npy", cpi_idx=0)
    print(f"[INFO] Loaded frame shape: {frame.shape} (channels × chirps × samples)")

    # Computing steering vectors based on actual λ and num_rx
    # -----------------------------------------------------------
    # Generates steering vectors across -90° to +90° in 1° steps
    # -----------------------------------------------------------
    angle_grid = np.linspace(-90, 90, 181)
    A = steering_vector(params, angle_grid, d_by_lambda=0.5)
    print(f"[INFO] Steering matrix shape: {A.shape}") #(3, 181)

    # Plotting |a(θ)| for all RX antennas
    import matplotlib.pyplot as plt


    # -----------------------------------------------------------
    # Visualizing |a(θ)| for each antenna element
    # -----------------------------------------------------------
    plt.figure(figsize=(7, 4))
    for rx in range(params["num_rx"]):
        plt.plot(angle_grid, np.abs(A[rx, :]), label=f"RX{rx+1}")
    plt.xlabel("Angle (degrees)")
    plt.ylabel("|a(θ)|")
    plt.title("Steering Vector Amplitude (Based on Infineon Radar Config)")
    plt.legend()
    plt.grid(True)
    plt.show()


    # -----------------------------------------------------------
    # Additional Visualization: Steering Vector Phase vs Angle
    # -----------------------------------------------------------

    plt.figure(figsize=(8, 4))
    for rx in range(params["num_rx"]):
        # Plot the phase (in radians) for each RX antenna
        plt.plot(angle_grid, np.angle(A[rx, :]), label=f"RX{rx+1}")

    plt.xlabel("Angle (degrees)")
    plt.ylabel("Phase (radians)")
    plt.title("Steering Vector Phase (Based on Infineon Radar Config)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

