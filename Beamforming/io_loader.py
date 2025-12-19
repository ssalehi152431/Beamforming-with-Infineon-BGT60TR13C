"""
io_loader.py
-------------------------------------------------------
Handles loading Infineon radar .npy data and
computing system parameters from configuration.
-------------------------------------------------------
"""

import numpy as np

# ===============================================================
#                 get_params()
# ===============================================================

def get_params(cfg):
    """
    Computes derived radar parameters from user configuration.
    'cfg' is a dictionary of radar hardware settings.
    The goal is to derive key parameters (bandwidth, slope,
    wavelength, range/velocity resolution, etc.) for signal processing.
    """
    params = {} # directory to store
    c = 3e8  # Speed of light

    # Basic derived parameters
    params["bw"] = cfg["end_frequency_Hz"] - cfg["start_frequency_Hz"] # FMCW bandwidth = f_end - f_start
    params["Tc"] = cfg["chirp_repetition_time_s"] # Chirp duration (time per chirp)
    params["slope"] = params["bw"] / params["Tc"] # Frequency slope (Hz/s): how fast the frequency increases in one chirp
    params["lambda"] = c / ((cfg["start_frequency_Hz"] + cfg["end_frequency_Hz"]) / 2) # Radar wavelength (m)
    params["range_res"] = c / (2 * params["bw"]) # Range resolution (m): smallest distinguishable range difference
    params["max_range"] = params["range_res"] * cfg["num_samples"] / 2 # Maximum measurable range (m): depends on sampling and range resolution
    params["prf"] = 1 / params["Tc"] # Pulse repetition frequency (Hz): how often chirps repeat
    params["max_vel"] = params["lambda"] / (4 * params["Tc"]) # Maximum unambiguous velocity (m/s)
    params["vel_res"] = 2 * params["max_vel"] / cfg["num_chirps"] # Velocity resolution (m/s): smallest distinguishable Doppler difference

    # ----------------- Copying essential hardware settings -----------------
    # These are directly needed for later processing steps
    params["num_rx"] = cfg["num_rx"]
    params["num_chirps"] = cfg["num_chirps"]
    params["num_samples"] = cfg["num_samples"]

    return params


# ===============================================================
#                 load_infineon_recording()
# ===============================================================

def load_infineon_recording(filepath, cpi_idx=0):
    """
    Load a saved Infineon radar .npy recording.
    - Expected file shape: [N_CPI, N_ch, N_chirps, N_samples]
    - Returns one CPI (Coherent Processing Interval):
      frame shape = [channels, chirps, samples]
    """
    data = np.load(filepath)
    print(f"[INFO] Raw data shape: {data.shape}")


    # handles both multi-CPI and single-CPI files
    if data.ndim == 4:
        frame = data[cpi_idx, :, :, :] # If multiple CPIs exist, select the requested one by index
    elif data.ndim == 3:
        frame = data  # If already single CPI, just use it directly
    else:
        raise ValueError("Unexpected data shape for radar.npy")

    print(f"[INFO] Extracted frame shape: {frame.shape} "
          f"(channels x chirps x samples)")

    return frame



# ===============================================================
#                 preprocess_frame()
# ===============================================================
def preprocess_frame(frame):
    """
    Reorders the raw frame and applies range FFT for each RX antenna.
    Output format: [range_bins, chirps, rx_antennas]
    Steps:
        1. Removing DC offset
        2. Applying window (Hanning)
        3. Computing Range FFT
        4. Stacking across antennas and reorder dimensions
    """
    num_rx, num_chirps, num_samples = frame.shape # Unpacking the frame dimensions
    range_window = np.hanning(num_samples) # Creating a Hanning window to reduce spectral leakage in range FFT
    range_fft_cube = [] # Storing per-antenna range FFT results


    # Loop over each receive antenna channel
    for rx in range(num_rx):
        mat = frame[rx, :, :] # Extracting data for one RX: matrix of shape [chirps × samples]
        mat = mat - np.mean(mat, axis=1, keepdims=True) # Removing DC bias (subtract mean per chirp)
        mat = mat * range_window # Applying the Hanning window along the fast-time (sample) axis
        range_fft = np.fft.fft(mat, n=num_samples, axis=1) # Performing Range FFT along the fast-time axis (samples)
        range_fft = range_fft[:, :num_samples // 2] # Keeping only the positive frequency half (physical ranges)
        range_fft_cube.append(range_fft) # Appending processed matrix to list

    cube = np.stack(range_fft_cube, axis=-1) # Stacking all antennas along a new dimension → shape [chirps × range × rx]
    cube = np.transpose(cube, (1, 0, 2))  # Reordering axes to [range, chirp, rx] for later Doppler/Angle processing
    print(f"[INFO] Preprocessed cube shape: {cube.shape}") 
    return cube


# ===============================================================
#                TEST SECTION (your config here)
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
        "tdm_mimo": 0                       
    }

    # Importing functions from same file for standalone testing
    from io_loader import get_params, load_infineon_recording, preprocess_frame

    params = get_params(cfg)
    print("=== Derived Radar Parameters ===")
    for k, v in params.items():
        print(f"{k:20s}: {v}")

    frame = load_infineon_recording("radar.npy", cpi_idx=0)   # Loading one CPI frame from the .npy file
    cube = preprocess_frame(frame) # Applying range FFT and reorganize dimensions for further processing