"""
FSAE IC CAN Bus Data Simulator
------------------------------
Generates 1kHz CAN bus data for a Formula Student (SUPRA 2025) Internal Combustion vehicle.
Includes data for low-voltage cell voltages (<60V), temperatures, currents, and wheel speeds.

Noise & EMI Modeling (per SUPRA 2025 A1.2 Rules):
A1.2 references high-vibration, high-noise environments typical of FSAE prototypes.
The simulation injects:
1. EMI/Ignition System Noise: Heavy-tailed high-frequency spikes on electrical lines using PyTorch Laplace distributions.
2. Vibration-induced Intermittency: Connector dropouts and sudden sensor signal loss (0 values) due to severe vibration.
3. Gaussian White Noise: Standard thermal and ADC quantization / sensor inaccuracy noise.
"""

import os
import numpy as np
import pandas as pd
import torch

def generate_fsae_can_data(duration_sec=60, sample_rate_hz=1000, output_path="can_training_data.csv"):
    print(f"Starting simulation for {duration_sec}s at {sample_rate_hz}Hz...")
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    num_samples = duration_sec * sample_rate_hz
    time = np.linspace(0, duration_sec, num_samples)
    
    # ---------------------------------------------------------
    # 1. Base Physical Models (Clean Signals)
    # ---------------------------------------------------------
    
    # Engine RPM: smoothly varying using a random walk + sine wave
    # Simulates shifting gears, acceleration and braking
    rpm_base = 3500 + 2000 * np.sin(2 * np.pi * 0.05 * time) + np.cumsum(np.random.normal(0, 5, num_samples))
    rpm_base = np.clip(rpm_base, 1500, 11000)  # Typical FSAE engine limits (e.g. CBR600, KTM390)
    
    # Wheel Speeds (km/h)
    gear_ratio = 3.5
    tire_radius = 0.25 # meters (approx 10 inches)
    ws_base = (rpm_base / gear_ratio) * (2 * np.pi * tire_radius) / 60 * 3.6 # km/h
    
    # Slight variations per wheel, rear wheels slightly faster to simulate slip
    ws_fl = ws_base * np.random.normal(1.0, 0.005, num_samples)
    ws_fr = ws_base * np.random.normal(1.0, 0.005, num_samples)
    ws_rl = ws_base * np.random.normal(1.05, 0.02, num_samples) 
    ws_rr = ws_base * np.random.normal(1.05, 0.02, num_samples)
    
    # Cell Voltages (12x Li-ion cells in series, nominal ~44V to 50V total, staying well below <60V LV limit)
    # Voltage drops slightly when RPM/load increases
    voltage_drop = (rpm_base - 1500) / 10000 * 0.15
    cells = []
    for i in range(12):
        # Base voltage per cell with normal sensor noise
        cell_v = 4.15 - voltage_drop + np.random.normal(0, 0.002, num_samples)
        cells.append(cell_v)
    cells = np.array(cells)
    total_voltage = np.sum(cells, axis=0)
    
    # LV Current (Amps) - proportional to RPM (fuel pump + ignition + cooling fans)
    current = 8.0 + (rpm_base / 1000) * 1.2 + np.random.normal(0, 0.2, num_samples)
    
    # Temperatures (Celsius) - slowly increasing due to runtime
    eng_temp = 85.0 + 0.05 * time + np.random.normal(0, 0.1, num_samples)
    ambient_temp = np.full(num_samples, 35.0) + np.random.normal(0, 0.05, num_samples)
    
    # ---------------------------------------------------------
    # 2. Noise & EMI Injection (SUPRA 2025 A1.2 High-Vibration)
    # ---------------------------------------------------------
    
    # Convert some signals to PyTorch tensors to utilize specific distribution sampling for EMI
    current_tensor = torch.tensor(current, dtype=torch.float32)
    voltage_tensor = torch.tensor(total_voltage, dtype=torch.float32)
    
    # EMI Noise (Spark/Ignition System spikes on electrical lines):
    # Laplace distribution is often used to model heavy-tailed impulse noise common in automotive electrical systems.
    emi_noise_dist = torch.distributions.laplace.Laplace(loc=0.0, scale=0.8)
    current_emi_noise = emi_noise_dist.sample((num_samples,))
    voltage_emi_noise = emi_noise_dist.sample((num_samples,)) * 0.15
    
    # Apply EMI
    current_noisy = (current_tensor + current_emi_noise).numpy()
    total_voltage_noisy = (voltage_tensor + voltage_emi_noise).numpy()
    
    # Vibration / Connector Dropouts (High vibration environment per A1.2):
    # Simulated by sudden drops to 0 or previous value hold. Here we simulate intermittent 0-reads
    # (e.g., loose hall effect wheel speed sensor connectors).
    dropout_prob = 0.0005 # 0.05% chance of dropout per sample due to severe chassis vibration
    
    # Bernoulli sampling for dropouts (1=connected, 0=dropped)
    connection_mask_fl = np.random.choice([1.0, 0.0], size=num_samples, p=[1-dropout_prob, dropout_prob])
    connection_mask_fr = np.random.choice([1.0, 0.0], size=num_samples, p=[1-dropout_prob, dropout_prob])
    
    ws_fl_noisy = ws_fl * connection_mask_fl
    ws_fr_noisy = ws_fr * connection_mask_fr
    
    # Sensor noise on temperature mappings occasionally throwing absurd values due to cold solder joints
    temp_spike_mask = np.random.choice([0.0, 1.0], size=num_samples, p=[0.9995, 0.0005])
    eng_temp_noisy = eng_temp + (temp_spike_mask * np.random.normal(50, 10, num_samples))
    
    # ---------------------------------------------------------
    # 3. Construct Final DataFrame and Verify
    # ---------------------------------------------------------
    
    df_dict = {
        'time_s': np.round(time, 4),
        'engine_rpm': np.round(rpm_base, 1),
        'wheelspeed_fl': np.round(ws_fl_noisy, 2),
        'wheelspeed_fr': np.round(ws_fr_noisy, 2),
        'wheelspeed_rl': np.round(ws_rl, 2),   # Rear left assumed fully secured
        'wheelspeed_rr': np.round(ws_rr, 2),   # Rear right assumed fully secured
        'total_voltage': np.round(total_voltage_noisy, 3),
        'lv_current_a': np.round(current_noisy, 2),
        'engine_temp_c': np.round(eng_temp_noisy, 1),
        'ambient_temp_c': np.round(ambient_temp, 1)
    }
    
    # Add individual cell data
    for i in range(12):
        df_dict[f'cell_{i+1}_v'] = np.round(cells[i], 3)
        
    df = pd.DataFrame(df_dict)
    
    # Save Output CSV
    print("Writing CAN simulation data to CSV format suitable for training...")
    df.to_csv(output_path, index=False)
    
    print(f"Finished! Generated {num_samples} records of synthetic CAN bus data.")
    print(f"Output saved to: {os.path.abspath(output_path)}")
    print("Columns include:", ", ".join(df.columns[:8]) + "...")
    
    return df

if __name__ == "__main__":
    generate_fsae_can_data(duration_sec=60, sample_rate_hz=1000, output_path="can_training_data.csv")
