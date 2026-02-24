import os
import time
import torch
import torch.nn as nn
from cryptography.fernet import Fernet
import numpy as np

# --- Dummy Model Architecture for demonstration ---
class EdgeModelV1(nn.Module):
    def __init__(self):
        super(EdgeModelV1, self).__init__()
        self.fc = nn.Linear(10, 1)
        self.version = "1.0.0-Base"
    def forward(self, x):
        return self.fc(x)

class EdgeModelV2(nn.Module):
    def __init__(self):
        super(EdgeModelV2, self).__init__()
        self.fc = nn.Linear(10, 1)
        self.version = "2.1.0-Aggressive-Tune"
    def forward(self, x):
        return self.fc(x)

# --- OTA Encryption Setup ---
# Pre-shared AES Key between Pit Wall and Car
SECRET_KEY = Fernet.generate_key()
cipher_suite = Fernet(SECRET_KEY)

def create_mock_ota_update():
    """
    Simulates the Pit Wall compiling a new PyTorch model, encrypting it,
    and pushing the binary payload over the air.
    """
    model_v2 = EdgeModelV2()
    # Save the raw state dict
    raw_path = "model_v2.pt"
    torch.save(model_v2.state_dict(), raw_path)
    
    # Encrypt the binary file
    with open(raw_path, "rb") as f:
        file_data = f.read()
    
    encrypted_data = cipher_suite.encrypt(file_data)
    encrypted_path = "model_v2.enc"
    
    with open(encrypted_path, "wb") as f:
        f.write(encrypted_data)
        
    os.remove(raw_path) # Clean up the plaintext trace
    print(f"[PIT WALL] Compiled & Encrypted Update V2.1.0 to {encrypted_path}")
    return encrypted_path

# --- Edge Deployment Runtime ---
class JetsonRuntime:
    def __init__(self):
        self.device = torch.device('cpu') # Simulating ARM CPU or tiny GPU
        
        # Load Base Model
        self.active_model = EdgeModelV1().to(self.device)
        self.active_model.eval()
        
        self.polling_rate_hz = 1000  # Default aggressive driving rate
        self.underclock_mode = False
        
        # Power Management Stats
        self.consecutive_low_load = 0
        self.LOW_LOAD_THRESHOLD = 50 # 50 frames of boring telemetry triggers underclock
        
    def thermal_management(self, telemetry_frame):
        """
        Dynamically throttles the inference engine to manage Edge device thermals.
        If the input variance/delta T is low, the car is likely cruising (low risk).
        """
        # Simulate simple variance check (e.g., standard deviation of recent telemetry)
        frame_variance = np.std(telemetry_frame)
        
        if frame_variance < 0.5:
            self.consecutive_low_load += 1
        else:
            self.consecutive_low_load = 0
            if self.underclock_mode:
                self.underclock_mode = False
                self.polling_rate_hz = 1000
                print("\n[THERMAL MGT] High Transient Detected! Engine restored to BOOST MODE (1kHz Inference).")
                
        if self.consecutive_low_load > self.LOW_LOAD_THRESHOLD and not self.underclock_mode:
            self.underclock_mode = True
            # Throttle inference down to 10Hz to save battery and drop thermals
            self.polling_rate_hz = 10 
            print("\n[THERMAL MGT] Low Load Sustained. Triggering UNDERCLOCK MODE (10Hz Inference) to save power/heat.")
            
    def receive_ota_update(self, encrypted_file_path):
        """
        Handles secure firmare updates. Decrypts and hot-swaps the active model.
        """
        print(f"\n[RUNTIME] Inbound OTA Update Detected: {encrypted_file_path}")
        try:
            # 1. Decrypt into memory
            with open(encrypted_file_path, "rb") as f:
                encrypted_data = f.read()
                
            decrypted_data = cipher_suite.decrypt(encrypted_data)
            
            # Temporary write to generic buffer so torch can load it
            temp_path = "temp_decrypted.pt"
            with open(temp_path, "wb") as f:
                f.write(decrypted_data)
                
            # 2. Hot-Swap the Model Logic
            print("[RUNTIME] Decryption Successful. Validating Cryptographic Signature...")
            
            # Instantiate the new architecture class (assume agreed upon structure)
            new_model = EdgeModelV2()
            new_model.load_state_dict(torch.load(temp_path))
            new_model = new_model.to(self.device)
            new_model.eval()
            
            # Atomic pointer swap to ensure zero dropped telemetry frames
            self.active_model = new_model
            os.remove(temp_path)
            
            print(f"[RUNTIME] >>> OTA HOT-SWAP COMPLETE. Active Model: {self.active_model.version} <<<")
            
        except Exception as e:
             print(f"[RUNTIME ERROR] OTA Security Failure: {e}")

    def run_telemetry_loop(self):
        print(f"--- INIT JETSON FORMULA SAE EDGE RUNTIME ---")
        print(f"Active Model: {self.active_model.version} | Base Polling Rate: {self.polling_rate_hz}Hz\n")
        
        frames_processed = 0
        ota_triggered = False
        
        try:
            while True:
                # 1. Simulate CAN Bus Ingestion
                # Inject high variance around frame 120 to wake up the system
                if 120 < frames_processed < 150:
                    raw_telemetry = np.random.normal(loc=0, scale=2.0, size=(10,))
                else:
                    raw_telemetry = np.random.normal(loc=0, scale=0.1, size=(10,))
                
                # 2. Thermal / Power Management
                self.thermal_management(raw_telemetry)
                
                # 3. Inference
                with torch.no_grad():
                    input_tensor = torch.tensor(raw_telemetry, dtype=torch.float32).unsqueeze(0).to(self.device)
                    prediction = self.active_model(input_tensor)
                
                # Sleep based on the current dynamic clock frequency
                sleep_time = 1.0 / self.polling_rate_hz
                time.sleep(sleep_time)
                
                frames_processed += 1
                
                if frames_processed % 20 == 0:
                     print(f"Frame {frames_processed:04d} processed. Status: [{'UNDERCLOCKED' if self.underclock_mode else 'BOOST'}] | Pred: {prediction.item():.2f}")
                
                # 4. Simulate Background OTA Push via Pit Wall (Mid-race update)
                if frames_processed == 70 and not ota_triggered:
                    encrypted_pkg = create_mock_ota_update()
                    self.receive_ota_update(encrypted_pkg)
                    ota_triggered = True
                    
                if frames_processed >= 200:
                    print("\n[RUNTIME] Telemetry Simulation Complete. Powering down.")
                    break
                    
        except KeyboardInterrupt:
             print("\n[RUNTIME] Manual Override. Terminating Loop.")


if __name__ == "__main__":
    edge_node = JetsonRuntime()
    edge_node.run_telemetry_loop()
