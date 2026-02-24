import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import os

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# VAE Architecture (Same as train_vae_ensemble.py)
class Conv1D_VAE(nn.Module):
    def __init__(self, seq_len, n_features, latent_dim=12):
        super(Conv1D_VAE, self).__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.latent_dim = latent_dim
        
        # Encoder
        self.enc_conv1 = nn.Conv1d(n_features, 16, kernel_size=3, padding=1)
        self.enc_conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1, stride=2)
        
        self.enc_out_len = seq_len // 2
        
        self.fc_mu = nn.Linear(32 * self.enc_out_len, latent_dim)
        self.fc_logvar = nn.Linear(32 * self.enc_out_len, latent_dim)
        
        # Decoder
        self.dec_fc = nn.Linear(latent_dim, 32 * self.enc_out_len)
        self.dec_conv1 = nn.ConvTranspose1d(32, 16, kernel_size=3, padding=1, stride=2, output_padding=1)
        self.dec_conv2 = nn.ConvTranspose1d(16, n_features, kernel_size=3, padding=1)
        
        self.relu = nn.ReLU()
        
    def encode(self, x):
        h = self.relu(self.enc_conv1(x))
        h = self.relu(self.enc_conv2(h))
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def decode(self, z):
        h = self.relu(self.dec_fc(z))
        h = h.view(h.size(0), 32, self.enc_out_len)
        h = self.relu(self.dec_conv1(h))
        return self.dec_conv2(h) # Linear output
        
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

def vae_loss_function(recon_x, x, mu, logvar, voltage_idx, voltage_limit_scaled, beta=0.1):
    mse_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Physics constraint
    recon_voltage = recon_x[:, voltage_idx, :]
    physics_penalty = torch.sum(torch.relu(recon_voltage - voltage_limit_scaled)) * 1000.0
    return mse_loss + beta * kld_loss + physics_penalty

def extract_feature_errors(model, batch_x, device):
    """ Extract raw feature-wise MSE for Triage """
    model.eval()
    with torch.no_grad():
        batch_x = batch_x.to(device)
        recon_x, _, _ = model(batch_x)
        # Error per sequence per feature -> (Batch, Features)
        feature_errors = torch.mean((batch_x - recon_x)**2, dim=2).cpu().numpy()
    return feature_errors

def train_triage_knn():
    """ 
    Synthesizes a rule-based training set for the k-NN classifier.
    Features: [MSE_Total_Voltage, MSE_Engine_Temp, MSE_LV_Current, MSE_Engine_RPM]
    """
    print("\n--- Initializing k-NN Triage Sub-module ---")
    
    # 0 = Normal, 1 = Battery/Power (IC4 violation), 2 = Cooling Fault, 3 = EMI/Ignition Spikes
    X_train_knn = np.array([
        # Base normal noise
        [0.1, 0.1, 0.1, 0.1],
        [0.2, 0.0, 0.1, 0.2],
        
        # High Voltage/Current Error -> Battery/Power
        [5.0, 0.1, 4.0, 0.2],
        [4.0, 0.2, 5.0, 0.1],
        
        # High Temp Error -> Cooling
        [0.1, 6.0, 0.2, 0.1],
        [0.2, 7.0, 0.1, 0.1],
        
        # High RPM/Voltage erratic noise -> EMI
        [2.0, 0.1, 1.0, 5.0],
        [1.5, 0.1, 1.5, 6.0]
    ])
    
    y_train_knn = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    
    # K=1 because our synthetic dataset is very small and absolute
    knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
    knn.fit(X_train_knn, y_train_knn)
    print("k-NN Classifier trained on multi-feature reconstruction error profiles.")
    return knn

def map_triage_label(label):
    triage_map = {
        0: "Normal Variance",
        1: "CRITICAL: Battery/Power Integrity Fault (IC4 limit breach)",
        2: "CRITICAL: Engine Cooling System Failure",
        3: "WARNING: Immediate EMI/Ignition Cross-talk detected"
    }
    return triage_map.get(label, "Unknown Anomaly")

def online_retrain(model, optimizer, batch_x, voltage_idx, voltage_limit_scaled, device, epochs=3):
    """
    Incremental Retraining Loop (Online Learning)
    Updates the model weights on a batch that was previously flagged as an anomaly, 
    but is now confirmed by race engineers as a "New Normal" driving maneuver.
    """
    print("\n--- ONLINE LEARNING ACTIVATED ---")
    print(f"Targeting sequence batch shape: {batch_x.shape}")
    print("Incorporating 'New Normal' sequence into VAE Latent Space...")
    
    model.train()
    batch_x = batch_x.to(device)
    
    # Before retraining, compute loss
    with torch.no_grad():
        recon_x_pre, mu_pre, logvar_pre = model(batch_x)
        loss_pre = vae_loss_function(recon_x_pre, batch_x, mu_pre, logvar_pre, voltage_idx, voltage_limit_scaled).item()
    print(f" > Initial Loss (False Positive Anomaly): {loss_pre / len(batch_x):.4f}")
    
    # Retrain loop
    for epoch in range(epochs):
        optimizer.zero_grad()
        recon_x, mu, logvar = model(batch_x)
        loss = vae_loss_function(recon_x, batch_x, mu, logvar, voltage_idx, voltage_limit_scaled)
        loss.backward()
        optimizer.step()
        
    # After retraining, compute loss
    with torch.no_grad():
        recon_x_post, mu_post, logvar_post = model(batch_x)
        loss_post = vae_loss_function(recon_x_post, batch_x, mu_post, logvar_post, voltage_idx, voltage_limit_scaled).item()
    print(f" > Post-Retrain Loss (Absorbed Normalcy): {loss_post / len(batch_x):.4f}")
    print("Optimization Complete. Model weights dynamically updated.")
    
def main():
    print("Loading telemetry anomalies for Triage and Online Learning...")
    # Read the dataset
    df = pd.read_csv("advanced_analyzed_data.csv")
    features = ['total_voltage', 'engine_temp_c', 'lv_current_a', 'engine_rpm']
    data = df[features].values
    
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    voltage_idx = features.index('total_voltage')
    voltage_mean = scaler.mean_[voltage_idx]
    voltage_scale = scaler.scale_[voltage_idx]
    voltage_limit_scaled = (60.0 - voltage_mean) / voltage_scale

    seq_len = 50 
    X_seq = []
    for i in range(len(data_scaled) - seq_len):
        X_seq.append(data_scaled[i:i+seq_len])
    X_seq = np.array(X_seq)
    X_seq = torch.tensor(X_seq, dtype=torch.float32).transpose(1, 2)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Conv1D_VAE(seq_len=seq_len, n_features=len(features), latent_dim=12).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Quick mimic of training to initialize weights correctly (1 epoch)
    train_size = min(1000, len(X_seq))
    train_loader = DataLoader(TensorDataset(X_seq[:train_size]), batch_size=128, shuffle=True)
    model.train()
    for batch_x in train_loader:
        batch_x = batch_x[0].to(device)
        optimizer.zero_grad()
        recon_x, mu, logvar = model(batch_x)
        loss = vae_loss_function(recon_x, batch_x, mu, logvar, voltage_idx, voltage_limit_scaled)
        loss.backward()
        optimizer.step()

    # Initialize Triage sub-module
    knn_triage = train_triage_knn()
    
    # Simulate intercepting a specific anomaly block (e.g. index 740 is near our Cooling Fault spike)
    # Let's say we intercept window 2900 (severe overheating spike simulated in advanced analysis)
    # We will pass this to the Triage module.
    idx_anomaly1 = 2900
    idx_anomaly2 = 2901  
    
    if idx_anomaly2 < len(X_seq):
        print(f"\n--- INTERCEPTED ANOMALOUS SEQUENCE (Timestep {idx_anomaly1}) ---")
        anomaly_batch = X_seq[idx_anomaly1:idx_anomaly2+1]
        
        # 1. Triage the Anomaly
        feature_errors = extract_feature_errors(model, anomaly_batch, device)
        predictions = knn_triage.predict(feature_errors)
        
        for i, pred in enumerate(predictions):
            print(f"Sequence {idx_anomaly1+i} Feature MSEs [Voltage, Temp, Current, RPM]: "
                  f"[{feature_errors[i][0]:.2f}, {feature_errors[i][1]:.2f}, {feature_errors[i][2]:.2f}, {feature_errors[i][3]:.2f}]")
            print(f">>> k-NN Triage Classification: {map_triage_label(pred)}")
    
    # Simulate a "False Positive" that we want the model to learn via Online Learning
    # Let's grab an early window that is just normal engine revving but was structurally novel initially.
    idx_false_pos = 100
    if idx_false_pos + 5 < len(X_seq):
        false_pos_batch = X_seq[idx_false_pos:idx_false_pos+5]
        
        print("\nRace Engineer flags Timesteps 100-104 as a False Positive (Normal Driver Maneuver).")
        # 2. Retrain incrementally
        online_retrain(model, optimizer, false_pos_batch, voltage_idx, voltage_limit_scaled, device, epochs=5)

if __name__ == "__main__":
    main()
