import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import os

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# VAE Architecture
class Conv1D_VAE(nn.Module):
    def __init__(self, seq_len, n_features, latent_dim=8):
        super(Conv1D_VAE, self).__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.latent_dim = latent_dim
        
        # Encoder
        self.enc_conv1 = nn.Conv1d(n_features, 16, kernel_size=3, padding=1)
        self.enc_conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1, stride=2)
        
        # Determine the size after convolutions
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
    # MSE Loss
    mse_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    
    # KL Divergence
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Physics constraint: SUPRA 2025 IC <60V limit 
    # Penalty if reconstructed voltage exceeds the scaled limit
    recon_voltage = recon_x[:, voltage_idx, :]
    physics_penalty = torch.sum(torch.relu(recon_voltage - voltage_limit_scaled)) * 1000.0 # Heavy penalty
    
    return mse_loss + beta * kld_loss + physics_penalty

def calculate_anomaly_scores(model, data_loader, device):
    model.eval()
    errors = []
    latents = []
    with torch.no_grad():
        for batch_x in data_loader:
            batch_x = batch_x[0].to(device)
            recon_x, mu, _ = model(batch_x)
            
            # Error per sequence
            mse = torch.mean((batch_x - recon_x)**2, dim=(1,2)).cpu().numpy()
            errors.extend(mse)
            latents.extend(mu.cpu().numpy())
    return np.array(errors), np.array(latents)

def main():
    print("Loading data...")
    # Read the dataset
    df = pd.read_csv("advanced_analyzed_data.csv")
    
    # Let's say we train on the first 10 seconds of data (10000 samples) as "normal"
    # and infer on the whole dataset to find anomalies.
    features = ['total_voltage', 'engine_temp_c', 'lv_current_a', 'engine_rpm']
    data = df[features].values
    
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Physics constraints calculations
    voltage_idx = features.index('total_voltage')
    voltage_mean = scaler.mean_[voltage_idx]
    voltage_scale = scaler.scale_[voltage_idx]
    voltage_limit_scaled = (60.0 - voltage_mean) / voltage_scale
    print(f"Voltage Limit scaled threshold: {voltage_limit_scaled:.3f}")

    # Create sequences
    seq_len = 50 # 50ms windows
    X_seq = []
    for i in range(len(data_scaled) - seq_len):
        X_seq.append(data_scaled[i:i+seq_len])
    X_seq = np.array(X_seq)
    
    # PyTorch expects shape (Batch, Channels, Timesteps)
    X_seq = torch.tensor(X_seq, dtype=torch.float32).transpose(1, 2)
    
    # Train on first 10000 windows (assumed normal/clean operation)
    train_size = min(10000, len(X_seq) // 2)
    X_train = X_seq[:train_size]
    X_all = X_seq
    
    train_loader = DataLoader(TensorDataset(X_train), batch_size=128, shuffle=True)
    all_loader = DataLoader(TensorDataset(X_all), batch_size=256, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Conv1D_VAE(seq_len=seq_len, n_features=len(features), latent_dim=12).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    print("\n--- Training VAE (Physics-Informed) ---")
    epochs = 10
    model.train()
    for epoch in range(epochs):
        train_loss = 0
        for batch_x in train_loader:
            batch_x = batch_x[0].to(device)
            optimizer.zero_grad()
            recon_x, mu, logvar = model(batch_x)
            loss = vae_loss_function(recon_x, batch_x, mu, logvar, voltage_idx, voltage_limit_scaled)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss / len(train_loader.dataset):.4f}")
        
    print("\n--- Extracting Reconstruction Errors & Latents ---")
    recon_errors, latents = calculate_anomaly_scores(model, all_loader, device)
    
    print("\n--- Training Isolation Forest Ensemble ---")
    # Features for iForest: mu latent vector + reconstruction error
    # This detects complex multi-variate structural deviations
    ensemble_features = np.hstack((latents, recon_errors.reshape(-1, 1)))
    
    # We train the isolation forest on the *training* (normal) embeddings
    train_latents = ensemble_features[:train_size]
    iso_forest = IsolationForest(contamination=0.01, random_state=42)
    iso_forest.fit(train_latents)
    
    print("\n--- Dynamic Thresholding & Anomaly Detection ---")
    # iForest predictions
    preds = iso_forest.predict(ensemble_features) # -1 is anomaly, 1 is normal
    
    # Dynamic Thresholding (+2 sigma rolling error)
    # We pad the initial values to maintain index matching
    error_series = pd.Series(recon_errors)
    roll_mean = error_series.rolling(window=1000, min_periods=1).mean()
    roll_std = error_series.rolling(window=1000, min_periods=1).std().fillna(0)
    dynamic_threshold = roll_mean + 2 * roll_std
    
    # Logical OR: Flag if iForest says anomaly OR if error > dynamic threshold
    iforest_flags = (preds == -1)
    dyn_thresh_flags = (error_series > dynamic_threshold).values
    
    ensemble_anomalies = iforest_flags | dyn_thresh_flags
    
    num_anomalies = np.sum(ensemble_anomalies)
    print(f"Total sequences analyzed: {len(X_seq)}")
    print(f"Anomalies detected by Ensemble VAE/iForest: {num_anomalies}")
    
    if num_anomalies > 0:
        print("\n--- Anomaly Sample Insights ---")
        anomaly_indices = np.where(ensemble_anomalies)[0]
        # Show the first few
        for idx in anomaly_indices[:5]:
            time_sec = df.iloc[idx]['time_s']
            err = recon_errors[idx]
            thresh = dynamic_threshold.iloc[idx]
            iforest_status = "Flagged" if preds[idx] == -1 else "Passed"
            print(f"Time: {time_sec:.3f}s | Recon Err: {err:.4f} (Thresh: {thresh:.4f}) | iForest: {iforest_status}")
            
    # Save the anomaly scores back to the dataframe
    print("\nSaving results...")
    # Pad the beginning because we windowed
    padding = [False] * seq_len
    df['vae_iforest_anomaly'] = np.concatenate([padding, ensemble_anomalies])
    df.to_csv("vae_ensemble_analyzed_data.csv", index=False)
    print("Saved to vae_ensemble_analyzed_data.csv")

if __name__ == "__main__":
    main()
