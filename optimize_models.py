import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import time
import os

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# --- Architecture ---
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1,
                                 self.conv2, self.chomp2, self.relu2)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size, stride=1, 
                                        dilation=dilation_size, padding=padding))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class PINN_LSTM_TCN(nn.Module):
    def __init__(self, num_inputs, tcn_channels, lstm_hidden, output_dim=1):
        super(PINN_LSTM_TCN, self).__init__()
        self.tcn = TemporalConvNet(num_inputs, tcn_channels)
        self.lstm = nn.LSTM(tcn_channels[-1], lstm_hidden, batch_first=True)
        self.fc = nn.Linear(lstm_hidden, output_dim)
        
    def forward(self, x):
        x_tcn = x.transpose(1, 2)
        out_tcn = self.tcn(x_tcn)
        out_tcn = out_tcn.transpose(1, 2)
        lstm_out, _ = self.lstm(out_tcn)
        r_pred = self.fc(lstm_out)
        return r_pred

# --- Knowledge Distillation Loss ---
def distillation_loss(student_pred, teacher_pred, true_target, alpha=0.5):
    """
    Combines hard dataset targets (MSE) with soft Teacher Targets (MSE).
    """
    hard_loss = nn.functional.mse_loss(student_pred, true_target)
    soft_loss = nn.functional.mse_loss(student_pred, teacher_pred)
    return (alpha * soft_loss) + ((1 - alpha) * hard_loss)

# --- Profiling Utility ---
def profile_model(model, test_X, iterations=100, name="Model"):
    # Size in memory
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        
    size_mb = (param_size + buffer_size) / 1024**2
    size_kb = (param_size + buffer_size) / 1024
    
    # Measure Latecy
    model.eval()
    times = []
    with torch.no_grad():
        # Warmup
        for _ in range(5):
            _ = model(test_X)
        # Benchmark
        for _ in range(iterations):
            start = time.perf_counter()
            _ = model(test_X)
            end = time.perf_counter()
            times.append((end - start) * 1000) # milliseconds
            
    avg_latency = np.mean(times)
    
    print(f"\n[{name}] Profiling Results:")
    if size_mb < 1.0:
        print(f" -> Memory Footprint: {size_kb:.2f} KB")
    else:
        print(f" -> Memory Footprint: {size_mb:.2f} MB")
    print(f" -> Target Constraint: < 1000.00 MB (< 1GB)")
    print(f" -> Average Inference Latency: {avg_latency:.3f} ms (Target < 20.0 ms)")
    
    return size_kb, avg_latency

def main():
    print("Initializing Edge Compute Deployment Optimization Pipeline...")
    # Generate synthetic input for benchmarking
    # Shape: (Batch=32, Sequence=50, Channels=3)
    X_train = torch.randn(64, 50, 3) 
    y_train = torch.randn(64, 50, 1)
    
    X_test_batch = torch.randn(1, 50, 3) # Online point-by-point latency
    
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=16)

    print("\n--- 1. Knowledge Distillation ---")
    # Teacher (Large, accurate)
    teacher = PINN_LSTM_TCN(num_inputs=3, tcn_channels=[32, 64], lstm_hidden=64)
    teacher.eval()
    
    # Student (Small, fast)
    student = PINN_LSTM_TCN(num_inputs=3, tcn_channels=[16, 16], lstm_hidden=16)
    
    print("Training Student Network on Teacher Soft Targets...")
    optimizer = optim.Adam(student.parameters(), lr=1e-3)
    student.train()
    for _ in range(10): # Demo loop
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            with torch.no_grad():
                t_pred = teacher(batch_x)
            s_pred = student(batch_x)
            loss = distillation_loss(s_pred, t_pred, batch_y)
            loss.backward()
            optimizer.step()
            
    # Base profiling
    orig_kb, orig_latency = profile_model(student, X_test_batch, name="Base Student (FP32, Dense)")

    print("\n--- 2. L1 Unstructured Pruning (60% Sparsity) ---")
    # Apply pruning to 1D Convs and Linear layers
    parameters_to_prune = []
    for module in student.modules():
        if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
            parameters_to_prune.append((module, 'weight'))
            
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=0.6,
    )
    
    # Make pruning permanent
    for module, name in parameters_to_prune:
        prune.remove(module, name)
        
    print("Pruned 60% of all Convolutional and Linear weights to absolute zero.")
    
    print("\n--- 3. INT8 Dynamic Quantization ---")
    # Convert FP32 PyTorch model to INT8 representation
    student.eval()
    # Dynamic Quantization currently supports nn.Linear and nn.LSTM
    quantized_student = torch.ao.quantization.quantize_dynamic(
        student,  
        {nn.LSTM, nn.Linear},  # Specify layers to quantize
        dtype=torch.qint8
    )
    
    print("Converted Student Model LSTM and Linear Layers to Discrete INT8 precision.")
    
    quant_kb, quant_latency = profile_model(quantized_student, X_test_batch, name="Optimized Student (INT8, 60% Sparse)")
    
    # Summary
    print("\n=== OPTIMIZATION SUMMARY ===")
    print(f"Memory Reduction: {orig_kb:.1f} KB -> {quant_kb:.1f} KB ({(1 - quant_kb/orig_kb)*100:.1f}%)")
    
    if quant_kb < 1024 * 1000:
         print(f"Goal (<1GB): PASS")
         
    if quant_latency < 20.0:
         print(f"Inference Latency ({quant_latency:.2f}ms < 20ms): PASS")
    else:
         print(f"Inference Latency ({quant_latency:.2f}ms < 20ms): FAIL")


if __name__ == "__main__":
    main()
