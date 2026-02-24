# Autonomous Battery Sentinel

## Project Overview
The **Autonomous Battery Sentinel** is an end-to-end AI-powered telemetry analysis and anomaly detection framework built for Formula SAE vehicles. Designed to operate directly on Edge-compute hardware (e.g., NVIDIA Jetson / VCU), this pipeline ingests raw CAN bus telemetry, detects faults using advanced machine learning ensembles (VAE + Isolation Forest, Kalman Filters, Fuzzy Logic), forecasts component degradation (PINN LSTM-TCN), and autonomously commands hardware fail-safes. 

## SUPRA 2025 Compliance Notes
This system is architected in strict adherence to the SUPRA 2025 rulebook:
- **Low-Voltage Limit (IC4.1 / EV System)**: The system strictly monitors the `<60V` continuous low-voltage/accumulator limits. Anomalies crossing this physics boundary trigger immediate algorithmic fault detection.
- **Shutdown Integration (CV4 / EV6)**: If the neural engine detects a probability of failure exceeding safety thresholds (>20% risk), the VCU software integration natively commands the High-Voltage / Master Relays to OPEN, seamlessly integrating into the tractive system hardware safety loops.
- **On-Device Only Processing (A3.3)**: To comply with prohibitions on off-vehicle telemetry control telemetry (A3.3), all models undergo Knowledge Distillation, L1 Pruning, and INT8 Quantization. This guarantees the entire deep-learning pipeline runs autonomously on the car's physical edge node (`<1GB` memory footprint, `<20ms` latency) with zero reliance on cloud/pit-wall compute.

---

## Submission Folder Structure

The repository is logically divided into pipeline stages:

```text
Submission.zip/
└── Code/
    ├── data_simulation/
    │   └── simulate_can_data.py           # Generates initial raw chaotic CAN telemetry with EMI Models
    ├── preprocessing/
    │   └── check_compliance.py            # Parses SUPRA 2025 `rulebook.txt` and performs basic boundary compliance checks
    ├── anomaly_detection/
    │   ├── advanced_can_analysis.py       # Kalman Filters, Fuzzy Logic, Bayesian Probabilities
    │   ├── train_vae_ensemble.py          # 1D-CNN VAE + Isolation Forest Anomaly Detection 
    │   └── online_triage_engine.py        # k-NN Triage and Online Learning (dynamic weight retraining)
    ├── forecasting/
    │   └── pinn_lstm_tcn_degradation.py   # Arrhenius-governed PINN for battery fatigue mapping
    ├── optimization/
    │   └── optimize_models.py             # Distillation, L1 Pruning (60%), and INT8 Dynamic Quantization
    ├── integration/
    │   ├── jetson_deployment_pipeline.py  # End-to-end edge deployment script (Thermal Underclocking/OTA)
    │   └── vcu_integration_node.py        # Master ROS2 loop applying ML overrides, CANopen OD Wrappers, and Hardware Fail-Safes
    ├── utils/
    │   ├── monte_carlo_validation.py      # Final chaotic validation scoring (Precision/Recall targets and Ah battery models)
    │   └── rulebook.txt                   # Baseline SUPRA 2025 constraints knowledge base
    ├── main.py                            # Pipeline Orchestrator and Master Execution switch
    └── README.md                          # This file
```

---

## Quick Run

Assuming the baseline data (`normal_can_data.csv` or standard `can_training_data.csv`) is present in the root directory:

```bash
python main.py --all
```
*This executes the entire pipeline sequentially from data simulation down to Monte Carlo validation.*

---

## Full Run Instructions 

The `main.py` orchestrator utilizes `argparse` to allow independent execution of the Autonomous Battery Sentinel's sub-modules. 

### Prerequisites
Ensure your environment has the following packages installed:
```bash
pip install torch pandas numpy scikit-learn filterpy cryptography matplotlib
```

### Execution Commands

You can trigger specific segments of the project by passing the corresponding flags to `main.py`.

1. **Simulate Data & Preprocessing:**
   Generate physical EMI noise profiles and test basic rule compliance.
   ```bash
   python main.py --simulate --compliance
   ```

2. **Run Anomaly Detection Engines:**
   Execute the Deep Learning suites (VAE, Ensemble methods, and dynamic triaging).
   ```bash
   python main.py --advanced-analysis --train-vae --triage
   ```

3. **Train Forecasting & Optimize:**
   Train the Physics-Informed Neural Network (PINN) for battery degradation, then compress it for Edge deployment.
   ```bash
   python main.py --pinn --optimize
   ```

4. **Simulate Edge Deployment & VCU Integration:**
   Run the thermal management loops, encrypted OTA updates, and CANopen/ROS2 integration node.
   ```bash
   python main.py --deploy --vcu
   ```

5. **Perform Final Validation:**
   Score the system rigorously against Monte Carlo randomized fault injections to verify F1-Scores and Battery Lifecycle extensions.
   ```bash
   python main.py --validate
   ```

To view all available execution arguments at any time, run:
```bash
python main.py --help
```
