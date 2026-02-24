import argparse
import subprocess
import sys
import os

def run_script(script_name):
    print(f"\n{'='*50}")
    print(f"Executing: {script_name}")
    print(f"{'='*50}\n")
    if not os.path.exists(script_name):
        print(f"Error: Could not find {script_name}")
        return False
    
    result = subprocess.run([sys.executable, script_name])
    return result.returncode == 0

def main():
    parser = argparse.ArgumentParser(description="Formula SAE AI Telemetry & VCU Execution Engine")
    
    parser.add_argument('--simulate', action='store_true', help='Run CAN data simulation')
    parser.add_argument('--compliance', action='store_true', help='Run SUPRA 2025 compliance checker')
    parser.add_argument('--advanced-analysis', action='store_true', help='Run advanced CAN analysis (Kalman/Fuzzy)')
    parser.add_argument('--train-vae', action='store_true', help='Train VAE + Isolation Forest anomaly ensemble')
    parser.add_argument('--triage', action='store_true', help='Run Online Learning & Triage engine')
    parser.add_argument('--pinn', action='store_true', help='Train PINN LSTM-TCN Degradation model')
    parser.add_argument('--optimize', action='store_true', help='Run deployment optimizations (INT8, Pruning, Distillation)')
    parser.add_argument('--deploy', action='store_true', help='Simulate Jetson Edge deployment pipeline')
    parser.add_argument('--vcu', action='store_true', help='Run ROS2/CANopen VCU integration node')
    parser.add_argument('--validate', action='store_true', help='Run Monte Carlo Validation Framework')
    parser.add_argument('--all', action='store_true', help='Run the entire pipeline sequentially')

    args = parser.parse_args()

    steps = [
        ('simulate_can_data.py', args.simulate),
        ('check_compliance.py', args.compliance),
        ('advanced_can_analysis.py', args.advanced_analysis),
        ('train_vae_ensemble.py', args.train_vae),
        ('online_triage_engine.py', args.triage),
        ('pinn_lstm_tcn_degradation.py', args.pinn),
        ('optimize_models.py', args.optimize),
        ('jetson_deployment_pipeline.py', args.deploy),
        ('vcu_integration_node.py', args.vcu),
        ('monte_carlo_validation.py', args.validate),
    ]

    ran_any = False
    for script, should_run in steps:
        if should_run or args.all:
            success = run_script(script)
            ran_any = True
            if not success and args.all:
                print(f"\n[FATAL] Pipeline halted due to error in {script}.")
                sys.exit(1)

    if not ran_any:
        print("No execution steps requested. Welcome to the FSAE AI Telemetry Platform.")
        print("Use --help to see available modules to run.")
        print("Example: python main.py --simulate --compliance")

if __name__ == "__main__":
    main()
