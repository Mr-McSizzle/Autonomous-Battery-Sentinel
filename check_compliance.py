import pandas as pd
import re
import os
import sys

def parse_rulebook(filepath="rulebook.txt"):
    """
    Parses the SUPRA 2025 rulebook text to extract compliance limits.
    In a full implementation, this might use NLP. Here, we use regex
    to find key parameters mentioned in A1.3, A3.3, and IC electrical rules.
    """
    print(f"Parsing rulebook: {filepath}...")
    rules = {
        'max_voltage_v': 60.0, # Default based on prompt, but we try to read it
        'max_temp_c': 120.0,
        'max_current_a': 30.0,
        'max_dropout_ms': 500.0 
    }
    
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found. Using default SUPRA limits.")
        return rules

    with open(filepath, 'r') as f:
        text = f.read()

    # Look for voltage limit (e.g., "<60V", "exceeding 60V", "60.0V")
    volt_match = re.search(r'exceeding\s+(\d+\.?\d*)\s*V', text, re.IGNORECASE)
    if volt_match:
        rules['max_voltage_v'] = float(volt_match.group(1))

    # Look for temperature limit (e.g., "exceeding 120C")
    temp_match = re.search(r'exceeding\s+(\d+\.?\d*)\s*C', text, re.IGNORECASE)
    if temp_match:
        rules['max_temp_c'] = float(temp_match.group(1))
        
    # Look for current limit (e.g., "above 30A")
    curr_match = re.search(r'above\s+(\d+\.?\d*)\s*A', text, re.IGNORECASE)
    if curr_match:
        rules['max_current_a'] = float(curr_match.group(1))
        
    # Look for max dropout duration (e.g., "<500ms")
    drop_match = re.search(r'<\s*(\d+\.?\d*)\s*ms', text, re.IGNORECASE)
    if drop_match:
        rules['max_dropout_ms'] = float(drop_match.group(1))

    print(f"Extracted Rules: {rules}")
    return rules

def check_compliance(data_file="can_training_data.csv", rules=None):
    if rules is None:
        rules = parse_rulebook()
        
    print(f"\nChecking compliance for {data_file}...")
    try:
        df = pd.read_csv(data_file)
    except FileNotFoundError:
        print(f"Error: {data_file} not found. Run the simulation script first.")
        sys.exit(1)

    sample_rate_hz = 1000 # Configured in our generator
    ms_per_sample = 1000 / sample_rate_hz
    
    violations = []

    # 1. IC Electrical Limits: Voltage < 60V
    max_recorded_v = df['total_voltage'].max()
    if max_recorded_v >= rules['max_voltage_v']:
        violations.append(f"FAIL (IC LIMITS): Max voltage recorded is {max_recorded_v}V, exceeding rule limit of {rules['max_voltage_v']}V.")
    else:
        print(f"PASS (IC LIMITS): Max voltage is {max_recorded_v}V (< {rules['max_voltage_v']}V).")

    # Check individual cells if present (assume 60V limit applies to total, but cells shouldn't be crazy high either)
    # The rule says "Any cell or cumulative voltage exceeding 60.0V". We already checked total.

    # 2. A3.3 Compliance: Sensor Spikes / Critical Temperatures
    # Check if engine temp spiked above limit
    max_temp = df['engine_temp_c'].max()
    if max_temp >= rules['max_temp_c']:
         violations.append(f"FAIL (A3.3 Compliance): Engine temp spiked to {max_temp}C, exceeding {rules['max_temp_c']}C limit.")
    else:
         print(f"PASS (A3.3 Compliance): Engine temp reached max {max_temp}C (< {rules['max_temp_c']}C).")

    # 3. A3.3 Compliance: Current limits (prolonged draw)
    # Just checking for absolute spikes here for simplicity, or 99th percentile
    max_curr = df['lv_current_a'].max()
    if max_curr >= rules['max_current_a']:
        violations.append(f"FAIL (A3.3 Compliance): LV Current spiked to {max_curr}A, exceeding {rules['max_current_a']}A limit.")
    else:
        print(f"PASS (A3.3 Compliance): Max LV Current is {max_curr}A (< {rules['max_current_a']}A).")

    # 4. A1.3 Good Engineering Practices: Signal Dropouts due to vibration
    # Find sequences of 0s in wheel speed (indicating dropout)
    for col in ['wheelspeed_fl', 'wheelspeed_fr', 'wheelspeed_rl', 'wheelspeed_rr']:
        # Create boolean mask where wheelspeed is 0
        is_zero = (df[col] == 0)
        # Calculate consecutive lengths of 0s
        # Cumsum of non-zeros creates unique IDs for groups of consecutive zeros
        zero_groups = is_zero.groupby((~is_zero).cumsum())
        # The sum of True values in each group is the run length
        max_consecutive_zeros = zero_groups.sum().max()
        
        dropout_ms = max_consecutive_zeros * ms_per_sample
        if dropout_ms >= rules['max_dropout_ms']:
            violations.append(f"FAIL (A1.3 Good Practices): {col} experienced a dropout lasting {dropout_ms}ms (>= {rules['max_dropout_ms']}ms max).")
            # Only report the worst offender once for A1.3 to avoid filling the log, or continue.
            
    if not any("Good Practices" in v for v in violations):
         print(f"PASS (A1.3 Good Practices): No vibration-induced sensor dropouts exceeded {rules['max_dropout_ms']}ms.")

    print("\n--- COMPLIANCE REPORT ---")
    if len(violations) == 0:
        print("RESULT: PASSED ALL CHECKS! The vehicle/pipeline is compliant.")
    else:
        print(f"RESULT: FAILED ({len(violations)} Violations Found):")
        for v in violations:
            print(f" - {v}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='SUPRA 2025 Compliance Checker')
    parser.add_argument('--rulebook', type=str, default='rulebook.txt', help='Path to rulebook text file')
    parser.add_argument('--data', type=str, default='can_training_data.csv', help='Path to CAN data CSV')
    args = parser.parse_args()
    
    rules = parse_rulebook(filepath=args.rulebook)
    check_compliance(data_file=args.data, rules=rules)
