import time
import json

# --- Dummy CANopen Protocol Wrapper ---
class CANopenObjectDictionary:
    def __init__(self):
        # Simulated OD indices for Formula SAE metrics
        self.od = {
            0x6000: {"name": "total_voltage", "value": 0.0, "type": "FLOAT32"},
            0x6001: {"name": "engine_temp", "value": 0.0, "type": "FLOAT32"},
            0x6002: {"name": "lv_current", "value": 0.0, "type": "FLOAT32"},
            0x6003: {"name": "engine_rpm", "value": 0.0, "type": "FLOAT32"}
        }

    def write_pdo(self, index, value):
        if index in self.od:
            self.od[index]["value"] = value

    def read_pdo(self, index):
        return self.od[index]["value"] if index in self.od else None
        
    def generate_payload(self):
        payload = {}
        for k, v in self.od.items():
            payload[v["name"]] = v["value"]
        return json.dumps(payload)


# --- Dummy Deep Learning Edge Model ---
class QuantizedEdgeEngine:
    def predict_failure_risk(self, telemetry):
        # Mocking an engine that takes a dictionary of telemetry and returns a risk percentage
        # Under normal conditions it returns ~5% risk based on our INT8 distilled models.
        risk = 5.0
        
        # Artificial anomaly: Engine Temp > 115C pushes risk past 80% (Critical cooling fault)
        if telemetry.get("engine_temp", 0) > 115.0:
            risk = 85.0
            
        return risk


# --- Simulated ROS2 Node ---
class VCUNode:
    def __init__(self):
        print("[VCU NODE] Initializing Vehicle Control Unit (ROS2 Canonical Sim).")
        self.canopen_od = CANopenObjectDictionary()
        self.ml_engine = QuantizedEdgeEngine()
        
        # SUPRA A3.10 Hardware Fail-Safe
        # True = Relays Closed (Vehicle Active)
        # False = Relays Open (Vehicle Dead)
        self.high_voltage_relays_closed = True 
        
        # Simulated Hardware GPIO Pin (Master Switch)
        self.physical_master_switch_on = True

    def publish_telemetry(self, name, voltage, temp, current, rpm):
        # Wrap data into CANopen PDOs
        self.canopen_od.write_pdo(0x6000, voltage)
        self.canopen_od.write_pdo(0x6001, temp)
        self.canopen_od.write_pdo(0x6002, current)
        self.canopen_od.write_pdo(0x6003, rpm)
        
        # Publish exactly like a ROS publisher over the wire
        payload = self.canopen_od.generate_payload()
        print(f"\n[CANopen TX] {name} -> {payload}")
        return json.loads(payload)

    def telemetry_callback(self, telemetry_data):
        """
        ROS2 Subscriber Callback. Receives CANopen objects, evaluates logic via edge AI, outputs command.
        """
        if not self.high_voltage_relays_closed:
            print("[VCU] Relays are currently OPEN. Ignoring telemetry logic. Vehicle is SAFE.")
            return

        # 1. HARDWARE FAIL-SAFE OVERRIDE (SUPRA A3.10)
        # Physics and hardware permanently trump software logic. 
        if not self.physical_master_switch_on:
            print("[CRITICAL INTERRUPT] Physical Master Switch GPIO dropped to OFF (A3.10 Compliance).")
            print(">>> BYPASSING SOFTWARE NETWORKS. COMMANDING HIGH VOLTAGE RELAYS OPEN! <<<")
            self.high_voltage_relays_closed = False
            return
            
        # 2. SOFTWARE LOGIC (ML ENSEMBLE)
        ml_risk_prob = self.ml_engine.predict_failure_risk(telemetry_data)
        print(f"[VCU Edge ML] Calculated INT8 Neural Network Failure Risk: {ml_risk_prob:.1f}%")
        
        if ml_risk_prob > 20.0:
             print("[SOFTWARE INTERRUPT] Deep Learning Model Anomaly Detected >20% Threshold.")
             print(">>> COMMANDING HIGH VOLTAGE RELAYS OPEN! <<<")
             self.high_voltage_relays_closed = False
             
    def trigger_physical_master_switch(self):
        """
        Simulate a driver physically punching the Big Red Button.
        """
        self.physical_master_switch_on = False


def run_simulation():
    vcu = VCUNode()
    print("\n--- SCENARIO 1: NORMAL DRIVING ---")
    data = vcu.publish_telemetry("Frame_001", voltage=58.2, temp=85.0, current=15.0, rpm=6000)
    vcu.telemetry_callback(data)
    time.sleep(0.5)
    
    print("\n--- SCENARIO 2: MACHINE LEARNING ANOMALY FAULT ---")
    # Simulate a sudden overheating spike causing an ML safety violation.
    data = vcu.publish_telemetry("Frame_002", voltage=58.0, temp=120.0, current=16.0, rpm=6500)
    vcu.telemetry_callback(data)
    
    # Reset the VCU relays and switch for the final test
    vcu.high_voltage_relays_closed = True
    vcu.physical_master_switch_on = True
    time.sleep(0.5)
    
    print("\n--- SCENARIO 3: A3.10 MASTER SWITCH FAIL-SAFE TEST ---")
    # In this scenario, the AI thinks the car runs perfectly normal. 
    # But suddenly, the driver triggers the physical Master Switch due to an unseen external hazard.
    vcu.trigger_physical_master_switch()
    data = vcu.publish_telemetry("Frame_003", voltage=58.5, temp=82.0, current=12.0, rpm=5000)
    vcu.telemetry_callback(data)
    
if __name__ == "__main__":
    run_simulation()
