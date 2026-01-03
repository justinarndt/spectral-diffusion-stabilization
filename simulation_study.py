import stim
import pymatching
import numpy as np
import time
from collections import deque
import matplotlib.pyplot as plt

# --- CONFIGURATION (GOOGLE SPEC) ---
DISTANCE = 11          
QEC_ROUNDS = 5
BATCH_SIZE = 1024      
ROUNDS = 20000         
LATENCY = 10           

# HARDWARE-RELEVANT PHYSICS
BASE_GATE_NOISE = 0.001   # 0.1% Gate
MEASURE_NOISE = 0.05      # 5.0% Measurement (Lethal)
RESET_NOISE = 0.01        # 1.0% Reset
ACTUATION_NOISE = 2e-5    

class IntegralController:
    def __init__(self):
        self.state = 0.0
        self.Ki = 0.05
        self.setpoint = 0.0
    
    def calibrate(self, plant):
        print("Calibrating Controller Setpoint...")
        densities = []
        for _ in range(10):
            _, d = plant.run_batch(0.0)
            densities.append(d)
        self.setpoint = np.mean(densities)
        print(f"Setpoint Locked: {self.setpoint:.5f}")

    def update(self, density):
        error = density - self.setpoint
        self.state += error * self.Ki
        self.state = max(-0.02, min(0.15, self.state))
        return self.state

class SurfaceCodePlant:
    def __init__(self, distance=DISTANCE):
        self.d = distance
        self.num_data_qubits = distance**2 
        self.per_qubit_drift = np.zeros(self.num_data_qubits)
        self.true_gate_drift = 0.0 
        
    def _gen_circuit(self, p_gate):
        c = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            distance=self.d, rounds=QEC_ROUNDS,
            after_clifford_depolarization=p_gate,
            before_measure_flip_probability=MEASURE_NOISE,
            after_reset_flip_probability=RESET_NOISE
        )
        return c

    def evolve_drift(self):
        target_global = 0.005
        force_global = 0.005 * (target_global - self.true_gate_drift)
        self.true_gate_drift += force_global + np.random.normal(0, 1e-6)
        
        target_local = 0.03 
        force_local = 0.008 * (target_local - self.per_qubit_drift)
        noise_local = np.random.normal(0, 2e-6, size=self.num_data_qubits)
        
        self.per_qubit_drift += force_local + noise_local
        self.per_qubit_drift = np.clip(self.per_qubit_drift, -0.02, 0.05)
        
        return self.true_gate_drift + np.mean(self.per_qubit_drift)

    def run_batch(self, correction_val):
        local_offsets = self.per_qubit_drift - correction_val
        avg_local_drift = np.mean(local_offsets)
        
        p_eff = BASE_GATE_NOISE + self.true_gate_drift + avg_local_drift
        p_eff = max(BASE_GATE_NOISE, min(0.4, p_eff))
        
        noisy_circuit = self._gen_circuit(p_eff)
        
        # ADAPTIVE DECODING
        decoder = pymatching.Matching.from_stim_circuit(noisy_circuit)
        
        # Use detector sampler with separate_observables=True
        sampler = noisy_circuit.compile_detector_sampler()
        detection_events, observable_flips = sampler.sample(shots=BATCH_SIZE, separate_observables=True)
        
        # detection_events: (BATCH_SIZE, num_detectors)
        # observable_flips: (BATCH_SIZE, 1) or flattened to (BATCH_SIZE,)
        preds = decoder.decode_batch(detection_events)
        if preds.ndim > 1:
            preds = preds.flatten()
        if observable_flips.ndim > 1:
            observable_flips = observable_flips.flatten()
        
        failures = np.sum(observable_flips != preds)
        density = np.mean(detection_events)
        
        return failures, density

def main():
    print(f"Init Plant: d={DISTANCE}, Batch={BATCH_SIZE}")
    print(f"Physics: Gate={BASE_GATE_NOISE}, Meas={MEASURE_NOISE}, Reset={RESET_NOISE}")
    plant = SurfaceCodePlant()
    controller = IntegralController()
    
    controller.calibrate(plant)
    
    correction_queue = deque([0.0] * LATENCY, maxlen=LATENCY)
    history = {"cycle": [], "p_gate": [], "cum_fail_std": [], "cum_fail_holo": []}
    cum_std = 0
    cum_holo = 0
    
    print(">>> RUNNING ADAPTIVE SIMULATION (20k Cycles) <<<")
    start_time = time.time()
    
    for t in range(ROUNDS):
        drift_avg = plant.evolve_drift()
        
        fail_std, _ = plant.run_batch(0.0)
        cum_std += fail_std
        
        u_active = correction_queue.popleft()
        u_noisy = u_active + np.random.normal(0, ACTUATION_NOISE)
        fail_holo, dens_holo = plant.run_batch(u_noisy)
        cum_holo += fail_holo
        
        new_u = controller.update(dens_holo)
        correction_queue.append(new_u)
        
        if t % 50 == 0:
            history["cycle"].append(t)
            history["p_gate"].append(BASE_GATE_NOISE + drift_avg)
            history["cum_fail_std"].append(cum_std)
            history["cum_fail_holo"].append(cum_holo)
            
            if t % 1000 == 0:
                print(f"t={t:5d} | AvgDrift={drift_avg:.4f} | StdFail={cum_std:,} | HoloFail={cum_holo:,}")

    duration = time.time() - start_time
    print(f"Done in {duration:.2f}s")
    
    plt.figure(figsize=(12, 7))
    plt.plot(history["cycle"], history["cum_fail_std"], 'r-', label="Standard Qubit (Uncorrected Drift)")
    plt.plot(history["cycle"], history["cum_fail_holo"], 'b-', linewidth=3, label="Holo-Neural Qubit (Stabilized)")
    plt.title(f"Google-Grade Drift Suppression (d={DISTANCE}, 5% Meas Noise, Local Vector Drift)")
    plt.xlabel("Cycles")
    plt.ylabel("Cumulative Logical Failures")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("final_proof.png", dpi=300)
    print("Saved 'final_proof.png'")

if __name__ == "__main__":
    main()