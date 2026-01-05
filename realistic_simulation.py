import stim
import pymatching
import numpy as np
import time
from collections import deque
import matplotlib.pyplot as plt

# --- CONFIGURATION (REALISTIC REGIME) ---
DISTANCE = 11
QEC_ROUNDS = 5
BATCH_SIZE = 1024
ROUNDS = 10000  # Quick run for realistic regime
LATENCY = 10

# PARAMETERS (Realistic "Daily Driver" Specs ~2025-2026 hardware)
BASE_GATE_NOISE = 0.005   # 0.5% two-qubit gates
MEASURE_NOISE = 0.01      # 1.0% measurement/readout
RESET_NOISE = 0.01        # 1.0% reset
ACTUATION_NOISE = 2e-5

class IntegralController:
    def __init__(self):
        self.state = 0.0
        # Slightly lower gain for better SNR at realistic noise levels
        self.Ki = 0.015
        self.setpoint = 0.0

    def calibrate(self, plant):
        print("Calibrating Controller Setpoint...")
        densities = []
        for _ in range(20):
            _, d = plant.run_batch(0.0)
            densities.append(d)
        self.setpoint = np.mean(densities)
        print(f"Setpoint Locked: {self.setpoint:.5f}")

    def update(self, density):
        error = density - self.setpoint
        self.state += error * self.Ki
        # Reasonable clamp for mild drifts
        self.state = max(-0.015, min(0.015, self.state))
        return self.state

class SurfaceCodePlant:
    def __init__(self, distance=DISTANCE):
        self.d = distance
        self.num_data_qubits = distance**2
        self.per_qubit_drift = np.zeros(self.num_data_qubits)
        self.true_gate_drift = 0.0  # Global component (rare but possible)

    def _gen_circuit(self, p_gate):
        c = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            distance=self.d,
            rounds=QEC_ROUNDS,
            after_clifford_depolarization=p_gate,
            before_measure_flip_probability=MEASURE_NOISE,
            after_reset_flip_probability=RESET_NOISE,
        )
        # Ensure logical observable is defined
        if c.num_observables == 0:
            c.append("OBSERVABLE_INCLUDE", [0], 0)
        return c

    def evolve_drift(self):
        # Mild global drift component
        target_global = 0.002  # Small contribution
        force_global = 0.005 * (target_global - self.true_gate_drift)
        self.true_gate_drift += force_global + np.random.normal(0, 1e-6)

        # Local per-qubit TLS-like drift (main source)
        target_local = 0.010  # Mean drift to reach ~1.5% total gate error
        force_local = 0.01 * (target_local - self.per_qubit_drift)
        noise_local = np.random.normal(0, 2e-6, size=self.num_data_qubits)

        self.per_qubit_drift += force_local + noise_local
        self.per_qubit_drift = np.clip(self.per_qubit_drift, -0.005, 0.025)

        # Effective mean drift
        return self.true_gate_drift + np.mean(self.per_qubit_drift)

    def run_batch(self, correction_val):
        local_offsets = self.per_qubit_drift - correction_val
        avg_local_drift = np.mean(local_offsets)

        # Effective gate error probability
        p_eff = BASE_GATE_NOISE + self.true_gate_drift + avg_local_drift
        p_eff = max(BASE_GATE_NOISE, min(0.2, p_eff))

        # Generate circuit with observable
        noisy_circuit = self._gen_circuit(p_eff)

        # Adaptive decoder
        decoder = pymatching.Matching.from_stim_circuit(noisy_circuit)

        # Sample with separate_observables=True for safety
        sampler = noisy_circuit.compile_detector_sampler()
        detection_events, observable_flips = sampler.sample(shots=BATCH_SIZE, separate_observables=True)

        # observable_flips is (BATCH_SIZE, num_obs), usually num_obs=1
        if observable_flips.ndim > 1 and observable_flips.shape[1] == 1:
            observable_flips = observable_flips.flatten()
        observable_flips = observable_flips.astype(bool)

        # Decode
        preds = decoder.decode_batch(detection_events)
        if isinstance(preds, tuple):
            preds = preds[0]
        if preds.ndim > 1:
            preds = preds.flatten()
        preds = preds.astype(bool)

        failures = np.sum(observable_flips != preds)
        density = np.mean(detection_events)

        return failures, density

def main():
    print(f"Init Plant: d={DISTANCE}, Batch={BATCH_SIZE}")
    print(f"Realistic Mode: Base Gate={BASE_GATE_NOISE*100:.1f}%, Meas={MEASURE_NOISE*100:.1f}%, Target Drift ~1.0% (total ~1.5%)")
    plant = SurfaceCodePlant()
    controller = IntegralController()
    controller.calibrate(plant)

    correction_queue = deque([0.0] * LATENCY, maxlen=LATENCY)
    history = {"cycle": [], "p_gate": [], "cum_fail_std": [], "cum_fail_holo": []}
    cum_std = 0
    cum_holo = 0

    print(">>> RUNNING REALISTIC SIMULATION (10k Cycles) <<<")
    start_time = time.time()

    for t in range(ROUNDS):
        drift_avg = plant.evolve_drift()

        # Standard (no correction)
        fail_std, _ = plant.run_batch(0.0)
        cum_std += fail_std

        # Holo-Neural (with correction)
        u_active = correction_queue.popleft()
        u_noisy = u_active + np.random.normal(0, ACTUATION_NOISE)
        fail_holo, dens_holo = plant.run_batch(u_noisy)
        cum_holo += fail_holo

        # Update controller
        new_u = controller.update(dens_holo)
        correction_queue.append(new_u)

        if t % 50 == 0:
            history["cycle"].append(t)
            history["p_gate"].append(BASE_GATE_NOISE + drift_avg)
            history["cum_fail_std"].append(cum_std)
            history["cum_fail_holo"].append(cum_holo)

            if t % 1000 == 0 or t == ROUNDS - 1:
                print(f"t={t:5d} | Avg Drift={drift_avg:.4f} | p_eff≈{BASE_GATE_NOISE + drift_avg:.4f} | "
                      f"Std Cum Fail={cum_std:,} | Holo Cum Fail={cum_holo:,}")

    duration = time.time() - start_time
    print(f"Simulation completed in {duration:.1f} seconds")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(history["cycle"], history["cum_fail_std"], 'r-', label="Standard (Uncorrected Drift)")
    plt.plot(history["cycle"], history["cum_fail_holo"], 'b-', linewidth=2, label="Holo-Neural (Stabilized)")
    plt.title(f"Realistic Regime: 1% Measurement Error, Drift to ~1.5% Gate Error (d={DISTANCE})")
    plt.xlabel("Error-Correction Cycles")
    plt.ylabel("Cumulative Logical Failures")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("realistic_proof.png", dpi=300)
    print("Plot saved as 'realistic_proof.png'")

if __name__ == "__main__":
    main()