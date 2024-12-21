from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram
from qiskit.circuit.library import QFT
import numpy as np
import matplotlib.pyplot as plt

def create_eigenstate(circuit, qubit):
    """
    Creates an eigenstate of our unitary operator.
    In this case, we'll use the state |1⟩ which is an eigenstate of the phase rotation.
    """
    circuit.x(qubit)

def controlled_unitary(circuit, control, target, phase):
    """
    Applies a controlled phase rotation.
    This is our unitary operator U whose eigenvalue we want to estimate.
    """
    circuit.cp(phase, control, target)

def create_qpe_circuit(n_counting_qubits, phase):
    """
    Creates a Quantum Phase Estimation circuit.
    
    Args:
        n_counting_qubits (int): Number of qubits used for counting/precision
        phase (float): The phase to estimate (in radians)
    """
    # Total number of qubits = n counting qubits + 1 eigenstate qubit
    n_total_qubits = n_counting_qubits + 1
    
    # Create quantum circuit
    qc = QuantumCircuit(n_total_qubits, n_counting_qubits)
    
    # Prepare eigenstate on the last qubit
    create_eigenstate(qc, n_total_qubits - 1)
    
    # Put counting qubits in superposition
    for qubit in range(n_counting_qubits):
        qc.h(qubit)
    
    # Apply controlled unitary operations
    for i in range(n_counting_qubits):
        # Apply 2^i controlled phase rotations
        for _ in range(2**i):
            controlled_unitary(qc, i, n_total_qubits - 1, phase)
    
    # Apply inverse QFT to counting qubits
    qc.append(QFT(n_counting_qubits, inverse=True), range(n_counting_qubits))
    
    # Measure counting qubits
    qc.measure(range(n_counting_qubits), range(n_counting_qubits))
    
    return qc

def binary_to_phase(binary_string, n_bits):
    """
    Converts a binary measurement result to a phase value.
    """
    value = int(binary_string, 2)
    phase = value / (2**n_bits)
    return phase

def run_phase_estimation(n_counting_qubits=4, true_phase=np.pi/4):
    """
    Runs the Quantum Phase Estimation algorithm.
    
    Args:
        n_counting_qubits (int): Number of qubits for precision
        true_phase (float): The actual phase we're trying to estimate
    """
    print(f"\nRunning Quantum Phase Estimation with {n_counting_qubits} counting qubits")
    print(f"True phase: {true_phase:.6f} radians ({true_phase/np.pi:.6f}π)")
    
    # Create and run the circuit
    qc = create_qpe_circuit(n_counting_qubits, true_phase)
    
    # Use the QASM simulator
    backend = Aer.get_backend('qasm_simulator')
    job = execute(qc, backend, shots=1000)
    result = job.result()
    counts = result.get_counts(qc)
    
    # Analyze and print results
    print("\nMeasurement results and corresponding phase estimates:")
    for measurement, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        estimated_phase = binary_to_phase(measurement, n_counting_qubits)
        estimated_phase_pi = estimated_phase * 2*np.pi / np.pi
        print(f"State |{measurement}⟩: {count} shots")
        print(f"  → Estimated phase: {estimated_phase * 2*np.pi:.6f} radians ({estimated_phase_pi:.6f}π)")
    
    return counts

if __name__ == "__main__":
    # Example: Estimate a phase of π/4
    true_phase = np.pi/4
    
    # Create a figure with subplots for each precision level
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Quantum Phase Estimation Results for Different Precision Levels')
    
    # Try with different precision levels
    for i, n_qubits in enumerate([3, 4, 5]):
        counts = run_phase_estimation(n_qubits, true_phase)
        
        # Plot histogram in the corresponding subplot
        plot_histogram(counts, ax=axes[i])
        axes[i].set_title(f'{n_qubits} Counting Qubits')
        
        print("\n" + "="*50)
    
    plt.tight_layout()
    plt.show()
    
    print("\nNote: As we increase the number of counting qubits,")
    print("the phase estimate becomes more precise.")
