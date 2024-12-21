from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram, plot_bloch_multivector
from qiskit.quantum_info import Statevector
import numpy as np
import matplotlib.pyplot as plt

def create_qft_circuit(n_qubits):
    """
    Creates a Quantum Fourier Transform circuit.
    """
    circuit = QuantumCircuit(n_qubits)
    
    def qft_rotations(circuit, n):
        """Performs QFT rotations on the first n qubits"""
        if n == 0:
            return circuit
        n -= 1
        circuit.h(n)
        for qubit in range(n):
            circuit.cp(np.pi/2**(n-qubit), qubit, n)
        qft_rotations(circuit, n)
    
    def swap_registers(circuit, n):
        """Swaps the order of qubits to match standard QFT definition"""
        for qubit in range(n//2):
            circuit.swap(qubit, n-qubit-1)
    
    qft_rotations(circuit, n_qubits)
    swap_registers(circuit, n_qubits)
    return circuit

def create_inverse_qft_circuit(n_qubits):
    """
    Creates an inverse QFT circuit.
    """
    return create_qft_circuit(n_qubits).inverse()

def prepare_input_state(n_qubits, input_state):
    """
    Prepares an input state for QFT demonstration.
    input_state: binary string representing the initial state
    """
    circuit = QuantumCircuit(n_qubits)
    
    # Convert input binary string to list of positions where we need X gates
    for i, bit in enumerate(reversed(input_state)):
        if bit == '1':
            circuit.x(i)
    
    return circuit

def run_qft_demo(n_qubits=3, input_state='101'):
    """
    Demonstrates QFT by:
    1. Preparing an input state
    2. Applying QFT
    3. Measuring the result
    """
    print(f"Performing QFT on input state: |{input_state}⟩")
    print(f"Using {n_qubits} qubits")
    
    # Create main circuit
    main_circuit = QuantumCircuit(n_qubits, n_qubits)
    
    # Prepare input state
    input_circuit = prepare_input_state(n_qubits, input_state)
    main_circuit = main_circuit.compose(input_circuit)
    
    # Apply QFT
    qft = create_qft_circuit(n_qubits)
    main_circuit = main_circuit.compose(qft)
    
    # Add measurements
    main_circuit.measure(range(n_qubits), range(n_qubits))
    
    # Execute the circuit
    backend = Aer.get_backend('qasm_simulator')
    job = execute(main_circuit, backend, shots=1000)
    result = job.result()
    counts = result.get_counts(main_circuit)
    
    # Plot the results
    fig = plot_histogram(counts)
    plt.show()
    
    # Also show the quantum state before measurement
    statevector_backend = Aer.get_backend('statevector_simulator')
    # Remove measurements for statevector simulation
    circuit_no_measure = main_circuit.remove_final_measurements(inplace=False)
    job = execute(circuit_no_measure, statevector_backend)
    state = job.result().get_statevector()
    
    print("\nQuantum state amplitudes after QFT:")
    for i, amplitude in enumerate(state):
        if abs(amplitude) > 0.01:  # Only show non-zero amplitudes
            print(f"|{format(i, f'0{n_qubits}b')}⟩: {amplitude:.3f}")
    
    return counts

if __name__ == "__main__":
    # Run QFT demonstration
    counts = run_qft_demo(n_qubits=3, input_state='101')
    
    print("\nNote: The QFT transforms the input state into a superposition")
    print("of states with complex amplitudes. The measurement results show")
    print("the probability distribution of these superposition states.")
