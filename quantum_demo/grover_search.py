from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram
import numpy as np
import matplotlib.pyplot as plt

def create_grover_circuit(n_qubits, marked_state):
    """
    Creates a Grover's algorithm circuit for finding a marked state.
    """
    # Initialize circuit with n qubits and n classical bits
    circuit = QuantumCircuit(n_qubits, n_qubits)
    
    # Initialize all qubits in superposition
    circuit.h(range(n_qubits))
    
    # Number of Grover iterations
    n_iterations = int(np.pi/4 * np.sqrt(2**n_qubits))
    
    for _ in range(n_iterations):
        # Oracle - marks the target state with a phase flip
        # For state |11>, we use CZ gate
        if n_qubits == 2 and marked_state == '11':
            circuit.cz(0, 1)
        
        # Diffusion operator (Grover's operator)
        circuit.h(range(n_qubits))
        circuit.x(range(n_qubits))
        
        # Multi-controlled Z gate
        circuit.h(n_qubits-1)
        circuit.mct(list(range(n_qubits-1)), n_qubits-1)  # Multi-controlled-NOT
        circuit.h(n_qubits-1)
        
        circuit.x(range(n_qubits))
        circuit.h(range(n_qubits))
    
    # Measure all qubits
    circuit.measure(range(n_qubits), range(n_qubits))
    
    return circuit

def run_grover_search(n_qubits=2, marked_state='11'):
    """
    Runs Grover's search algorithm to find a specific marked state.
    """
    print(f"Searching for marked state: |{marked_state}⟩")
    print(f"Using {n_qubits} qubits")
    
    # Create the circuit
    circuit = create_grover_circuit(n_qubits, marked_state)
    
    # Use Aer's qasm_simulator
    backend = Aer.get_backend('qasm_simulator')
    
    # Execute the circuit
    job = execute(circuit, backend, shots=1000)
    result = job.result()
    
    # Get the counts of measurement results
    counts = result.get_counts(circuit)
    
    # Plot the results
    fig = plot_histogram(counts)
    plt.show()
    
    return counts

if __name__ == "__main__":
    # Run Grover's algorithm to search for the state |11⟩
    counts = run_grover_search(n_qubits=2, marked_state='11')
    
    print("\nMeasurement results:", counts)
    print("\nNote: The results should show a high probability of measuring")
    print("the marked state '11' compared to other states.")
