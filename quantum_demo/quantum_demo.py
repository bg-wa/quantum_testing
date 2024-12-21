from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

def create_bell_state():
    """
    Creates a Bell state (maximally entangled state) using two qubits.
    This demonstrates quantum entanglement, a key quantum phenomenon.
    """
    # Create a quantum circuit with 2 qubits and 2 classical bits
    circuit = QuantumCircuit(2, 2)
    
    # Put the first qubit in superposition
    circuit.h(0)
    
    # Create entanglement between qubits
    circuit.cx(0, 1)
    
    # Measure both qubits
    circuit.measure([0,1], [0,1])
    
    return circuit

def run_quantum_circuit():
    """
    Executes the quantum circuit and displays results
    """
    # Create the quantum circuit
    circuit = create_bell_state()
    
    # Use Aer's qasm_simulator
    simulator = Aer.get_backend('qasm_simulator')
    
    # Execute the circuit on the simulator
    job = execute(circuit, simulator, shots=1000)
    result = job.result()
    
    # Get the counts of measurement results
    counts = result.get_counts(circuit)
    
    # Plot the results
    fig = plot_histogram(counts)
    plt.show()
    
    return counts

if __name__ == "__main__":
    print("Creating and executing a Bell State circuit...")
    counts = run_quantum_circuit()
    print("\nMeasurement results:", counts)
    print("\nNote: The results should show approximately equal distribution between '00' and '11' states,")
    print("demonstrating quantum entanglement and superposition.")
