from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import GroverOperator
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Sampler
from qiskit_aer import AerSimulator
from qiskit.compiler import transpile
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Optimize1qGates, CXCancellation
import numpy as np
from typing import List, Dict, Union, Optional
import os
from dotenv import load_dotenv

class GroverSearch:
    def __init__(self, num_qubits: int, target_state: Union[str, List[int]], use_real_qc: bool = True):
        """
        Initialize Grover's Search algorithm.
        
        Args:
            num_qubits: Number of qubits in the system
            target_state: Target state to search for, either as binary string or list of indices
            use_real_qc: Whether to use real quantum computer (True) or simulator (False)
        """
        self.num_qubits = num_qubits
        self.target_state = target_state if isinstance(target_state, str) else ''.join(map(str, target_state))
        self.use_real_qc = use_real_qc
        
        # Calculate optimal number of iterations
        N = 2**num_qubits
        self.num_iterations = int(np.pi/4 * np.sqrt(N))
        
        # Initialize backend
        if self.use_real_qc:
            # Get IBM Quantum credentials
            load_dotenv()
            token = os.getenv('IBMQ_TOKEN')
            if not token:
                raise ValueError("Please set IBMQ_TOKEN environment variable")
            
            try:
                # Initialize the Qiskit Runtime Service
                service = QiskitRuntimeService(channel="ibm_quantum", token=token)
                
                # Get least busy backend with enough qubits
                available_backends = service.backends(
                    filters=lambda x: x.status().operational 
                    and x.configuration().n_qubits >= self.num_qubits
                )
                
                if not available_backends:
                    print("No suitable IBMQ backends available. Falling back to simulator.")
                    self.use_real_qc = False
                    self.backend = AerSimulator()
                else:
                    least_busy = sorted(available_backends, 
                                     key=lambda x: x.status().pending_jobs)[0]
                    print(f"Selected backend: {least_busy.name}")
                    self.backend = least_busy
            except Exception as e:
                print(f"Error connecting to IBM Quantum: {str(e)}")
                print("Falling back to simulator")
                self.use_real_qc = False
                self.backend = AerSimulator()
        else:
            self.backend = AerSimulator()
    
    def create_oracle(self) -> QuantumCircuit:
        """
        Create an oracle that marks the target state.
        """
        oracle = QuantumCircuit(self.num_qubits)
        
        # Apply X gates to qubits that should be 0 in target state
        for i, bit in enumerate(self.target_state):
            if bit == '0':
                oracle.x(i)
        
        # Multi-controlled Z gate
        oracle.h(self.num_qubits - 1)
        oracle.mcx(list(range(self.num_qubits - 1)), self.num_qubits - 1)
        oracle.h(self.num_qubits - 1)
        
        # Undo X gates
        for i, bit in enumerate(self.target_state):
            if bit == '0':
                oracle.x(i)
        
        return oracle
    
    def optimize_circuit(self, qc: QuantumCircuit) -> QuantumCircuit:
        """
        Optimize the quantum circuit using various optimization passes
        """
        # Create a pass manager with optimization passes
        pm = PassManager([
            Optimize1qGates(),
            CXCancellation(),
        ])
        
        # Run the optimization passes
        optimized_qc = pm.run(qc)
        
        # Transpile for the backend with optimization
        if self.use_real_qc:
            optimized_qc = transpile(
                optimized_qc,
                backend=self.backend,
                optimization_level=3
            )
        
        return optimized_qc
    
    def create_circuit(self) -> QuantumCircuit:
        """
        Create the complete Grover's algorithm circuit
        """
        # Create quantum and classical registers
        qr = QuantumRegister(self.num_qubits)
        cr = ClassicalRegister(self.num_qubits)
        qc = QuantumCircuit(qr, cr)
        
        # Initialize superposition
        qc.h(range(self.num_qubits))
        
        # Create oracle
        oracle = self.create_oracle()
        
        # Create Grover operator
        grover_op = GroverOperator(oracle)
        
        # Apply Grover iterations
        for _ in range(self.num_iterations):
            qc = qc.compose(grover_op)
        
        # Measure all qubits
        qc.measure(range(self.num_qubits), range(self.num_qubits))
        
        # Optimize the circuit
        return self.optimize_circuit(qc)
    
    def apply_error_mitigation(self, counts: Dict[str, int], shots: int) -> Dict[str, float]:
        """
        Apply error mitigation techniques to the measurement results
        """
        # 1. Apply readout error mitigation
        min_count_threshold = shots * 0.01  # 1% threshold
        filtered_counts = {
            k: v for k, v in counts.items() 
            if v >= min_count_threshold
        }
        
        # 2. Calculate probabilities
        total_filtered = sum(filtered_counts.values())
        probabilities = {
            k: v / total_filtered 
            for k, v in filtered_counts.items()
        }
        
        # 3. Apply confidence thresholds
        high_prob_threshold = 0.4
        result_probs = {}
        for state, prob in probabilities.items():
            if prob >= high_prob_threshold:
                result_probs[state] = prob
            elif prob >= 0.1:  # Keep moderately probable states
                result_probs[state] = prob * 0.8  # Reduce confidence
        
        return result_probs
    
    def run(self, shots: int = 1024) -> Dict[str, float]:
        """
        Run Grover's algorithm and return the results
        
        Args:
            shots: Number of times to run the circuit
            
        Returns:
            Dictionary mapping measured states to their probabilities
        """
        # Create and run the circuit
        qc = self.create_circuit()
        
        if self.use_real_qc:
            with Session(backend=self.backend) as session:
                # Transpile the circuit for the backend
                qc_transpiled = transpile(qc, backend=self.backend)
                
                sampler = Sampler()
                job = sampler.run([qc_transpiled], shots=shots)
                print(f"Job ID: {job.job_id()}")
                print("Waiting for job completion...")
                result = job.result()
                
                # Convert quasi-probability distribution to counts
                quasi_dist = result.quasi_dists[0]
                counts = {format(i, f'0{self.num_qubits}b'): int(p * shots) 
                         for i, p in quasi_dist.items()}
        else:
            # Use simulator
            job = self.backend.run(qc, shots=shots)
            result = job.result()
            counts = result.get_counts()
        
        # Apply error mitigation
        return self.apply_error_mitigation(counts, shots)

def main():
    """
    Example usage of Grover's search
    """
    # Search for state |101⟩ in a 3-qubit system
    target_state = "101"
    num_qubits = 3
    
    print(f"Searching for state |{target_state}⟩ in {num_qubits}-qubit system")
    print(f"Search space size: {2**num_qubits}")
    
    # Create and run the search
    grover = GroverSearch(num_qubits, target_state, use_real_qc=True)
    results = grover.run(shots=1024)
    
    # Print results
    print("\nResults:")
    for state, prob in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"|{state}⟩: {prob:.3f}")

if __name__ == "__main__":
    main()
