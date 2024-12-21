from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv

class SATSolver:
    def __init__(self, clauses, num_vars, use_real_qc=True):
        """
        Initialize SAT solver with a list of clauses in CNF form
        Args:
            clauses: List of clauses, where each clause is a list of literals
            num_vars: Number of variables in the SAT problem
            use_real_qc: Whether to use real quantum computer (True) or simulator (False)
        """
        self.clauses = clauses
        self.num_vars = num_vars
        self.use_real_qc = use_real_qc
        
        # Initialize backend
        if self.use_real_qc:
            # Load IBMQ credentials
            load_dotenv()
            token = os.getenv('IBMQ_TOKEN')
            if not token:
                raise ValueError("IBMQ_TOKEN not found in environment variables")
            
            try:
                # Initialize the Qiskit Runtime Service
                service = QiskitRuntimeService(channel="ibm_quantum", token=token)
                
                # Get least busy backend
                available_backends = service.backends(
                    filters=lambda x: x.status().operational 
                    and x.configuration().n_qubits >= self.num_vars
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
                print(f"Error connecting to IBMQ: {str(e)}")
                print("Falling back to simulator.")
                self.use_real_qc = False
                self.backend = AerSimulator()
        else:
            self.backend = AerSimulator()
    
    def create_quantum_circuit(self, gamma, beta):
        """Create quantum circuit for QAOA"""
        qr = QuantumRegister(self.num_vars)
        cr = ClassicalRegister(self.num_vars)
        qc = QuantumCircuit(qr, cr)
        
        # Initial state
        qc.h(qr)
        
        # Cost Hamiltonian
        for clause in self.clauses:
            # Convert clause to list of indices and signs
            indices = []
            signs = []
            for literal in clause:
                if literal > 0:
                    indices.append(literal - 1)
                    signs.append(1)
                else:
                    indices.append(abs(literal) - 1)
                    signs.append(-1)
            
            # Apply phase rotations based on clause satisfaction
            for i in range(len(indices)):
                if signs[i] == -1:
                    qc.x(indices[i])
            
            # Multi-control phase rotation
            if len(indices) == 1:
                qc.rz(2 * gamma, indices[0])
            elif len(indices) == 2:
                qc.cx(indices[0], indices[1])
                qc.rz(2 * gamma, indices[1])
                qc.cx(indices[0], indices[1])
            else:  # len(indices) == 3
                # Use CNOT decomposition for Toffoli to improve fidelity
                qc.h(indices[2])
                qc.cx(indices[1], indices[2])
                qc.tdg(indices[2])
                qc.cx(indices[0], indices[2])
                qc.t(indices[2])
                qc.cx(indices[1], indices[2])
                qc.tdg(indices[2])
                qc.cx(indices[0], indices[2])
                qc.t(indices[2])
                qc.h(indices[2])
                
                qc.rz(2 * gamma, indices[2])
                
                # Inverse Toffoli decomposition
                qc.h(indices[2])
                qc.t(indices[2])
                qc.cx(indices[0], indices[2])
                qc.tdg(indices[2])
                qc.cx(indices[1], indices[2])
                qc.t(indices[2])
                qc.cx(indices[0], indices[2])
                qc.tdg(indices[2])
                qc.cx(indices[1], indices[2])
                qc.h(indices[2])
            
            # Restore basis
            for i in range(len(indices)):
                if signs[i] == -1:
                    qc.x(indices[i])
        
        # Mixer Hamiltonian
        for i in range(self.num_vars):
            qc.rx(2 * beta, i)
        
        # Measure
        qc.measure(qr, cr)
        
        return qc
    
    def solve(self, steps=3, shots=1000):
        """
        Solve SAT problem using QAOA
        Args:
            steps: Number of QAOA steps
            shots: Number of circuit executions
        Returns:
            Best assignment found and its score
        """
        best_score = -float('inf')
        best_assignment = None
        
        # Fewer parameter points for real hardware to reduce runtime
        gamma_points = 3 if self.use_real_qc else 4
        beta_points = 3 if self.use_real_qc else 4
        
        for step in range(steps):
            print(f"\nStep {step + 1}/{steps}")
            
            # Linear parameter schedules
            gamma_vals = np.linspace(0.1, np.pi, gamma_points)
            beta_vals = np.linspace(0.1, np.pi/2, beta_points)
            
            for gamma in gamma_vals:
                for beta in beta_vals:
                    # Create and run circuit
                    qc = self.create_quantum_circuit(gamma, beta)
                    transpiled_qc = transpile(qc, backend=self.backend, optimization_level=3)
                    
                    # Print circuit stats
                    print(f"\nCircuit depth: {transpiled_qc.depth()}")
                    print(f"Circuit gates: {transpiled_qc.count_ops()}")
                    
                    # Run the circuit
                    job = self.backend.run(transpiled_qc, shots=shots)
                    
                    if self.use_real_qc:
                        print(f"Job ID: {job.job_id()}")
                        print("Waiting for job completion...")
                    
                    result = job.result()
                    counts = result.get_counts()
                    
                    # Analyze results
                    for bitstring in counts:
                        if counts[bitstring] > shots/100:  # Filter out noise
                            assignment = [1 if bit == '1' else -1 for bit in bitstring[::-1]]
                            score = self.evaluate_assignment(assignment)
                            if score > best_score:
                                best_score = score
                                best_assignment = assignment
                                print(f"New best assignment found! Score: {score}")
        
        # Clean up IBM Quantum
        if self.use_real_qc:
            pass  # No need to delete account
        
        return best_assignment, best_score
    
    def evaluate_assignment(self, assignment):
        """
        Evaluate an assignment
        Args:
            assignment: List of 1 and -1 values for each variable
        Returns:
            Number of satisfied clauses
        """
        score = 0
        for clause in self.clauses:
            # Check if clause is satisfied
            satisfied = False
            for literal in clause:
                var_idx = abs(literal) - 1
                var_val = assignment[var_idx]
                if literal < 0:
                    var_val = -var_val
                if var_val == 1:
                    satisfied = True
                    break
            if satisfied:
                score += 1
        return score

def main():
    # Example SAT problem: (x1 OR x2) AND (NOT x1 OR x3) AND (NOT x2 OR NOT x3)
    clauses = [[1, 2], [-1, 3], [-2, -3]]
    num_vars = 3
    
    print("Solving SAT problem:")
    print("Clauses:", clauses)
    print("Number of variables:", num_vars)
    
    # Use real quantum computer
    solver = SATSolver(clauses, num_vars, use_real_qc=True)
    assignment, score = solver.solve(steps=2, shots=1000)  # Reduced steps for real hardware
    
    print("\nFinal Results:")
    print("Best assignment:", ["x%d=%d" % (i+1, 1 if x == 1 else 0) for i, x in enumerate(assignment)])
    print("Satisfied clauses:", score, "out of", len(clauses))

if __name__ == "__main__":
    main()
