from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Sampler, Options
import numpy as np
from typing import List, Tuple
import os
from dotenv import load_dotenv

class SATSolver:
    def __init__(self, clauses, num_vars, use_ibmq=True):
        """
        Initialize SAT solver with a list of clauses in CNF form
        Args:
            clauses: List of clauses, where each clause is a list of literals
            num_vars: Number of variables in the SAT problem
            use_ibmq: Whether to use IBMQ (True) or simulator (False)
        """
        self.clauses = clauses
        self.num_vars = num_vars
        self.use_ibmq = use_ibmq
        
        if self.use_ibmq:
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
                    and x.configuration().n_qubits >= self.num_vars + 1  # +1 for ancilla
                )
                
                if not available_backends:
                    print("No suitable IBMQ backends available. Falling back to simulator.")
                    self.use_ibmq = False
                    self.backend = AerSimulator()
                else:
                    self.backend = sorted(available_backends, 
                                     key=lambda x: x.status().pending_jobs)[0]
                    print(f"Selected backend: {self.backend.name}")
            except Exception as e:
                print(f"Error connecting to IBM Quantum: {str(e)}")
                print("Falling back to simulator")
                self.use_ibmq = False
                self.backend = AerSimulator()
        else:
            from qiskit_aer import AerSimulator
            self.backend = AerSimulator()
    
    def apply_oracle(self, qc: QuantumCircuit, qr: QuantumRegister, anc: QuantumRegister):
        """Apply the oracle operations to the given circuit"""
        for i, clause in enumerate(self.clauses):
            # Create clause verification with unique name
            clause_anc = QuantumRegister(1, f'clause_anc_{i}')
            qc.add_register(clause_anc)
            qc.x(clause_anc)
            
            # Check each literal in the clause
            for var, is_positive in clause:
                if not is_positive:
                    qc.x(qr[var])
                qc.cx(qr[var], clause_anc[0])
                if not is_positive:
                    qc.x(qr[var])
            
            # If clause is satisfied, flip the phase
            qc.x(clause_anc)
            qc.h(anc)
            qc.cx(clause_anc[0], anc[0])
            qc.h(anc)
            qc.x(clause_anc)
            
            # Uncompute clause ancilla
            for var, is_positive in reversed(clause):
                if not is_positive:
                    qc.x(qr[var])
                qc.cx(qr[var], clause_anc[0])
                if not is_positive:
                    qc.x(qr[var])
    
    def create_grover_circuit(self, num_iterations=2) -> QuantumCircuit:
        """Create complete Grover's algorithm circuit"""
        # Create quantum registers
        qr = QuantumRegister(self.num_vars, 'q')
        anc = QuantumRegister(1, 'ancilla')
        cr = ClassicalRegister(self.num_vars, 'c')
        qc = QuantumCircuit(qr, anc, cr)
        
        # Initialize all qubits in superposition
        qc.h(qr)
        qc.x(anc)
        qc.h(anc)
        
        # Grover iteration
        for _ in range(num_iterations):
            # Oracle
            self.apply_oracle(qc, qr, anc)
            
            # Diffusion operator
            qc.h(qr)
            qc.x(qr)
            qc.h(qr[self.num_vars-1])
            qc.mcx(qr[:-1], qr[self.num_vars-1])  # Using mcx instead of deprecated mct
            qc.h(qr[self.num_vars-1])
            qc.x(qr)
            qc.h(qr)
        
        # Measure
        qc.measure(qr, cr)
        
        return qc
    
    def solve(self, shots=1024):
        """
        Solve SAT problem using Grover's algorithm
        Returns:
            Best assignment found and its score
        """
        # Calculate optimal number of iterations
        n = self.num_vars
        num_iterations = max(1, int(np.pi/4 * np.sqrt(2**n)))
        
        # Create the circuit
        qc = self.create_grover_circuit(num_iterations)
        
        if self.use_ibmq:
            # Use IBMQ backend with runtime
            with Session(backend=self.backend) as session:
                # Transpile the circuit for the backend
                qc_transpiled = transpile(qc, backend=self.backend)
                
                sampler = Sampler()  # Create the sampler within the session
                job = sampler.run([qc_transpiled], shots=shots)
                print(f"Job ID: {job.job_id()}")
                print("Waiting for job completion...")
                result = job.result()
                
                # Extract bitstring from the result
                bitarray = result[0].data.c._array
                counts = {}
                for bitstring in map(lambda x: ''.join(str(int(bit)) for bit in x), bitarray):
                    counts[bitstring] = counts.get(bitstring, 0) + 1
        else:
            # Use simulator
            job = self.backend.run(qc, shots=shots)
            result = job.result()
            counts = result.get_counts()
        
        # Find the most frequent measurement
        max_count = 0
        best_bitstring = None
        for bitstring in counts:
            if counts[bitstring] > max_count:
                max_count = counts[bitstring]
                best_bitstring = bitstring
        
        # Convert bitstring to solution
        if best_bitstring:
            # For IBMQ results, convert from integer to binary string
            if self.use_ibmq:
                best_bitstring = format(int(best_bitstring), f'0{self.num_vars}b')
            
            solution = [bit == '1' for bit in best_bitstring[:self.num_vars]]
            # Verify if it's a valid solution
            if self.verify_solution(solution):
                return True, solution
        
        return False, []
    
    def verify_solution(self, assignment):
        """Verify if an assignment satisfies all clauses"""
        for clause in self.clauses:
            # Check if clause is satisfied
            satisfied = False
            for var, is_positive in clause:
                if assignment[var] == is_positive:
                    satisfied = True
                    break
            if not satisfied:
                return False
        return True

def parse_dimacs(dimacs_str: str) -> List[List[Tuple[int, bool]]]:
    """
    Parse a DIMACS CNF format string into a clause list.
    
    Args:
        dimacs_str: DIMACS CNF format string
    
    Returns:
        List[List[Tuple[int, bool]]]: Clause list
    """
    clauses = []
    for line in dimacs_str.strip().split('\n'):
        line = line.strip()
        if line and not line.startswith('c') and not line.startswith('p'):
            clause = []
            for lit in line.split()[:-1]:  # Ignore the trailing 0
                var = int(lit)
                clause.append((abs(var) - 1, var > 0))
            clauses.append(clause)
    return clauses

def main():
    # Example SAT problem in DIMACS CNF format
    # This represents (x1 OR x2) AND (NOT x1 OR x2)
    dimacs = """
    c Example SAT problem
    p cnf 2 2
    1 2 0
    -1 2 0
    """
    
    # Parse the DIMACS format
    clauses = parse_dimacs(dimacs)
    
    print("Solving SAT problem...")
    print("Clauses:", clauses)
    solver = SATSolver(clauses, 2, use_ibmq=True)  # Use IBMQ backend
    is_sat, solution = solver.solve(shots=1024)
    
    if is_sat:
        print("Solution found!")
        print("Variable assignments:", ["x%d=%d" % (i+1, int(x)) for i, x in enumerate(solution)])
    else:
        print("No solution exists.")

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    main()
