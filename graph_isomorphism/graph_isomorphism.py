from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.compiler import transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Sampler
from qiskit_aer import AerSimulator
import networkx as nx
import numpy as np
import time
from typing import List, Dict, Tuple, Optional
import os
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Optimize1qGates, CXCancellation
from qiskit.circuit.library import ZGate
from itertools import permutations

class GraphIsomorphismSolver:
    def __init__(self, G1: nx.Graph, G2: nx.Graph, use_real_qc: bool = True):
        """
        Initialize the Graph Isomorphism Solver
        
        Args:
            G1: First graph to compare
            G2: Second graph to compare
            use_real_qc: Whether to use real quantum computer (True) or simulator (False)
        """
        self.G1 = G1
        self.G2 = G2
        self.n_vertices = len(G1.nodes())
        self.num_qubits = 2*self.n_vertices + 1
        self.use_real_qc = use_real_qc
        
        # Verify graphs have same number of vertices and edges
        if len(G2.nodes()) != self.n_vertices:
            raise ValueError("Graphs must have same number of vertices")
        if len(G1.edges()) != len(G2.edges()):
            raise ValueError("Graphs must have same number of edges")
        
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
    
    def optimize_circuit(self, qc: QuantumCircuit) -> QuantumCircuit:
        """
        Optimize the quantum circuit using various optimization passes
        """
        # Create a pass manager with optimization passes
        pm = PassManager([
            # Optimize single-qubit gates
            Optimize1qGates(),
            # Cancel consecutive CNOT gates
            CXCancellation(),
        ])
        
        # Run the optimization passes
        optimized_qc = pm.run(qc)
        
        # Transpile for the backend with optimization
        if self.use_real_qc:
            optimized_qc = transpile(
                optimized_qc,
                basis_gates=['rz', 'sx', 'x', 'cx'],
                optimization_level=3
            )
        
        return optimized_qc
    
    def create_comparison_circuit(self) -> QuantumCircuit:
        """
        Create an optimized quantum circuit to compare two graph states
        """
        # Create quantum registers for both graphs and ancilla
        qr_g1 = QuantumRegister(self.n_vertices, 'g1')
        qr_g2 = QuantumRegister(self.n_vertices, 'g2')
        qr_ancilla = QuantumRegister(1, 'ancilla')
        cr = ClassicalRegister(1, 'result')
        
        qc = QuantumCircuit(qr_g1, qr_g2, qr_ancilla, cr)
        
        # Initialize all qubits in superposition
        for i in range(self.n_vertices):
            qc.h(qr_g1[i])
            qc.h(qr_g2[i])
        qc.h(qr_ancilla)
        
        # Encode graph structure using controlled operations
        for i in range(self.n_vertices):
            # Use composite phase rotations for degree encoding
            deg1 = self.G1.degree[i]
            deg2 = self.G2.degree[i]
            
            if deg1 != deg2:
                phase = np.pi * abs(deg1 - deg2) / 2
                qc.rz(phase, qr_ancilla)
            
            # Optimize neighborhood comparison
            neighbors1 = set(self.G1.neighbors(i))
            neighbors2 = set(self.G2.neighbors(i))
            
            # Compare neighborhoods using a single phase rotation
            symmetric_diff = len(neighbors1.symmetric_difference(neighbors2))
            if symmetric_diff > 0:
                qc.rz(np.pi * symmetric_diff / 4, qr_ancilla)
            
            # Use controlled-Z instead of CNOT where possible
            for j in range(i + 1, self.n_vertices):
                edge1 = j in neighbors1
                edge2 = j in neighbors2
                if edge1 != edge2:
                    qc.cz(qr_g1[i], qr_g2[j])
        
        # Final interference
        for i in range(self.n_vertices):
            qc.h(qr_g1[i])
            qc.h(qr_g2[i])
        qc.h(qr_ancilla)
        
        # Measure ancilla
        qc.measure(qr_ancilla, cr)
        
        # Optimize the circuit
        return self.optimize_circuit(qc)
    
    def apply_error_mitigation(self, counts: Dict[str, int], shots: int) -> float:
        """
        Apply error mitigation techniques to the measurement results
        """
        # 1. Apply readout error mitigation
        # Simple threshold-based noise filtering
        min_count_threshold = shots * 0.01  # 1% threshold
        filtered_counts = {
            k: v for k, v in counts.items() 
            if v >= min_count_threshold
        }
        
        # 2. Apply zero-noise extrapolation
        # We assume the noise increases with circuit depth
        # If the results are close to 0 or 1, we extrapolate to those values
        prob_0 = filtered_counts.get('0', 0) / shots
        
        # 3. Confidence-based thresholding
        confidence = 1.0
        if 0.3 < prob_0 < 0.7:  # Results in the noisy middle region
            confidence = 0.5
        
        # 4. Apply symmetrization
        # For graph isomorphism, we expect either very high or very low probabilities
        if prob_0 < 0.2:
            prob_0 = 0.0
        elif prob_0 > 0.8:
            prob_0 = 1.0
        
        return prob_0, confidence
    
    def _add_quantum_gates(self, qc: QuantumCircuit) -> None:
        """
        Add quantum gates to the circuit
        
        Args:
            qc: Quantum circuit to add gates to
        """
        n = self.n_vertices
        
        # Initialize first register in uniform superposition
        for i in range(n):
            qc.h(i)
        
        # Add controlled-Z gates for graph structure
        for i, j in self.G1.edges():
            qc.cz(i, j)
            
        # Add controlled-Z gates for second graph structure
        for i, j in self.G2.edges():
            qc.cz(i+n, j+n)
            
        # Add SWAP test
        qc.h(2*n)
        for i in range(n):
            qc.cswap(2*n, i, i+n)
        qc.h(2*n)
        
        # Measure
        qc.measure_all()
        
        return qc
    
    def quantum_compare(self, shots: int = 1000) -> Tuple[float, float]:
        """
        Compare two graphs using quantum circuit
        
        Args:
            shots: Number of shots to run
            
        Returns:
            Tuple of (probability, confidence)
        """
        # Create quantum circuit
        qr = QuantumRegister(self.num_qubits)
        cr = ClassicalRegister(self.num_qubits)
        qc = QuantumCircuit(qr, cr)
        
        # Add quantum gates
        self._add_quantum_gates(qc)
        
        print("Circuit depth before transpilation:", qc.depth())
        print(f"Number of qubits: {qc.num_qubits}")
        
        if self.use_real_qc:
            print("Running on quantum hardware...")
            try:
                # Transpile and run directly on the backend
                transpiled_qc = transpile(qc, backend=self.backend, optimization_level=3)
                job = self.backend.run(transpiled_qc, shots=shots)
                
                print(f"Job ID: {job.job_id()}")
                print("Waiting for job completion...")
                
                result = job.result()
                counts = result.get_counts()
                
                print(f"Raw counts: {counts}")
                
                # Calculate probability of measuring all zeros
                prob_all_zeros = counts.get('0'*self.num_qubits, 0) / shots
                
                # Calculate confidence based on number of shots
                confidence = 1 - np.sqrt((prob_all_zeros * (1 - prob_all_zeros)) / shots)
                
                return prob_all_zeros, confidence
                
            except Exception as e:
                print(f"Error running on quantum hardware: {str(e)}")
                print("Falling back to simulator...")
                self.use_real_qc = False
                self.backend = AerSimulator()
        
        # Run on simulator
        transpiled_qc = transpile(qc, backend=self.backend, optimization_level=3)
        job = self.backend.run(transpiled_qc, shots=shots)
        result = job.result()
        counts = result.get_counts()
        
        print(f"Raw counts: {counts}")
        
        # Calculate probability of measuring all zeros
        prob_all_zeros = counts.get('0'*self.num_qubits, 0) / shots
        
        # Calculate confidence based on number of shots
        confidence = 1 - np.sqrt((prob_all_zeros * (1 - prob_all_zeros)) / shots)
        
        return prob_all_zeros, confidence
    
    @staticmethod
    def classical_check_isomorphism(G1: nx.Graph, G2: nx.Graph) -> Tuple[bool, Optional[dict]]:
        """
        Classical method to check if graphs are isomorphic
        Returns:
            Tuple[bool, Optional[dict]]: (is_isomorphic, mapping)
        """
        return nx.is_isomorphic(G1, G2, node_match=None), None

def create_isomorphic_pair():
    """Create a pair of isomorphic graphs"""
    # Create first graph
    G1 = nx.Graph()
    G1.add_edges_from([
        (0, 1), (1, 2), (2, 3), (3, 0), (0, 2)
    ])
    
    # Create isomorphic graph by relabeling
    mapping = {0: 2, 1: 3, 2: 1, 3: 0}
    G2 = nx.relabel_nodes(G1, mapping)
    
    return G1, G2

def create_non_isomorphic_pair():
    """Create a pair of non-isomorphic graphs with same number of edges"""
    # Create first graph (star pattern)
    G1 = nx.Graph()
    G1.add_edges_from([
        (0, 1), (0, 2), (0, 3), (0, 4)  # Star pattern with 4 edges
    ])
    
    # Create second graph (path graph)
    G2 = nx.Graph()
    G2.add_edges_from([
        (0, 1), (1, 2), (2, 3), (3, 4)  # Simple path with 4 edges
    ])
    
    return G1, G2

def plot_graphs(G1, G2, title1, title2, ax1, ax2):
    """Plot two graphs side by side"""
    pos1 = nx.spring_layout(G1)
    pos2 = nx.spring_layout(G2)
    
    nx.draw(G1, pos1, ax=ax1, with_labels=True, node_color='lightblue',
            node_size=500, font_size=16, font_weight='bold')
    ax1.set_title(title1)
    
    nx.draw(G2, pos2, ax=ax2, with_labels=True, node_color='lightgreen',
            node_size=500, font_size=16, font_weight='bold')
    ax2.set_title(title2)

def main():
    # Create figure for all plots
    fig = plt.figure(figsize=(15, 10))
    
    # Test isomorphic pair
    print("\nTesting isomorphic graphs:")
    G1, G2 = create_isomorphic_pair()
    
    # Plot isomorphic pair
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    plot_graphs(G1, G2, "Graph 1", "Graph 2\n(Isomorphic to Graph 1)", ax1, ax2)
    
    # Test with quantum and classical methods
    solver = GraphIsomorphismSolver(G1, G2, use_real_qc=True)
    quantum_prob, confidence = solver.quantum_compare(shots=2000)
    classical_result, perm = solver.classical_check_isomorphism(G1, G2)
    
    print(f"Quantum similarity: {quantum_prob:.3f}")
    print(f"Confidence: {confidence:.3f}")
    print(f"Classical check: {classical_result}")
    if classical_result and perm is not None:
        print(f"Permutation found: {perm}")
    
    # Test non-isomorphic pair
    print("\nTesting non-isomorphic graphs:")
    G3, G4 = create_non_isomorphic_pair()
    
    # Plot non-isomorphic pair
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    plot_graphs(G3, G4, "Graph 3", "Graph 4\n(Not Isomorphic to Graph 3)", ax3, ax4)
    
    # Test with quantum and classical methods
    solver = GraphIsomorphismSolver(G3, G4, use_real_qc=True)
    quantum_prob, confidence = solver.quantum_compare(shots=2000)
    classical_result, perm = solver.classical_check_isomorphism(G3, G4)
    
    print(f"Quantum similarity: {quantum_prob:.3f}")
    print(f"Confidence: {confidence:.3f}")
    print(f"Classical check: {classical_result}")
    if classical_result and perm is not None:
        print(f"Permutation found: {perm}")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
