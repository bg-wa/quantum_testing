from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from itertools import permutations

class GraphIsomorphismSolver:
    def __init__(self, G1, G2):
        """
        Initialize with two graphs to compare
        Args:
            G1, G2: NetworkX graphs to compare
        """
        self.G1 = G1
        self.G2 = G2
        self.n_vertices = len(G1.nodes())
        
        # Verify graphs have same number of vertices and edges
        if len(G2.nodes()) != self.n_vertices:
            raise ValueError("Graphs must have same number of vertices")
        if len(G1.edges()) != len(G2.edges()):
            raise ValueError("Graphs must have same number of edges")
    
    def create_comparison_circuit(self):
        """
        Create a quantum circuit to compare two graph states
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
            # Encode vertex degrees with phase rotations
            deg1 = self.G1.degree[i]
            for _ in range(deg1):
                qc.rz(np.pi/2, qr_g1[i])
            
            deg2 = self.G2.degree[i]
            for _ in range(deg2):
                qc.rz(np.pi/2, qr_g2[i])
            
            # Compare degrees using controlled operations
            if deg1 != deg2:
                qc.cx(qr_g1[i], qr_ancilla)
                qc.cx(qr_g2[i], qr_ancilla)
            
            # Encode and compare neighborhood structure
            neighbors1 = set(self.G1.neighbors(i))
            neighbors2 = set(self.G2.neighbors(i))
            
            # Add phase based on neighborhood size difference
            size_diff = abs(len(neighbors1) - len(neighbors2))
            if size_diff > 0:
                qc.rz(np.pi * size_diff / 2, qr_ancilla)
            
            for j in range(self.n_vertices):
                if j != i:
                    # Encode connectivity with controlled operations
                    if j in neighbors1:
                        qc.cx(qr_g1[i], qr_g1[j])
                        qc.rz(np.pi/2, qr_g1[j])
                    if j in neighbors2:
                        qc.cx(qr_g2[i], qr_g2[j])
                        qc.rz(np.pi/2, qr_g2[j])
                    
                    # Compare connectivity
                    in1 = j in neighbors1
                    in2 = j in neighbors2
                    if in1 != in2:
                        qc.cx(qr_g1[i], qr_ancilla)
                        qc.cx(qr_g2[j], qr_ancilla)
        
        # Final interference
        for i in range(self.n_vertices):
            qc.h(qr_g1[i])
            qc.h(qr_g2[i])
        qc.h(qr_ancilla)
        
        # Measure ancilla
        qc.measure(qr_ancilla, cr)
        
        return qc
    
    def quantum_compare(self, shots=1000):
        """
        Compare graphs using quantum circuit
        Returns probability of graphs being isomorphic
        """
        qc = self.create_comparison_circuit()
        
        # Execute circuit
        backend = Aer.get_backend('qasm_simulator')
        job = execute(qc, backend, shots=shots)
        result = job.result()
        counts = result.get_counts(qc)
        
        # Calculate probability of isomorphism
        prob_iso = counts.get('0', 0) / shots
        return prob_iso
    
    @staticmethod
    def classical_check_isomorphism(G1, G2):
        """
        Classical method to check if graphs are isomorphic
        Returns True if isomorphic, False otherwise
        """
        # Use NetworkX's built-in isomorphism checker
        return nx.is_isomorphic(G1, G2), None

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
    solver = GraphIsomorphismSolver(G1, G2)
    quantum_prob = solver.quantum_compare(shots=1000)
    classical_result, perm = solver.classical_check_isomorphism(G1, G2)
    
    print(f"Quantum similarity: {quantum_prob:.3f}")
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
    solver = GraphIsomorphismSolver(G3, G4)
    quantum_prob = solver.quantum_compare(shots=1000)
    classical_result, perm = solver.classical_check_isomorphism(G3, G4)
    
    print(f"Quantum similarity: {quantum_prob:.3f}")
    print(f"Classical check: {classical_result}")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
