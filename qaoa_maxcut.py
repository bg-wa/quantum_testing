from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram
from qiskit.circuit import Parameter
import numpy as np
from scipy.optimize import minimize
import networkx as nx
import matplotlib.pyplot as plt

class QAOACircuit:
    """Previous class implementation remains the same"""
    def __init__(self, graph, p=1):
        self.graph = graph
        self.n_qubits = len(graph.nodes)
        self.p = p
        self.betas = [Parameter(f'β_{i}') for i in range(p)]
        self.gammas = [Parameter(f'γ_{i}') for i in range(p)]
        self.qc = self.create_circuit()
    
    def create_circuit(self):
        qc = QuantumCircuit(self.n_qubits, self.n_qubits)
        qc.h(range(self.n_qubits))
        for i in range(self.p):
            for edge in self.graph.edges():
                qc.cx(edge[0], edge[1])
                qc.rz(2 * self.gammas[i], edge[1])
                qc.cx(edge[0], edge[1])
            for qubit in range(self.n_qubits):
                qc.rx(2 * self.betas[i], qubit)
        qc.measure(range(self.n_qubits), range(self.n_qubits))
        return qc
    
    def compute_expectation(self, counts):
        expectation = 0
        total_counts = sum(counts.values())
        for bitstring, count in counts.items():
            cut_value = 0
            state = [int(bit) for bit in bitstring]
            for edge in self.graph.edges():
                if state[edge[0]] != state[edge[1]]:
                    cut_value += 1
            expectation += count * cut_value / total_counts
        return expectation
    
    def execute_circuit(self, beta, gamma):
        parameter_dict = {}
        for i in range(self.p):
            parameter_dict[self.betas[i]] = beta[i]
            parameter_dict[self.gammas[i]] = gamma[i]
        qc_bound = self.qc.bind_parameters(parameter_dict)
        backend = Aer.get_backend('qasm_simulator')
        job = execute(qc_bound, backend, shots=1000)
        counts = job.result().get_counts()
        return counts
    
    def objective_function(self, params):
        beta = params[:self.p]
        gamma = params[self.p:]
        counts = self.execute_circuit(beta, gamma)
        expectation = self.compute_expectation(counts)
        return -expectation

def create_example_graph():
    G = nx.Graph()
    G.add_edges_from([
        (0, 1), (1, 2), (2, 3), (3, 0), (0, 2)  # Square with one diagonal
    ])
    return G

def visualize_graph(G, partition=None, ax=None):
    """Modified to use a specific axis"""
    if ax is None:
        ax = plt.gca()
    
    pos = nx.spring_layout(G)
    
    if partition:
        nx.draw_networkx_nodes(G, pos, 
                             nodelist=[n for n in G.nodes if partition[n] == 0],
                             node_color='lightblue', node_size=500, ax=ax)
        nx.draw_networkx_nodes(G, pos, 
                             nodelist=[n for n in G.nodes if partition[n] == 1],
                             node_color='lightgreen', node_size=500, ax=ax)
    else:
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                             node_size=500, ax=ax)
    
    nx.draw_networkx_edges(G, pos, ax=ax)
    nx.draw_networkx_labels(G, pos, ax=ax)
    ax.set_axis_off()

def main():
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 5))
    gs = fig.add_gridspec(1, 3)
    
    # Create example graph
    G = create_example_graph()
    
    # Initial graph visualization
    ax1 = fig.add_subplot(gs[0])
    visualize_graph(G, ax=ax1)
    ax1.set_title("Initial Graph")
    
    # Create and run QAOA
    qaoa = QAOACircuit(G, p=2)
    
    # Initial parameters
    initial_params = np.random.rand(4)  # 2 betas and 2 gammas for p=2
    
    # Classical optimization
    print("Optimizing QAOA parameters...")
    result = minimize(qaoa.objective_function, initial_params, method='COBYLA')
    
    # Get optimal parameters and execute
    optimal_beta = result.x[:qaoa.p]
    optimal_gamma = result.x[qaoa.p:]
    counts = qaoa.execute_circuit(optimal_beta, optimal_gamma)
    
    # Plot QAOA results histogram
    ax2 = fig.add_subplot(gs[1])
    plot_histogram(counts, ax=ax2)
    ax2.set_title("QAOA Results")
    
    # Find and visualize best solution
    best_bitstring = max(counts.items(), key=lambda x: x[1])[0]
    best_partition = [int(bit) for bit in best_bitstring]
    
    ax3 = fig.add_subplot(gs[2])
    visualize_graph(G, best_partition, ax=ax3)
    ax3.set_title("Optimal Partition Found")
    
    # Calculate and print the cut value
    cut_value = sum(1 for (u, v) in G.edges() 
                   if best_partition[u] != best_partition[v])
    
    print(f"\nBest partition found: {best_partition}")
    print(f"Cut value: {cut_value}")
    print(f"Measurement counts: {counts}")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
