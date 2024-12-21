from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_provider import IBMProvider
from qiskit.circuit import Parameter
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from itertools import permutations
import random
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class TSPSolver:
    def __init__(self, distances, use_real_qc=False, api_token=None):
        """
        Initialize TSP solver
        Args:
            distances: Distance matrix between cities
            use_real_qc: Whether to use real quantum computer
            api_token: IBM Quantum API token
        """
        self.distances = distances
        self.num_cities = len(distances)
        self.use_real_qc = use_real_qc
        self.api_token = api_token
        
        # Initialize simulator
        self.simulator = AerSimulator()
        
        # Try to initialize IBM Quantum if requested
        if use_real_qc:
            if api_token is None:
                print("Warning: No API token provided")
                print("Falling back to simulator")
                self.use_real_qc = False

    def get_backend(self):
        """Get appropriate backend (real QC or simulator)"""
        if self.use_real_qc:
            try:
                # Initialize IBM Quantum provider
                IBMProvider.save_account(token=self.api_token, overwrite=True)
                provider = IBMProvider()
                print("Successfully connected to IBM Quantum")
                
                # Get least busy backend with enough qubits
                backend = None
                for b in provider.backends():
                    if b.configuration().n_qubits >= self.required_qubits():
                        if backend is None or b.status().pending_jobs < backend.status().pending_jobs:
                            backend = b
                
                if backend is None:
                    print("No suitable backend found")
                    print("Falling back to simulator")
                    return AerSimulator()
                
                print(f"\nUsing real quantum computer")
                print(f"Selected backend: {backend.name}")
                return backend
            
            except Exception as e:
                print(f"Error connecting to IBM Quantum: {str(e)}")
                print("Falling back to simulator")
                return AerSimulator()
        else:
            return AerSimulator()
    
    def required_qubits(self):
        """Calculate required number of qubits"""
        return self.num_cities * self.num_cities  # Simplified estimate
    
    def create_cost_hamiltonian(self):
        """
        Create the cost Hamiltonian for the TSP
        Returns coefficients for each term in the Hamiltonian
        """
        terms = []
        coeffs = []
        
        # Add terms for distance costs
        for i in range(self.num_cities):
            for j in range(self.num_cities):
                for k in range(self.num_cities):
                    if k < self.num_cities - 1:
                        # Add cost between consecutive cities
                        next_pos = k + 1
                        for l in range(self.num_cities):
                            if i != l:
                                coeff = self.distances[i][l]
                                q1 = i * self.num_cities + k
                                q2 = l * self.num_cities + next_pos
                                terms.append(([q1, q2], coeff))
        
        # Add constraint terms to ensure one city per position
        constraint_weight = np.max(self.distances) * self.num_cities
        
        # Each position must have exactly one city
        for pos in range(self.num_cities):
            # Sum of qubits for this position should be 1
            pos_qubits = [city * self.num_cities + pos for city in range(self.num_cities)]
            for q1, q2 in permutations(pos_qubits, 2):
                terms.append(([q1, q2], constraint_weight))
        
        # Each city must be visited exactly once
        for city in range(self.num_cities):
            # Sum of qubits for this city should be 1
            city_qubits = [city * self.num_cities + pos for pos in range(self.num_cities)]
            for q1, q2 in permutations(city_qubits, 2):
                terms.append(([q1, q2], constraint_weight))
        
        return terms
    
    def create_mixer_hamiltonian(self):
        """
        Create the mixer Hamiltonian for QAOA
        Returns list of qubit indices to apply X gates
        """
        return list(range(self.required_qubits()))
    
    def create_qaoa_circuit(self, gamma, beta):
        """
        Create QAOA circuit for given parameters
        Args:
            gamma, beta: QAOA parameters
        """
        qr = QuantumRegister(self.required_qubits())
        cr = ClassicalRegister(self.required_qubits())
        qc = QuantumCircuit(qr, cr)
        
        # Initial state: try to create valid tour states
        # Initialize first position
        for city in range(self.num_cities):
            qc.h(city * self.num_cities)
        
        # For other positions, use controlled rotations
        for pos in range(1, self.num_cities):
            pos_qubits = [city * self.num_cities + pos for city in range(self.num_cities)]
            # Create superposition of remaining cities
            qc.h(pos_qubits)
        
        # Cost unitary
        cost_terms = self.create_cost_hamiltonian()
        for qubits, coeff in cost_terms:
            if len(qubits) == 2:
                q1, q2 = qubits
                qc.cx(q1, q2)
                qc.rz(2 * gamma * coeff, q2)
                qc.cx(q1, q2)
        
        # Simple mixer
        for q in range(self.required_qubits()):
            qc.rx(2 * beta, q)
        
        # Measure all qubits
        qc.measure(qr, cr)
        
        return qc
    
    def solve(self, steps=1, shots=1000):
        """
        Solve TSP using QAOA
        Args:
            steps: Number of QAOA steps
            shots: Number of circuit executions
        Returns:
            Best tour found and its cost
        """
        # Initialize parameters
        gamma = Parameter('γ')
        beta = Parameter('β')
        
        # Get appropriate backend
        backend = self.get_backend()
        
        # Create parameterized circuit
        qc = self.create_qaoa_circuit(gamma, beta)
        print(f"\nCircuit depth: {qc.depth()}")
        print(f"Number of qubits: {qc.num_qubits}")
        
        # Try different parameter values
        best_cost = float('inf')
        best_tour = None
        
        # More points in parameter space
        gamma_points = 4
        beta_points = 4
        
        for step in range(steps):
            print(f"\nStep {step + 1}/{steps}")
            
            # Linear parameter schedules with wider range
            gamma_vals = np.linspace(0.1, np.pi, gamma_points)
            beta_vals = np.linspace(0.1, np.pi/2, beta_points)
            
            for gamma_val in gamma_vals:
                for beta_val in beta_vals:
                    # Bind parameters
                    bound_qc = qc.assign_parameters({
                        gamma: gamma_val,
                        beta: beta_val
                    })
                    
                    # Transpile circuit for target backend
                    transpiled_qc = transpile(bound_qc, backend=backend)
                    print(f"\nTranspiled circuit depth: {transpiled_qc.depth()}")
                    print(f"Transpiled circuit gates: {transpiled_qc.count_ops()}")
                    
                    # Execute circuit
                    job = backend.run(transpiled_qc, shots=shots)
                    
                    if self.use_real_qc:
                        print(f"Job ID: {job.job_id}")
                        print("Waiting for job completion...")
                        
                    result = job.result()
                    counts = result.get_counts(transpiled_qc)
                    
                    # Analyze results
                    for bitstring, count in counts.items():
                        if count > shots/100:  # 1% threshold
                            tour = self.bitstring_to_tour(bitstring)
                            if tour is not None:
                                cost = self.calculate_tour_cost(tour)
                                if cost < best_cost:
                                    best_cost = cost
                                    best_tour = tour
                                    print(f"New best tour found! Cost: {cost:.3f}")
        
        # Clean up IBM Quantum
        if self.use_real_qc:
            IBMProvider.delete_account()
        
        return best_tour, best_cost
    
    def bitstring_to_tour(self, bitstring):
        """
        Convert a measurement bitstring to a tour
        Returns None if invalid tour
        """
        # Convert string to array
        bits = np.array([int(b) for b in bitstring])
        tour = [-1] * self.num_cities
        
        # For each position
        for pos in range(self.num_cities):
            # Find which city is in this position
            cities_at_pos = []
            for city in range(self.num_cities):
                if bits[city * self.num_cities + pos]:
                    cities_at_pos.append(city)
            
            # Check validity
            if len(cities_at_pos) != 1:
                return None
            
            tour[pos] = cities_at_pos[0]
        
        # Verify each city appears exactly once
        if len(set(tour)) != self.num_cities:
            return None
        
        return tour
    
    def calculate_tour_cost(self, tour):
        """Calculate total cost of a tour"""
        cost = 0
        for i in range(len(tour)):
            cost += self.distances[tour[i]][tour[(i + 1) % len(tour)]]
        return cost

def create_random_tsp(num_cities, seed=42):
    """Create a random TSP instance"""
    np.random.seed(seed)
    # Generate random city coordinates
    coords = np.random.rand(num_cities, 2)
    
    # Calculate distances
    distances = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(num_cities):
            if i != j:
                distances[i][j] = np.sqrt(
                    (coords[i][0] - coords[j][0])**2 +
                    (coords[i][1] - coords[j][1])**2
                )
    
    return distances, coords

def plot_tour(ax, coords, tour, title=""):
    """Plot a TSP tour on a given axis"""
    # Plot cities
    ax.scatter(coords[:, 0], coords[:, 1], c='red', s=100)
    
    # Plot tour
    if tour is not None:
        for i in range(len(tour)):
            city1 = tour[i]
            city2 = tour[(i + 1) % len(tour)]
            ax.plot([coords[city1][0], coords[city2][0]],
                    [coords[city1][1], coords[city2][1]], 'b-')
    
    # Add city labels
    for i in range(len(coords)):
        ax.annotate(f'City {i}', (coords[i][0], coords[i][1]))
    
    ax.set_title(title)
    ax.grid(True)

def main():
    # Create a small TSP instance (real QC has limited qubits)
    num_cities = 4  # Reduced for real quantum computer
    distances, coords = create_random_tsp(num_cities)
    
    print(f"Solving TSP for {num_cities} cities")
    print("\nDistance matrix:")
    print(distances)
    
    # Create a single figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # Plot city locations
    ax1 = fig.add_subplot(221)
    plot_tour(ax1, coords, None, "City Locations")
    
    # Get API token from environment variable
    api_token = os.getenv('IBMQ_TOKEN')
    use_real_qc = api_token is not None
    
    if not use_real_qc:
        print("Warning: No IBMQ_TOKEN found in environment variables")
        print("Using simulator instead")
    
    # Create solver
    solver = TSPSolver(distances, use_real_qc=use_real_qc, api_token=api_token)
    
    print("\nSolving with QAOA...")
    print(f"Using {'real quantum computer' if use_real_qc else 'simulator'}")
    
    # Increased shots for better statistics
    best_tour, best_cost = solver.solve(steps=3, shots=10000)
    
    print("\nFinal Results:")
    print("Best tour found:", best_tour)
    print("Tour cost:", best_cost)
    
    # Plot QAOA solution
    ax2 = fig.add_subplot(222)
    if best_tour is not None:
        plot_tour(ax2, coords, best_tour, f"QAOA Solution\nCost: {best_cost:.2f}")
    else:
        plot_tour(ax2, coords, None, "QAOA Solution\nNo valid tour found")
    
    # Compare with classical nearest neighbor solution
    print("\nComparing with classical nearest neighbor solution...")
    
    def nearest_neighbor_tsp(distances):
        n = len(distances)
        unvisited = set(range(1, n))
        tour = [0]  # Start from city 0
        
        while unvisited:
            last = tour[-1]
            next_city = min(unvisited, key=lambda x: distances[last][x])
            tour.append(next_city)
            unvisited.remove(next_city)
        
        return tour
    
    nn_tour = nearest_neighbor_tsp(distances)
    nn_cost = sum(distances[nn_tour[i]][nn_tour[(i + 1) % num_cities]] 
                 for i in range(num_cities))
    
    print("Nearest neighbor tour:", nn_tour)
    print("Nearest neighbor cost:", nn_cost)
    
    if best_tour is not None:
        print(f"QAOA solution is {(best_cost/nn_cost - 1)*100:.1f}% away from nearest neighbor solution")
    
    # Plot classical solution
    ax3 = fig.add_subplot(223)
    plot_tour(ax3, coords, nn_tour, 
             f"Classical Nearest Neighbor Solution\nCost: {nn_cost:.2f}")

    # Plot cost comparison
    ax4 = fig.add_subplot(224)
    if best_tour is not None:
        costs = [best_cost, nn_cost]
        labels = ['QAOA', 'Nearest Neighbor']
    else:
        costs = [nn_cost]
        labels = ['Nearest Neighbor']
    
    ax4.bar(labels, costs)
    ax4.set_title('Solution Cost Comparison')
    ax4.set_ylabel('Tour Cost')
    ax4.grid(True)

    # Adjust layout and display
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
