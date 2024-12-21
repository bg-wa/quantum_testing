# Quantum TSP Solver

This project implements a quantum algorithm for solving the Traveling Salesman Problem (TSP) using a hybrid quantum-classical approach. It combines QAOA principles with TSP-specific optimizations.

## Requirements

- Python 3.8+
- Qiskit 0.45.0
- IBM Quantum account (for running on real quantum hardware)

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up IBM Quantum credentials:
Create a `.env` file in the project directory with your IBM Quantum token:
```
IBMQ_TOKEN=your_token_here
```

## Usage

The implementation can solve TSP instances:

```python
from tsp_solver import QuantumTSPSolver
import networkx as nx

# Create a complete graph with distances
G = nx.complete_graph(4)
for (u, v) in G.edges():
    G[u][v]['weight'] = np.random.randint(1, 10)

# Create solver instance
solver = QuantumTSPSolver(G)

# Solve the problem
result = solver.solve()
print(f"Optimal tour: {result.tour}")
print(f"Tour length: {result.length}")
```

## Features

- Support for various TSP instance formats
- Multiple solving strategies:
  - QAOA-based approach
  - Quantum annealing simulation
  - Hybrid decomposition
- Tour visualization
- Performance analysis tools
- Support for both symmetric and asymmetric TSP
- Integration with TSPLIB format

## Implementation Details

The implementation includes:
1. Problem encoding into QUBO format
2. Circuit construction for tour constraints
3. Optimization of quantum parameters
4. Measurement and classical post-processing
5. Tour validation and optimization

Optimizations include:
- Problem size reduction techniques
- Dynamic circuit depth adjustment
- Error mitigation strategies
- Classical preprocessing
- Local search improvements
