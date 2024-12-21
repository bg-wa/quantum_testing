# QAOA MaxCut Solver

This project implements the Quantum Approximate Optimization Algorithm (QAOA) to solve the MaxCut problem. QAOA is a hybrid quantum-classical algorithm that can find approximate solutions to combinatorial optimization problems.

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

The implementation solves the MaxCut problem for given graphs:

```python
from qaoa_maxcut import QAOAMaxCut
import networkx as nx

# Create a graph
G = nx.random_regular_graph(3, 6)

# Initialize QAOA solver
solver = QAOAMaxCut(G, p=2)  # p is the number of QAOA layers

# Run the algorithm
result = solver.solve()
print(f"Maximum cut found: {result.cut_value}")
print(f"Cut configuration: {result.configuration}")
```

## Features

- Implementation of QAOA circuit for MaxCut
- Classical optimization of QAOA parameters
- Support for both quantum hardware and simulator
- Visualization of results and cut configurations
- Performance analysis and comparison with classical solutions
- Parameterized circuit depth (p-layers)

## Implementation Details

The algorithm consists of the following components:
1. Problem Hamiltonian encoding
2. Mixer Hamiltonian implementation
3. QAOA circuit construction
4. Parameter optimization
5. Measurement and classical post-processing

The implementation includes:
- Efficient parameter initialization
- Gradient-based optimization
- Multiple measurement shots for statistical accuracy
- Result visualization and analysis tools
