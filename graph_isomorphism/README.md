# Quantum Graph Isomorphism Solver

This project implements a quantum algorithm to solve the graph isomorphism problem using Qiskit. The algorithm uses a quantum SWAP test to compare two graph states and determine if they are isomorphic.

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

The main script `graph_isomorphism.py` provides a `GraphIsomorphismSolver` class that can:
- Compare two graphs using quantum circuits
- Fall back to classical simulation if quantum hardware is not available
- Provide confidence scores for the results

Example usage:
```python
import networkx as nx
from graph_isomorphism import GraphIsomorphismSolver

# Create two isomorphic graphs
G1 = nx.Graph()
G1.add_edges_from([(0, 1), (1, 2), (2, 0)])

G2 = nx.Graph()
G2.add_edges_from([(0, 2), (2, 1), (1, 0)])

# Create solver
solver = GraphIsomorphismSolver(G1, G2)

# Compare graphs
quantum_prob, confidence = solver.quantum_compare(shots=2000)
print(f"Quantum similarity: {quantum_prob:.3f}")
print(f"Confidence: {confidence:.3f}")
```

## Features

- Quantum circuit implementation using SWAP test
- Support for both quantum hardware and simulator
- Error mitigation techniques
- Confidence scoring
- Automatic fallback to simulator if quantum hardware is unavailable

## Implementation Details

The quantum circuit implementation:
1. Creates quantum registers for both graphs
2. Encodes graph structure using controlled-Z gates
3. Performs SWAP test between the two graph states
4. Measures the result

The similarity between graphs is determined by the probability of measuring all zeros in the final state.
