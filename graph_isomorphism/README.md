# Quantum Graph Isomorphism Solver

A quantum algorithm implementation for solving the graph isomorphism problem using IBM Quantum. This solver uses an optimized quantum circuit design with error mitigation strategies to determine if two graphs are isomorphic.

## Features

- **Optimized Circuit Design**: Uses controlled-Z gates and phase rotations for efficient graph comparison
- **Error Mitigation**: Implements multiple error mitigation techniques:
  - Readout error mitigation with threshold-based filtering
  - Zero-noise extrapolation
  - Confidence-based thresholding
  - Symmetrization of results
- **Automatic Backend Selection**: Dynamically selects the least busy IBMQ backend with sufficient qubits
- **Circuit Optimization**: Includes gate cancellation and combination optimizations
- **Graceful Fallback**: Automatically switches to simulator if quantum hardware is unavailable
- **Visualization Tools**: Built-in functions for graph visualization and comparison

## Requirements

- Python 3.11+
- qiskit==1.3.1
- qiskit-ibm-runtime==0.34.0
- qiskit-aer==0.13.3
- networkx>=3.0
- matplotlib>=3.0
- python-dotenv>=0.19.0

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
Create a `.env` file in the project directory:
```
IBMQ_TOKEN=your_token_here
```

## Usage

```python
import networkx as nx
from graph_isomorphism import GraphIsomorphismSolver

# Create two graphs to compare
G1 = nx.Graph()
G1.add_edges_from([(0, 1), (1, 2), (2, 0)])  # Triangle

G2 = nx.Graph()
G2.add_edges_from([(0, 2), (2, 1), (1, 0)])  # Same triangle, different labeling

# Initialize solver (use_real_qc=True for IBMQ backend)
solver = GraphIsomorphismSolver(G1, G2, use_real_qc=True)

# Create and run the quantum circuit
qc = solver.create_comparison_circuit()
result = solver.run_circuit(qc, shots=1024)

# Get results with error mitigation
probability, confidence = solver.apply_error_mitigation(result, shots=1024)

print(f"Probability of isomorphism: {probability:.3f}")
print(f"Confidence: {confidence:.3f}")
```

## Implementation Details

### Circuit Design
1. **Initialization**: Creates separate quantum registers for each graph and an ancilla qubit
2. **Graph Encoding**: 
   - Uses degree-based phase rotations for vertex comparison
   - Employs controlled-Z gates for edge structure encoding
   - Optimizes neighborhood comparisons using symmetric differences
3. **Measurement**: Measures the ancilla qubit to determine isomorphism probability

### Error Mitigation Strategy
1. **Circuit Level**:
   - Gate cancellation and combination
   - Optimal basis gate transpilation
   - Qubit mapping optimization

2. **Measurement Level**:
   - Threshold-based noise filtering (1% threshold)
   - Zero-noise extrapolation
   - Confidence scoring based on measurement distribution
   - Result symmetrization for clear decision boundaries

### Backend Management
- Automatically selects least busy backend with sufficient qubits
- Implements session-based execution for efficiency
- Includes automatic fallback to simulator with error handling

## Performance Considerations

- **Circuit Depth**: Scales with O(n²) where n is the number of vertices
- **Required Qubits**: Uses 2n + 1 qubits for n-vertex graphs
- **Shot Count**: Recommended 1000+ shots for reliable results
- **Error Rates**: Best results on low-noise quantum processors
- **Classical Overhead**: Minimal, mainly in result post-processing

## Limitations

- Maximum graph size limited by available qubits (2n + 1 required)
- Result quality depends on quantum hardware noise levels
- May require multiple runs for high confidence on noisy hardware
- Queue times on real quantum hardware can affect execution time

## Contributing

Contributions are welcome! Areas for improvement include:
- Additional error mitigation techniques
- Circuit optimization strategies
- Support for directed graphs
- Integration with other quantum algorithms

## License

This project is licensed under the MIT License - see the LICENSE file for details.
