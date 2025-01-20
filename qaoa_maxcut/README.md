# QAOA MaxCut Solver

A quantum implementation of the Quantum Approximate Optimization Algorithm (QAOA) for solving the MaxCut problem using IBM Quantum. This implementation includes optimized circuit construction, error mitigation, and automatic backend selection.

## Features

- **Optimized Circuit Design**:
  - Efficient implementation of problem and mixer Hamiltonians
  - Automatic parameter optimization
  - Circuit optimization with gate cancellation
  - Backend-specific transpilation
  
- **Error Handling**:
  - Automatic backend selection and fallback
  - Session-based execution
  - Comprehensive error reporting
  - Result validation
  
- **Classical Optimization**:
  - Multiple optimizer support (COBYLA, Nelder-Mead)
  - Random parameter initialization
  - Convergence monitoring
  - Parameter bounds handling
  
- **Visualization**:
  - Graph partition visualization
  - Measurement statistics
  - Cut value analysis
  - Interactive result plots

## Requirements

- Python 3.11+
- qiskit==1.3.1
- qiskit-ibm-runtime==0.34.0
- qiskit-aer==0.13.3
- qiskit-optimization==0.5.0
- numpy>=1.26.2
- networkx>=3.2.1
- matplotlib>=3.8.2
- python-dotenv>=1.0.0
- scipy>=1.11.4

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
from qaoa_maxcut import QAOAMaxCut
import networkx as nx

# Create a graph
G = nx.Graph()
G.add_edges_from([
    (0, 1), (1, 2), (2, 3), (3, 0),  # Square
    (0, 2), (1, 3)  # Diagonals
])

# Initialize QAOA solver
solver = QAOAMaxCut(
    graph=G,
    p=2,  # Number of QAOA layers
    use_real_qc=True  # Set to False for simulator
)

# Run the algorithm
counts, optimal_params = solver.solve(
    shots=1024,
    optimizer='COBYLA'
)

# Analyze results
bitstring, cut_value, partition = solver.analyze_results(counts)

print(f"Best partition: {partition}")
print(f"Cut value: {cut_value}")
```

## Implementation Details

### Circuit Design

1. **State Preparation**:
   - Initialize all qubits in superposition
   - Efficient Hadamard gate application
   
2. **QAOA Layers**:
   - Problem unitary (phase separation)
   - Mixer unitary (XY mixing)
   - Parameter optimization
   
3. **Measurement**:
   - Multi-qubit measurement
   - Result aggregation
   - Statistical analysis

### Optimization Strategy

1. **Parameter Initialization**:
   - Uniform random initialization
   - Bounded parameter ranges
   - Multiple random starts
   
2. **Classical Optimization**:
   - Gradient-free optimization
   - Convergence criteria
   - Parameter refinement
   
3. **Result Processing**:
   - Cut value calculation
   - Partition extraction
   - Solution validation

### Backend Management

- Automatic selection of least busy backend
- Dynamic fallback to simulator
- Session-based execution
- Error handling and recovery

## Performance Considerations

- **Circuit Depth**: O(p|E|) where p is the number of QAOA layers and |E| is the number of edges
- **Qubit Count**: Equal to the number of vertices in the graph
- **Shot Count**: Recommended 1000+ for reliable statistics
- **Classical Optimization**: Scales with the number of QAOA parameters (2p)

## Limitations

- Graph size limited by available qubits
- Solution quality depends on:
  - Number of QAOA layers
  - Optimization convergence
  - Quantum hardware noise
- Queue times on real quantum hardware
- Classical optimization overhead

## Contributing

Areas for improvement include:
- Additional classical optimizers
- Improved error mitigation
- Parameter initialization strategies
- Hardware-efficient circuit variants
- Support for weighted graphs

## License

This project is licensed under the MIT License - see the LICENSE file for details.
