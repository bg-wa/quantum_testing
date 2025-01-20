# Grover's Search Algorithm Implementation

A quantum implementation of Grover's Search Algorithm using IBM Quantum. This implementation includes optimized circuit construction, error mitigation, and automatic backend selection for finding marked states in an unstructured database with quadratic speedup over classical algorithms.

## Features

- **Optimized Circuit Construction**:
  - Automatic calculation of optimal number of iterations
  - Efficient oracle implementation using multi-controlled operations
  - Circuit optimization using gate cancellation and combination
  
- **Error Mitigation**:
  - Readout error mitigation with threshold filtering
  - Confidence-based probability adjustment
  - Automatic noise handling for real quantum hardware
  
- **Backend Management**:
  - Automatic selection of least busy IBMQ backend
  - Dynamic fallback to simulator if needed
  - Session-based execution for improved efficiency

- **Flexible Input Handling**:
  - Support for binary string or list of indices as target state
  - Automatic qubit count validation
  - Runtime parameter optimization

## Requirements

- Python 3.11+
- qiskit==1.3.1
- qiskit-ibm-runtime==0.34.0
- qiskit-aer==0.13.3
- numpy>=1.26.2
- python-dotenv>=1.0.0
- matplotlib>=3.8.2

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
from grover_search import GroverSearch

# Create a Grover's search instance
# Search for state |101⟩ in a 3-qubit system
grover = GroverSearch(
    num_qubits=3,
    target_state="101",
    use_real_qc=True  # Set to False for simulator
)

# Run the search with 1024 shots
results = grover.run(shots=1024)

# Print results (states and their probabilities)
for state, prob in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"|{state}⟩: {prob:.3f}")
```

## Implementation Details

### Circuit Design

1. **Initialization**:
   - Creates quantum register with specified number of qubits
   - Applies Hadamard gates for uniform superposition
   
2. **Oracle Construction**:
   - Efficiently marks target state using X gates and multi-controlled Z
   - Optimized for minimal gate count
   
3. **Grover Operator**:
   - Uses Qiskit's optimized GroverOperator implementation
   - Automatically calculates optimal number of iterations (π/4 * √N)
   
4. **Measurement and Post-processing**:
   - Measures all qubits
   - Applies error mitigation to results
   - Returns probability distribution of states

### Error Mitigation Strategy

1. **Circuit Level**:
   - Gate cancellation and combination
   - Optimal basis gate transpilation
   - Backend-specific optimizations

2. **Measurement Level**:
   - 1% threshold filtering for noise reduction
   - Confidence-based probability adjustment
   - High-probability state emphasis (>40%)

### Performance Considerations

- **Circuit Depth**: O(√N) where N = 2^n for n qubits
- **Success Probability**: Theoretically peaks at π/4 * √N iterations
- **Shot Count**: Recommended 1000+ for reliable statistics
- **Error Rates**: Best results on low-noise quantum processors

## Limitations

- Maximum number of qubits limited by available quantum hardware
- Success probability affected by decoherence and gate errors
- Queue times on real quantum hardware can affect execution
- Optimal for single target state (multiple targets require modification)

## Contributing

Areas for improvement include:
- Multiple target state support
- Additional error mitigation techniques
- Circuit depth optimization
- Alternative oracle implementations

## License

This project is licensed under the MIT License - see the LICENSE file for details.
