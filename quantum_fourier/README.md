# Quantum Fourier Transform (QFT)

A quantum implementation of the Quantum Fourier Transform, a fundamental building block for many quantum algorithms including Shor's factoring algorithm, quantum phase estimation, and quantum signal processing.

## Features

- **Core QFT Operations**:
  - Forward QFT implementation
  - Inverse QFT (IQFT) implementation
  - Optimized circuit depth
  - Automatic qubit reordering
  
- **Circuit Optimizations**:
  - Controlled phase rotation synthesis
  - Gate cancellation and combination
  - Approximate QFT support
  - Dynamic circuit depth reduction
  
- **Error Mitigation**:
  - Noise-aware gate scheduling
  - Error detection capabilities
  - Robust phase estimation
  - Statistical error analysis
  
- **Visualization Tools**:
  - Circuit diagrams
  - Bloch sphere visualization
  - State vector plots
  - Phase space representations

## Requirements

- Python 3.8+
- qiskit==0.45.0
- qiskit-ibm-runtime==0.11.0
- qiskit-aer==0.12.2
- numpy>=1.20.0
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

3. Configure IBM Quantum access:
Create a `.env` file in the project directory:
```
IBMQ_TOKEN=your_token_here
```

## Usage Examples

### Basic QFT

```python
from quantum_fourier import QuantumFourier

# Initialize QFT with 4 qubits
qft = QuantumFourier(n_qubits=4)

# Create and apply QFT circuit
circuit = qft.create_circuit()
result = qft.run(circuit)

# Visualize results
qft.plot_results(result)
```

### Inverse QFT

```python
# Apply inverse QFT
iqft_circuit = qft.create_inverse_circuit()
result = qft.run(iqft_circuit)
```

### Approximate QFT

```python
# Create approximate QFT with reduced gates
approx_qft = QuantumFourier(n_qubits=4, approximation_degree=2)
circuit = approx_qft.create_approximate_circuit()
```

## Implementation Details

### Circuit Construction

1. **State Preparation**:
   - Initialize quantum register
   - Apply preprocessing gates
   - Configure measurement basis

2. **Core QFT Components**:
   - Hadamard transformations
   - Controlled phase rotations
   - Qubit reordering operations

3. **Optimization Techniques**:
   - Gate cancellation patterns
   - Commutation rules
   - Circuit depth minimization
   - Approximate QFT truncation

### Mathematical Foundation

The QFT transforms a quantum state according to:
```
|j⟩ → 1/√N ∑_{k=0}^{N-1} e^{2πijk/N} |k⟩
```

where:
- N = 2^n (n is number of qubits)
- j, k are computational basis states
- The transformation preserves quantum superposition

### Performance Characteristics

- **Circuit Depth**: O(n²) for exact QFT, O(n log n) for approximate
- **Gate Count**: O(n²) controlled-phase rotations
- **Error Scaling**: Error grows linearly with circuit depth
- **Memory Requirements**: O(n) classical memory

## Advanced Features

### Error Mitigation

- Dynamical decoupling sequences
- Noise-resilient gate decompositions
- Error detection circuits
- Statistical error bounds

### Circuit Analysis

- Gate-level profiling
- Error rate estimation
- Resource estimation
- Circuit optimization metrics

### Visualization Options

- Quantum circuit diagrams
- State vector evolution
- Phase space representations
- Error distribution plots

## Applications

1. **Quantum Phase Estimation**:
   - Eigenvalue estimation
   - Quantum chemistry
   - Period finding

2. **Quantum Signal Processing**:
   - Signal analysis
   - Frequency estimation
   - Quantum filtering

3. **Quantum Algorithms**:
   - Shor's algorithm
   - HHL algorithm
   - Quantum simulation

## Contributing

Areas for potential improvement:
- Additional optimization techniques
- Hardware-specific optimizations
- Enhanced error mitigation
- New visualization tools
- Application-specific variants

## References

1. Nielsen, M. A. & Chuang, I. L. (2010). Quantum Computation and Quantum Information
2. Coppersmith, D. (2002). An approximate Fourier transform useful in quantum factoring
3. Qiskit Textbook: Quantum Fourier Transform

## License

This project is licensed under the MIT License - see the LICENSE file for details.
