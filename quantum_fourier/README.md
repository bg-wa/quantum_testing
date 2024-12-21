# Quantum Fourier Transform

This project implements the Quantum Fourier Transform (QFT), a quantum analog of the classical discrete Fourier transform. The QFT is a fundamental building block for many quantum algorithms.

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

The implementation provides both QFT and inverse QFT:

```python
from quantum_fourier import QFT

# Create QFT instance for 4 qubits
qft = QFT(num_qubits=4)

# Apply QFT to a quantum circuit
circuit = qft.apply(quantum_circuit)

# Apply inverse QFT
inverse_circuit = qft.apply_inverse(quantum_circuit)
```

## Features

- Implementation of QFT and inverse QFT
- Optimized circuit construction
- Support for arbitrary number of qubits
- Controlled-phase rotation gates
- Circuit visualization tools
- Performance analysis utilities

## Implementation Details

The implementation includes:
1. Hadamard gates for basis transformation
2. Controlled phase rotations
3. SWAP operations for bit reversal
4. Optimization techniques:
   - Circuit depth reduction
   - Gate cancellation
   - Approximation methods for large systems

The QFT circuit uses O(n²) gates for n qubits, with various optimizations available for specific use cases.
