# Quantum Phase Estimation

This project implements Quantum Phase Estimation (QPE), a fundamental quantum algorithm used to estimate the eigenphase of a unitary operator. QPE is a key component in many quantum algorithms, including Shor's algorithm.

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

The implementation provides functionality to estimate the phase of a unitary operator:

```python
from phase_estimation import PhaseEstimator

# Create a phase estimator for a custom unitary
estimator = PhaseEstimator(unitary_matrix, precision_qubits=4)

# Run the estimation
phase = estimator.estimate()
print(f"Estimated phase: {phase}")
```

## Features

- Implementation of controlled unitary operations
- Inverse Quantum Fourier Transform
- Configurable precision with additional qubits
- Support for both quantum hardware and simulator
- Error mitigation techniques
- Phase kickback implementation

## Implementation Details

The algorithm consists of the following steps:
1. Initialize precision qubits in superposition
2. Apply controlled unitary operations
3. Apply inverse QFT
4. Measure and process results

The implementation includes optimizations for:
- Circuit depth reduction
- Error mitigation
- Resource optimization for different unitary operators
