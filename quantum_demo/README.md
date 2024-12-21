# Quantum Computing Demonstration

This project provides a collection of basic quantum computing demonstrations using Qiskit. It includes examples of fundamental quantum operations, gates, and measurements.

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

The demo includes various quantum computing examples:

```python
from quantum_demo import QuantumDemo

# Create a demo instance
demo = QuantumDemo()

# Run Bell state demonstration
demo.create_bell_state()

# Run quantum teleportation
demo.quantum_teleportation()

# Run superposition demonstration
demo.superposition_demo()
```

## Features

- Bell state creation and measurement
- Quantum teleportation implementation
- Superposition and interference demonstrations
- Basic quantum gate operations
- Visualization of quantum states
- Circuit depth analysis

## Implementation Details

The demonstrations cover:
1. Single qubit operations (X, H, Z gates)
2. Multi-qubit operations (CNOT, SWAP)
3. Measurement in different bases
4. Quantum state visualization
5. Circuit optimization techniques

Each example includes:
- Circuit construction
- State preparation
- Measurement and analysis
- Visualization of results
