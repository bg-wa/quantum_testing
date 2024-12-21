# Quantum SAT Solver

This project implements a quantum algorithm for solving Boolean Satisfiability (SAT) problems. It uses a hybrid quantum-classical approach combining QAOA principles with SAT-specific optimizations.

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

The implementation can solve SAT problems in CNF form:

```python
from sat_solver import QuantumSATSolver

# Define a SAT problem in CNF form
clauses = [
    [1, -2, 3],  # x1 OR NOT x2 OR x3
    [-1, 2, 4],  # NOT x1 OR x2 OR x4
    [2, -3, -4]  # x2 OR NOT x3 OR NOT x4
]

# Create solver instance
solver = QuantumSATSolver(clauses)

# Solve the problem
result = solver.solve()
print(f"Satisfying assignment: {result.assignment}")
print(f"Is satisfiable: {result.is_satisfiable}")
```

## Features

- Support for CNF formula input
- Quantum circuit optimization for SAT problems
- Hybrid quantum-classical solving approach
- Multiple solving strategies:
  - Pure quantum approach
  - QAOA-based solver
  - Hybrid decomposition
- Solution verification
- Performance analysis tools

## Implementation Details

The algorithm implements:
1. Problem encoding into quantum Hamiltonian
2. Circuit construction for clause satisfaction
3. Optimization of quantum parameters
4. Measurement and classical post-processing
5. Solution verification

Optimizations include:
- Clause reduction techniques
- Dynamic circuit depth adjustment
- Error mitigation strategies
- Classical preprocessing
