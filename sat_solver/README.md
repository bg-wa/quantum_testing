# Quantum SAT Solver

This project implements a quantum algorithm for solving Boolean Satisfiability (SAT) problems using Grover's algorithm. It can run on both local simulators and real quantum hardware through IBM Quantum.

## Requirements

- Python 3.11+
- Qiskit 1.3.1
- qiskit-ibm-runtime 0.34.0
- qiskit-aer 0.13.3
- Other dependencies listed in `requirements.txt`

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

The solver can handle SAT problems in CNF form. Here's an example:

```python
from sat_solver import SATSolver

# Example SAT problem: (x1 OR x2) AND (NOT x1 OR x2)
clauses = [
    [(0, True), (1, True)],      # x1 OR x2
    [(0, False), (1, True)]      # NOT x1 OR x2
]

# Create solver instance (use_ibmq=True to use IBMQ backend)
solver = SATSolver(clauses, use_ibmq=True)

# Solve with specified number of shots
is_sat, solution = solver.solve(shots=1024)

if is_sat:
    print("Solution found!")
    print("Variable assignments:", solution)
else:
    print("No solution found.")
```

## Features

- Quantum circuit implementation using Grover's algorithm
- Support for both local simulator and IBMQ quantum hardware
- Automatic backend selection based on required qubits
- Circuit transpilation for optimal execution
- Error handling and graceful fallback to simulator
- Support for CNF formula input
- Solution verification

## Implementation Details

The solver implements:

1. **Oracle Construction**: Creates a quantum oracle that marks satisfying assignments
2. **Grover's Algorithm**: Applies amplitude amplification to find solutions
3. **IBMQ Integration**: 
   - Automatic selection of least busy backend
   - Circuit transpilation for the target hardware
   - Session management for efficient job execution
4. **Result Processing**:
   - Measurement aggregation
   - Classical post-processing
   - Solution verification

## Error Mitigation

The implementation includes several error mitigation strategies:
- Circuit optimization through transpilation
- Automatic fallback to simulator if no suitable quantum backend is available
- Session-based execution to reduce overhead
- Multiple shots to improve reliability

## Limitations

- The number of variables in the SAT problem is limited by the available qubits on the quantum hardware
- Real quantum hardware may have longer execution times due to queue times
- Results may contain some noise, especially on real quantum hardware
