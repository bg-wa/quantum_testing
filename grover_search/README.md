# Grover's Search Algorithm Implementation

This project implements Grover's Search Algorithm using Qiskit. Grover's algorithm provides a quadratic speedup over classical search algorithms for finding a specific element in an unstructured database.

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

The implementation provides a quantum circuit that can search for a specific marked state in a superposition of all possible states:

```python
from grover_search import GroverSearch

# Create a Grover's search instance for 3 qubits
# searching for state |101⟩
grover = GroverSearch(3, "101")

# Run the algorithm
result = grover.run()
print(f"Most probable state: {result}")
```

## Features

- Implementation of Grover's diffusion operator
- Oracle implementation for marking target states
- Support for both quantum hardware and simulator
- Automatic calculation of optimal number of iterations
- Error mitigation techniques

## Implementation Details

The algorithm consists of the following steps:
1. Initialize qubits in uniform superposition
2. Apply Grover iterations:
   - Oracle marks the target state
   - Diffusion operator amplifies marked state amplitude
3. Measure the result

The number of Grover iterations is calculated as π/4 * √N where N is the size of the search space.
