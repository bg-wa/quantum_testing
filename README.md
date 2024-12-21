# Quantum Computing Demonstrations

This project contains several quantum computing demonstrations using IBM's Qiskit framework. Each example showcases different fundamental aspects of quantum computing and quantum algorithms.

## Setup
1. Create a virtual environment:
```bash
python3 -m venv venv
```

2. Activate the virtual environment:
```bash
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Environment Setup
1. Create a `.env` file in the project root with your IBM Quantum token:
```bash
IBMQ_TOKEN=your_token_here
```

## Demonstrations

### 1. Basic Quantum Circuits (`quantum_demo.py`)
Demonstrates fundamental quantum computing concepts:
- Quantum superposition using Hadamard gates
- Quantum entanglement using Bell states
- Basic quantum measurements

Run the demo:
```bash
python quantum_demo.py
```

### 2. Grover's Search Algorithm (`grover_search.py`)
Implements Grover's quantum search algorithm:
- Searches for a marked state in a quantum superposition
- Demonstrates quadratic speedup over classical search
- Shows amplitude amplification of the target state

Run the demo:
```bash
python grover_search.py
```

### 3. Quantum Fourier Transform (`quantum_fourier.py`)
Demonstrates the Quantum Fourier Transform (QFT):
- Quantum version of the classical Fourier transform
- Shows how QFT transforms quantum states
- Includes visualization of the transformed states

Run the demo:
```bash
python quantum_fourier.py
```

### 4. Quantum Phase Estimation (`phase_estimation.py`)
Implements Quantum Phase Estimation (QPE):
- Estimates eigenvalues of quantum operators
- Uses QFT as a key component
- Shows increasing precision with more qubits

Run the demo:
```bash
python phase_estimation.py
```

### 5. QAOA for MaxCut (`qaoa_maxcut.py`)
Implements the Quantum Approximate Optimization Algorithm:
- Solves the MaxCut graph partitioning problem
- Demonstrates hybrid quantum-classical optimization
- Visualizes the problem graph and solution

Run the demo:
```bash
python qaoa_maxcut.py
```

### 6. SAT Solver using QAOA (`sat_solver.py`)
Implements a quantum SAT solver using QAOA:
- Solves boolean satisfiability problems
- Uses real IBM Quantum hardware
- Shows circuit optimization and error mitigation
- Demonstrates hybrid quantum-classical optimization

Run the demo:
```bash
python sat_solver.py
```

## Key Features
- All demonstrations include visualizations
- Results are shown in single-window displays
- Each example includes detailed output and explanations
- Implementations follow Qiskit best practices

## Requirements
- Python 3.8+
- Qiskit and Qiskit Aer
- Qiskit IBM Runtime (for real quantum hardware)
- Python-dotenv (for managing IBM Quantum credentials)
- NumPy
- Matplotlib
- NetworkX (for QAOA)
- SciPy (for optimization)

## Notes
- These demonstrations run on Qiskit's QASM simulator
- Results may vary slightly between runs due to the probabilistic nature of quantum measurements
- The number of qubits and circuit depths are chosen to be suitable for simulation on classical hardware

## References
- [Qiskit Documentation](https://qiskit.org/documentation/)
- [IBM Quantum Computing](https://quantum-computing.ibm.com/)
- Matthew Brisse's work on practical quantum computing applications
