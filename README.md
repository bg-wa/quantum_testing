# Quantum Algorithm Implementations

This repository contains implementations of various quantum algorithms using Qiskit. Each algorithm is organized in its own directory with specific dependencies and documentation.

## Project Structure

- `graph_isomorphism/` - Quantum algorithm for testing graph isomorphism
- `grover_search/` - Implementation of Grover's search algorithm
- `phase_estimation/` - Quantum phase estimation algorithm
- `qaoa_maxcut/` - QAOA solver for the MaxCut problem
- `quantum_demo/` - Basic quantum computing demonstrations
- `quantum_fourier/` - Quantum Fourier Transform implementation
- `quantum_period_finding/` - Quantum period finding algorithm
- `sat_solver/` - Quantum SAT solver
- `tsp_solver/` - Quantum solver for Traveling Salesman Problem

## Getting Started

Each project has its own `requirements.txt` and setup instructions. To get started with a specific algorithm:

1. Navigate to the algorithm's directory:
```bash
cd algorithm_name
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up IBM Quantum credentials:
Create a `.env` file with your IBM Quantum token:
```
IBMQ_TOKEN=your_token_here
```

## Common Dependencies

All projects use:
- Qiskit 0.45.0
- Qiskit IBM Runtime 0.11.0
- Qiskit Aer 0.12.2

Additional dependencies are specified in each project's `requirements.txt` file.

## Features

- Real quantum hardware support via IBM Quantum
- Automatic fallback to simulation
- Error mitigation techniques
- Visualization tools
- Performance analysis utilities

## Contributing

Feel free to open issues or submit pull requests for improvements to any of the implementations.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
