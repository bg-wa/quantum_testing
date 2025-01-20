# Quantum Algorithm Implementations

This repository contains implementations of various quantum algorithms using Qiskit and IBM Quantum. Each implementation is designed to run on both quantum simulators and real quantum hardware.

## Project Structure

- `graph_isomorphism/` - Tests if two graphs are isomorphic using quantum circuits and Grover's algorithm
- `grover_search/` - Implementation of Grover's search algorithm for unstructured database search
- `phase_estimation/` - Quantum phase estimation for finding eigenvalues of unitary operators
- `qaoa_maxcut/` - Quantum Approximate Optimization Algorithm (QAOA) for solving MaxCut problems
- `quantum_demo/` - Basic quantum computing demonstrations and tutorials
- `quantum_fourier/` - Quantum Fourier Transform (QFT) implementation with examples
- `quantum_period_finding/` - Period finding algorithm, a key component of Shor's algorithm
- `sat_solver/` - Boolean satisfiability (SAT) solver using Grover's algorithm
- `tsp_solver/` - Quantum approach to the Traveling Salesman Problem

## Key Features

- **Real Quantum Hardware Support**: All implementations can run on IBM Quantum processors
- **Automatic Backend Selection**: Dynamically chooses the least busy quantum backend
- **Error Mitigation**: Includes transpilation optimization and error handling
- **Hybrid Approaches**: Combines quantum and classical processing for optimal results
- **Visualization Tools**: Includes plotting and visualization capabilities for results
- **Modern API Usage**: Uses the latest Qiskit Runtime primitives and session management

## Dependencies

Core dependencies (latest versions):
- `qiskit==1.3.1`
- `qiskit-ibm-runtime==0.34.0`
- `qiskit-aer==0.13.3`
- `python-dotenv==1.0.0`
- `networkx==3.2.1`
- `matplotlib==3.8.2`
- `numpy==1.26.2`

Each subproject may have additional specific dependencies listed in its own `requirements.txt`.

## Getting Started

1. Clone the repository:
```bash
git clone <repository-url>
cd quantum_testing
```

2. Set up IBM Quantum credentials:
Create a `.env` file in the root directory:
```
IBMQ_TOKEN=your_token_here
```

3. Choose an algorithm to work with and navigate to its directory:
```bash
cd algorithm_name
```

4. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
pip install -r requirements.txt
```

## Implementation Highlights

- **Graph Isomorphism**: Uses quantum circuits to test graph isomorphism with automatic backend selection and error mitigation
- **QAOA MaxCut**: Implements QAOA with customizable depth and automatic parameter optimization
- **SAT Solver**: Uses Grover's algorithm with a novel oracle construction for boolean satisfiability
- **Quantum Fourier Transform**: Clean implementation with support for arbitrary qubit counts
- **Phase Estimation**: Includes error mitigation strategies for improved accuracy

## Error Mitigation Strategies

1. **Circuit Optimization**:
   - Automatic transpilation for target hardware
   - Gate cancellation and combination
   - Optimal qubit mapping

2. **Runtime Management**:
   - Session-based execution
   - Automatic backend selection
   - Fallback to simulator when needed

3. **Result Processing**:
   - Multiple shots for statistical significance
   - Post-processing for noise reduction
   - Result verification

## Limitations

- Number of qubits limited by available quantum hardware
- Execution time affected by queue times on real quantum computers
- Results may contain noise, especially on real hardware
- Some algorithms may require significant classical post-processing

## Contributing

Contributions are welcome! Please feel free to submit pull requests for:
- New algorithm implementations
- Improved error mitigation strategies
- Better documentation
- Bug fixes and optimizations

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
