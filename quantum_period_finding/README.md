# Quantum Period Finding Algorithm

This project implements the Quantum Period Finding algorithm, which is a key component of Shor's algorithm for factoring large numbers. The implementation uses Qiskit and demonstrates the quantum Fourier transform and modular arithmetic.

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

The implementation provides functionality to find the period of a modular function:

```python
from quantum_period_finding import PeriodFinder

# Create a period finder instance for f(x) = a^x mod N
# Example: finding period of 2^x mod 15
finder = PeriodFinder(base=2, modulus=15)

# Run the algorithm
period = finder.find_period()
print(f"Period: {period}")
```

## Features

- Implementation of Quantum Fourier Transform
- Modular exponentiation circuit
- Support for both quantum hardware and simulator
- Continued fraction expansion for period extraction
- Error mitigation techniques

## Implementation Details

The algorithm consists of the following steps:
1. Initialize quantum registers
2. Apply modular exponentiation function
3. Perform Quantum Fourier Transform
4. Measure and process results
5. Use continued fraction expansion to find period

The implementation includes optimizations for:
- Circuit depth reduction
- Error mitigation
- Quantum resource optimization
