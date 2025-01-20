# Quantum Phase Estimation Implementation

A quantum implementation of Phase Estimation using IBM Quantum. This implementation provides a robust and efficient way to estimate the eigenphase of a unitary operator, a fundamental component in many quantum algorithms including Shor's algorithm and quantum chemistry simulations.

## Features

- **Optimized Circuit Design**:
  - Efficient implementation of controlled unitary operations
  - Automatic calculation of optimal iteration counts
  - Circuit optimization with gate cancellation and combination
  
- **Error Mitigation**:
  - Threshold-based noise filtering
  - Weighted phase estimation
  - Confidence scoring for results
  - Automatic error handling
  
- **Backend Management**:
  - Automatic selection of least busy IBMQ backend
  - Dynamic fallback to simulator if needed
  - Session-based execution for improved efficiency
  
- **Flexible Usage**:
  - Support for arbitrary unitary operators
  - Configurable precision with qubit count
  - Comprehensive result analysis and visualization

## Requirements

- Python 3.11+
- qiskit==1.3.1
- qiskit-ibm-runtime==0.34.0
- qiskit-aer==0.13.3
- numpy>=1.26.2
- matplotlib>=3.8.2
- python-dotenv>=1.0.0
- scipy>=1.11.4

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
Create a `.env` file in the project directory:
```
IBMQ_TOKEN=your_token_here
```

## Usage

```python
from phase_estimation import PhaseEstimator

# Create estimator with 4 counting qubits
# For testing, we can provide a known phase
estimator = PhaseEstimator(
    n_counting_qubits=4,
    phase=np.pi/4,  # Optional, for testing
    use_real_qc=True  # Set to False for simulator
)

# Run estimation with 1024 shots
estimated_phase, confidence = estimator.estimate(shots=1024)

print(f"Estimated phase: {estimated_phase:.6f} radians")
print(f"Confidence: {confidence:.2%}")
```

## Implementation Details

### Circuit Design

1. **Initialization**:
   - Creates quantum registers for counting and eigenstate qubits
   - Prepares eigenstate of the unitary operator
   - Initializes counting qubits in superposition

2. **Phase Estimation**:
   - Applies controlled unitary operations with increasing powers
   - Uses efficient implementation of controlled phase rotations
   - Optimizes circuit depth through gate cancellation

3. **Quantum Fourier Transform**:
   - Applies inverse QFT to extract phase information
   - Uses Qiskit's optimized QFT implementation
   - Automatically handles qubit ordering

4. **Measurement and Analysis**:
   - Measures counting qubits
   - Applies error mitigation
   - Calculates phase estimate and confidence

### Error Mitigation Strategy

1. **Circuit Level**:
   - Gate cancellation and combination
   - Optimal basis gate transpilation
   - Backend-specific optimizations

2. **Measurement Level**:
   - 1% threshold filtering for noise reduction
   - Weighted average phase calculation
   - Confidence scoring based on measurement distribution

### Backend Management

- Automatically selects least busy backend with sufficient qubits
- Implements session-based execution for efficiency
- Includes automatic fallback to simulator with error handling

## Performance Considerations

- **Circuit Depth**: O(n) where n is the number of counting qubits
- **Required Qubits**: n + 1 where n is the number of counting qubits
- **Precision**: Improves exponentially with number of counting qubits
- **Shot Count**: Recommended 1000+ for reliable statistics
- **Error Rates**: Best results on low-noise quantum processors

## Limitations

- Maximum precision limited by available qubits
- Success probability affected by circuit depth and noise
- Queue times on real quantum hardware can affect execution
- Specific unitary operators may require custom implementation

## Contributing

Areas for improvement include:
- Additional error mitigation techniques
- Support for more general unitary operators
- Circuit optimization strategies
- Integration with quantum chemistry applications

## License

This project is licensed under the MIT License - see the LICENSE file for details.
