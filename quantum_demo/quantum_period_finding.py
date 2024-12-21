from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.visualization import plot_histogram
import numpy as np
import matplotlib.pyplot as plt
from fractions import Fraction
from math import gcd

class QuantumPeriodFinding:
    def __init__(self, a, N):
        """
        Initialize period finding for f(x) = a^x mod N
        Args:
            a (int): base of the modular exponentiation
            N (int): modulus
        """
        self.a = a
        self.N = N
        # Number of qubits needed for QPE
        self.n_count = 8  # Precision qubits
        self.n_state = N.bit_length()  # State register size
    
    def create_controlled_ua(self, power):
        """
        Creates a controlled unitary that implements: |y⟩ → |a^(2^p)y mod N⟩
        """
        qc = QuantumCircuit(self.n_state)
        
        # Compute a^(2^power) mod N
        a_power = pow(self.a, pow(2, power), self.N)
        
        # For demonstration, we'll use a simplified version
        # In practice, this would be implemented using quantum arithmetic
        def apply_modular_multiplication(circuit):
            for i in range(self.n_state):
                if (a_power >> i) & 1:
                    circuit.x(i)
        
        apply_modular_multiplication(qc)
        
        return qc.to_gate().control(1)
    
    def create_qpe_circuit(self):
        """
        Creates the Quantum Phase Estimation circuit
        """
        # Create registers
        qr_count = QuantumRegister(self.n_count, 'count')
        qr_state = QuantumRegister(self.n_state, 'state')
        cr = ClassicalRegister(self.n_count, 'c')
        
        qc = QuantumCircuit(qr_count, qr_state, cr)
        
        # Initialize state register to |1⟩
        qc.x(qr_state[0])
        
        # Apply H gates to counting qubits
        for qubit in range(self.n_count):
            qc.h(qubit)
        
        # Apply controlled U operations
        for i in range(self.n_count):
            controlled_ua = self.create_controlled_ua(i)
            qc.append(controlled_ua, [qr_count[i]] + list(qr_state))
        
        # Apply inverse QFT to counting register
        qc.barrier()
        for i in range(self.n_count//2):
            qc.swap(i, self.n_count-i-1)
        
        for j in range(self.n_count):
            for k in range(j):
                qc.cp(-np.pi/float(2**(j-k)), k, j)
            qc.h(j)
        
        # Measure counting register
        qc.barrier()
        qc.measure(qr_count, cr)
        
        return qc
    
    def find_period(self, shots=1000):
        """
        Execute the period finding algorithm
        """
        # Create and run the circuit
        qc = self.create_qpe_circuit()
        
        # Use the QASM simulator
        backend = Aer.get_backend('qasm_simulator')
        job = execute(qc, backend, shots=shots)
        result = job.result()
        counts = result.get_counts(qc)
        
        # Process and analyze results
        measured_phases = []
        for bitstring, count in counts.items():
            phase = int(bitstring, 2) / (2**self.n_count)
            measured_phases.extend([phase] * count)
        
        return counts, measured_phases

def plot_results(counts, phases, a, N, ax=None):
    """Plot the results of period finding"""
    if ax is None:
        ax = plt.gca()
    
    # Create histogram of phases
    n_bins = 50
    ax.hist(phases, bins=n_bins, density=True)
    ax.set_xlabel('Phase')
    ax.set_ylabel('Probability')
    ax.set_title(f'Phase Distribution for a={a}, N={N}')

def find_period_classical(phases, a, N):
    """
    Analyze the phases to find the period
    """
    # Try a few of the most common phases
    unique_phases = np.unique(phases)
    sorted_phases = sorted(unique_phases, key=lambda x: phases.count(x), reverse=True)
    
    for phase in sorted_phases[:5]:  # Try top 5 most common phases
        frac = Fraction(phase).limit_denominator(N)
        r = frac.denominator
        
        # Verify if this is actually the period
        if r > 0 and pow(a, r, N) == 1:
            # Check if this is the smallest valid period
            for i in range(1, r):
                if pow(a, i, N) == 1:
                    r = i
                    break
            return r
    
    return None

def main():
    # Create figure for all plots
    fig = plt.figure(figsize=(15, 5))
    
    # Test cases with known periods
    test_cases = [
        (2, 15),  # Period should be 4
        (7, 15),  # Period should be 4
        (4, 15),  # Period should be 2
    ]
    
    for i, (a, N) in enumerate(test_cases):
        print(f"\nFinding period for f(x) = {a}^x mod {N}")
        
        # Create subplot
        ax = fig.add_subplot(1, 3, i+1)
        
        # Run quantum period finding
        pf = QuantumPeriodFinding(a, N)
        counts, phases = pf.find_period(shots=1000)
        
        # Plot results
        plot_results(counts, phases, a, N, ax)
        
        # Find and verify period
        period = find_period_classical(phases, a, N)
        
        if period is None:
            print("Could not determine period conclusively")
            continue
            
        print(f"Quantum computation suggests period: {period}")
        print(f"Verification: {a}^{period} mod {N} = {pow(a, period, N)}")
        
        # If this was being used for factoring:
        if period and period % 2 == 0:
            candidate1 = gcd(pow(a, period//2) + 1, N)
            candidate2 = gcd(pow(a, period//2) - 1, N)
            if candidate1 * candidate2 == N:
                print(f"Factors found: {candidate1} × {candidate2} = {N}")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
