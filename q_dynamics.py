import numpy as np

from qiskit.aqua.operators.primitive_ops.pauli_op import PauliOp
from qiskit.aqua.operators import (X, Y, Z, I, CX, H, EvolvedOp, PauliTrotterEvolution)

str_to_op = {'X': X, 'Y': Y, 'Z': Z, 'I': I}

def convert_to_op(pauli_str):
    op = str_to_op[pauli_str[0]]
    for i in range(1, len(pauli_str)):
        op ^= str_to_op[pauli_str[i]]
    return op

def get_hamiltonian(num_qubits, h_field_coef):
    h_field = h_field_coef * np.ones(num_qubits)
    ising_chain_ham = 0*convert_to_op('I'*num_qubits)
    # iterate over ZZ terms in Hamiltonian
    for k in range(num_qubits-1):
        pauli_str = 'I'*k + 'ZZ' + 'I'*(num_qubits-k-2)
        op = convert_to_op(pauli_str)
        ising_chain_ham += op
    pauli_str = 'Z' + 'I'*(num_qubits-2) + 'Z'
    op = convert_to_op(pauli_str)
    ising_chain_ham += op
    return ising_chain_ham


def trotter_dynamics(num_qubits, trotter_steps, h_field_coef, seed=1):
    """ trotterization of TF Ising model """
    assert num_qubits > 2
    np.random.seed(seed)

    ising_chain_ham = get_hamiltonian(num_qubits, h_field_coef)

    evo = PauliTrotterEvolution(trotter_mode='suzuki', reps=trotter_steps)
    circ_op = evo.convert(EvolvedOp(ising_chain_ham))
    circuit = circ_op.to_circuit()
    qasm_data = circuit.qasm();
    return qasm_data
