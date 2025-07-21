import jax.numpy as jnp
from ...operator import ObservableExpression, OperatorComposition, Operator, Observable


def evaluate_observable(state: jnp.array, num_qubits: int, observable: Observable, little_endian: bool) -> jnp.ndarray:
    if isinstance(observable, ObservableExpression):
        return sum(
            coeff * evaluate_observable(state, num_qubits, term, little_endian) for coeff, term in observable.terms
        )

    elif isinstance(observable, OperatorComposition):
        # Evaluate tensor product: e.g., Z(0) @ Z(1)
        return _evaluate_pauli_string(state, num_qubits, observable.terms, little_endian)

    elif isinstance(observable, Operator):
        # Single operator: treat as composition of length 1
        return _evaluate_pauli_string(state, num_qubits, [observable], little_endian)

    else:
        raise TypeError(f"Unsupported observable type: {type(observable)}")


def _evaluate_pauli_string(state, num_qubits, ops: list[Operator], little_endian: bool):
    import jax.numpy as jnp

    op_dict = {
        "I": jnp.eye(2, dtype=complex),
        "X": jnp.array([[0, 1], [1, 0]], dtype=complex),
        "Y": jnp.array([[0, -1j], [1j, 0]], dtype=complex),
        "Z": jnp.array([[1, 0], [0, -1]], dtype=complex),
    }

    # Initialize local operators (one per qubit)
    local_ops = [op_dict["I"] for _ in range(num_qubits)]

    for op in ops:
        if op.name not in op_dict:
            raise ValueError(f"Unsupported operator {op.name}")
        for q in op.qubits:
            local_ops[q] = op_dict[op.name]

    # reverse tensor product order if little-endian
    if little_endian:
        local_ops = local_ops[::-1]

    full_op = local_ops[0]
    for i in range(1, num_qubits):
        full_op = jnp.kron(full_op, local_ops[i])

    return jnp.vdot(state, full_op @ state)
