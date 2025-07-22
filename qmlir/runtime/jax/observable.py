import jax.numpy as jnp
from ...observable import ObservableExpression, ObservableComposition, Observable


def evaluate_observable(state: jnp.array, num_qubits: int, observable: Observable, little_endian: bool) -> jnp.ndarray:
    if isinstance(observable, ObservableExpression):
        return sum(
            coeff * evaluate_observable(state, num_qubits, term, little_endian) for coeff, term in observable.terms
        )

    elif isinstance(observable, ObservableComposition):
        # Evaluate tensor product: e.g., Z(0) @ Z(1)
        return _evaluate_pauli_string(state, num_qubits, observable.terms, little_endian)

    elif isinstance(observable, Observable):
        # Single observable: treat as composition of length 1
        return _evaluate_pauli_string(state, num_qubits, [observable], little_endian)

    else:
        raise TypeError(f"Unsupported observable type: {type(observable)}")


def _evaluate_pauli_string(state, num_qubits, obs_list: list[Observable], little_endian: bool):
    obs_dict = {
        "I": jnp.eye(2, dtype=complex),
        "X": jnp.array([[0, 1], [1, 0]], dtype=complex),
        "Y": jnp.array([[0, -1j], [1j, 0]], dtype=complex),
        "Z": jnp.array([[1, 0], [0, -1]], dtype=complex),
    }

    # Initialize local observables (one per qubit)
    local_obs_list = [obs_dict["I"] for _ in range(num_qubits)]

    for obs in obs_list:
        if obs.name not in obs_dict:
            raise ValueError(f"Unsupported observable {obs.name}")
        for q in obs.qubits:
            local_obs_list[q] = obs_dict[obs.name]

    # reverse tensor product order if little-endian
    if little_endian:
        local_obs_list = local_obs_list[::-1]

    full_obs = local_obs_list[0]
    for i in range(1, num_qubits):
        full_obs = jnp.kron(full_obs, local_obs_list[i])

    return jnp.vdot(state, full_obs @ state)
