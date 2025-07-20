import pytest
from qmlir.operator import Operator, OperatorComposition, ObservableExpression
from qmlir.operator import X, Y, Z, RX
from qmlir.parameter import Parameter
from qmlir.circuit import QuantumCircuit


def test_basic_operator_repr():
    op = X(0)
    assert isinstance(op, Operator)
    assert op.name == "X"
    assert op.qubits == (0,)
    assert "X|0⟩" in repr(op)


def test_operator_composition():
    op = X(0) @ Z(1)
    assert isinstance(op, OperatorComposition)
    assert len(op.terms) == 2
    assert isinstance(op.terms[0], Operator)
    assert op.terms[0].name == "X"


def test_observable_expression_add():
    H = 0.5 * Z(0) + 0.5 * X(1)
    assert isinstance(H, ObservableExpression)
    assert len(H.terms) == 2
    assert H.terms[0][0] == 0.5
    assert isinstance(H.terms[0][1], Operator)


def test_observable_expression_sub():
    H = Z(0) - X(1)
    assert isinstance(H, ObservableExpression)
    assert len(H.terms) == 2
    assert H.terms[1][0] == -1.0


def test_operator_reverse_sub():
    with pytest.raises(TypeError, match="Cannot add 2.0"):
        2.0 - Z(0)


def test_operator_composition_expression():
    composed = X(0) @ Y(1)
    H = 0.7 * composed + 0.3 * Z(0)
    assert isinstance(H, ObservableExpression)
    assert len(H.terms) == 2
    assert isinstance(H.terms[0][1], OperatorComposition)


def test_rotation_operator_construction():
    theta = Parameter(1.23)
    op = RX(theta)(0)
    assert isinstance(op, Operator)
    assert op.name == "RX"
    assert op.parameters[0].value == 1.23


def test_parameter_negation():
    p = Parameter(0.7, name="theta")
    neg = -p
    assert neg.value == -0.7
    assert neg.name.startswith("-theta")


def test_operator_adds_to_active_circuit():
    c = QuantumCircuit(2)
    with c:
        X(0)
        Y(1)
    assert len(c.operators) == 2
    assert c.operators[0].name == "X"
    assert c.operators[1].name == "Y"


def test_repr_output_for_expression():
    expr = 0.5 * X(0) + Y(1)
    s = repr(expr)
    assert "0.5" in s and "X" in s and "Y" in s


def test_operator_qubit_range_check():
    circuit = QuantumCircuit(2)
    with pytest.raises(ValueError):
        with circuit:
            X(2)  # Invalid qubit index


def test_rotation_gate_factory_lambda():
    rx_op = RX(0.3)
    gate = rx_op(1)
    assert gate.name == "RX"
    assert gate.qubits == (1,)
    assert gate.parameters[0].value == 0.3


def test_operator_sub_expression():
    expr = Z(0) - 0.5 * X(1)
    assert isinstance(expr, ObservableExpression)
    assert len(expr.terms) == 2
    assert expr.terms[1][0] == -0.5


def test_nested_composition_expression():
    term = X(0) @ Y(1) @ Z(2)
    expr = 1.0 * term
    assert isinstance(expr, ObservableExpression)
    assert isinstance(expr.terms[0][1], OperatorComposition)
    assert len(expr.terms[0][1].terms) == 3


def test_operator_repr_has_pipe_symbol():
    rep = repr(X(0))
    assert "|" in rep
