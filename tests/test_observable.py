import pytest
from qmlir.observable import Observable, ObservableComposition, ObservableExpression
from qmlir.operator import X, Y, Z


def test_basic_operator_repr():
    obs = X(0)
    assert isinstance(obs, Observable)
    assert obs.name == "X"
    assert obs.qubits == (0,)
    assert "X|0⟩" in repr(obs)


def test_operator_composition():
    obs = X(0) @ Z(1)
    assert isinstance(obs, ObservableComposition)
    assert len(obs.terms) == 2
    assert isinstance(obs.terms[0], Observable)
    assert obs.terms[0].name == "X"


def test_observable_expression_add():
    H = 0.5 * Z(0) + 0.5 * X(1)
    assert isinstance(H, ObservableExpression)
    assert len(H.terms) == 2
    assert H.terms[0][0] == 0.5
    assert isinstance(H.terms[0][1], Observable)


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
    assert isinstance(H.terms[0][1], ObservableComposition)


def test_repr_output_for_expression():
    expr = 0.5 * X(0) + Y(1)
    s = repr(expr)
    assert "0.5" in s and "X" in s and "Y" in s


def test_operator_sub_expression():
    expr = Z(0) - 0.5 * X(1)
    assert isinstance(expr, ObservableExpression)
    assert len(expr.terms) == 2
    assert expr.terms[1][0] == -0.5


def test_nested_composition_expression():
    term = X(0) @ Y(1) @ Z(2)
    expr = 1.0 * term
    assert isinstance(expr, ObservableExpression)
    assert isinstance(expr.terms[0][1], ObservableComposition)
    assert len(expr.terms[0][1].terms) == 3


def test_operator_repr_has_pipe_symbol():
    rep = repr(X(0))
    assert "|" in rep
