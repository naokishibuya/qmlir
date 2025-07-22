from typing import Tuple


class Observable:
    def __init__(self, name: str, qubits: Tuple[int, ...]):
        self.name = name
        self.qubits = qubits

    def __repr__(self):
        return f"{self.name}|{', '.join(map(str, self.qubits))}⟩"

    def __add__(self, other: "Observable") -> "ObservableExpression":
        return ObservableExpression([(1.0, self)]) + other

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self + (-1.0 * other)

    def __rsub__(self, other):
        return (-1.0 * self) + other

    def __matmul__(self, other: "Observable") -> "Observable":
        if isinstance(other, ObservableComposition):
            return ObservableComposition([self] + other.terms)
        return ObservableComposition([self, other])

    def __rmul__(self, scalar: float) -> "ObservableExpression":
        return ObservableExpression([(scalar, self)])


class ObservableComposition(Observable):
    def __init__(self, terms: list[Observable]):
        # validate disjoint qubits
        qset = set()
        for obs in terms:
            for q in obs.qubits:
                if q in qset:
                    raise ValueError(f"Duplicate qubit {q} in tensor product")
                qset.add(q)
        self.terms = terms

    def __repr__(self):
        return " @ ".join(map(str, self.terms))

    @property
    def qubits(self):
        return tuple(q for obs in self.terms for q in obs.qubits)

    def __matmul__(self, other: "Observable") -> "Observable":
        if isinstance(other, ObservableComposition):
            return ObservableComposition(self.terms + other.terms)
        return ObservableComposition(self.terms + [other])


class ObservableExpression(Observable):
    def __init__(self, terms: list[tuple[float, Observable]]):
        self.terms = terms

    def __repr__(self):
        return " + ".join(f"{coeff}*({obs})" for coeff, obs in self.terms)

    def __add__(self, other):
        if isinstance(other, ObservableExpression):
            return ObservableExpression(self.terms + other.terms)
        elif isinstance(other, Observable):
            return ObservableExpression(self.terms + [(1.0, other)])
        else:
            raise TypeError(f"Cannot add {other} to ObservableExpression")

    def __rmul__(self, scalar):
        return ObservableExpression([(scalar * c, o) for c, o in self.terms])
