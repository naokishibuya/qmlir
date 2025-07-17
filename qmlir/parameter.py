"""Parameter class for parametric quantum circuits.

This module defines the Parameter class used for symbolic parameters in quantum gates.
Parameters are always differentiable and can be used with automatic differentiation
frameworks like JAX and PyTorch.
"""

import uuid


class Parameter:
    """Represents a symbolic parameter for quantum gates.

    Parameters are symbolic placeholders that can be used in parametric quantum gates
    like RX, RY, RZ. They are always differentiable and designed to work with
    automatic differentiation frameworks.

    Each parameter has a unique identifier to handle parameter deduplication during
    compilation.
    """

    def __init__(self, initial_value: float, name: str = None):
        """Initialize a parameter.

        Args:
            initial_value: Initial value for the parameter. Required for meaningful
                          parameter usage in quantum circuits.
            name: Optional name for the parameter. If not provided, a unique name
                  will be generated.
        """
        self.id = str(uuid.uuid4())  # Unique identifier for deduplication
        self.name = name or f"param_{self.id[:8]}"  # Human-readable name
        self.initial_value = float(initial_value)  # Store as float

    def __repr__(self) -> str:
        """Return string representation of the parameter."""
        return f"Parameter(name='{self.name}', initial_value={self.initial_value}, id='{self.id[:8]}...')"

    def __str__(self) -> str:
        """Return string representation for display."""
        return f"{self.name}={self.initial_value}"

    def __eq__(self, other) -> bool:
        """Check equality based on unique ID."""
        if not isinstance(other, Parameter):
            return False
        return self.id == other.id

    def __hash__(self) -> int:
        """Hash based on unique ID."""
        return hash(self.id)
