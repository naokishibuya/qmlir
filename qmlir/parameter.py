import uuid


class Parameter:
    def __init__(self, value: float, name: str = None):
        self.id = str(uuid.uuid4())  # Unique identifier for deduplication
        self.name = name or f"param_{self.id[:8]}"  # Human-readable name
        self.value = float(value)  # Store as float

    def __repr__(self) -> str:
        return f"Parameter(name='{self.name}', value={self.value}, id='{self.id[:8]}...')"

    def __str__(self) -> str:
        return f"{self.name}={self.value}"

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Parameter):
            return False
        return self.id == other.id

    def __neg__(self) -> "Parameter":
        return Parameter(-self.value, name=f"-{self.name}" if self.name else None)
