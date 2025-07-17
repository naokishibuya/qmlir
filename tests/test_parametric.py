"""Test parametric circuits functionality."""

import math
from qmlir import QuantumCircuit, Parameter


class TestParameter:
    """Test the Parameter class."""

    def test_parameter_creation_with_value_and_name(self):
        """Test Parameter creation with value and name."""
        theta = Parameter(1.57, "theta")
        assert theta.name == "theta"
        assert theta.initial_value == 1.57
        assert len(theta.id) == 36  # UUID length

    def test_parameter_creation_with_value_only(self):
        """Test Parameter creation with value only (auto-generated name)."""
        phi = Parameter(3.14159)
        assert phi.name.startswith("param_")
        assert phi.initial_value == 3.14159
        assert len(phi.id) == 36

    def test_parameter_creation_with_zero_value(self):
        """Test Parameter creation with zero value and name."""
        omega = Parameter(0.0, "omega")
        assert omega.name == "omega"
        assert omega.initial_value == 0.0

    def test_parameter_equality(self):
        """Test parameter equality based on ID."""
        theta = Parameter(1.57, "theta")
        phi = Parameter(3.14, "phi")
        same_theta = theta

        assert theta == same_theta
        assert theta != phi

    def test_parameter_string_representations(self):
        """Test parameter string representations."""
        theta = Parameter(1.57, "theta")

        assert "initial_value=1.57" in repr(theta)
        assert "theta=1.57" in str(theta)

    def test_parameter_hash(self):
        """Test parameter hashing for use in sets/dicts."""
        theta = Parameter(1.57, "theta")
        phi = Parameter(3.14, "phi")

        param_set = {theta, phi, theta}  # Should deduplicate
        assert len(param_set) == 2


class TestParametricGates:
    """Test parametric gate functionality."""

    def test_parametric_gate_creation(self):
        """Test creating parametric gates."""
        circuit = QuantumCircuit(2)
        theta = Parameter(1.57, "theta")  # π/2
        phi = Parameter(0.785, "phi")  # π/4

        # Add parametric gates
        circuit.rx(0, theta)
        circuit.ry(1, phi)
        circuit.rz(0, theta)  # Reuse parameter

        # Verify gates were added
        assert len(circuit.gates) == 3

        # Check first gate
        rx_gate = circuit.gates[0]
        assert rx_gate.name == "rx"
        assert rx_gate.q == (0,)
        assert len(rx_gate.parameters) == 1
        assert rx_gate.parameters[0] == theta
        assert rx_gate.parameters[0].initial_value == 1.57

        # Check parameter reuse
        rz_gate = circuit.gates[2]
        assert rz_gate.parameters[0] == theta  # Same parameter object
        assert rz_gate.parameters[0].initial_value == 1.57

    def test_mixed_parametric_and_non_parametric_gates(self):
        """Test circuit with both parametric and non-parametric gates."""
        circuit = QuantumCircuit(3)
        theta = Parameter(1.57, "rotation_angle")  # π/2

        # Build a mixed circuit
        circuit.h(0)
        circuit.rx(1, theta)
        circuit.cx(0, 1)
        circuit.ry(2, theta)  # Reuse parameter
        circuit.z(2)

        assert len(circuit.gates) == 5

        # Check that only rx and ry gates have parameters
        param_gates = [g for g in circuit.gates if g.parameters]
        non_param_gates = [g for g in circuit.gates if not g.parameters]

        assert len(param_gates) == 2
        assert len(non_param_gates) == 3

        # Check parameter values
        assert param_gates[0].parameters[0].initial_value == 1.57
        assert param_gates[1].parameters[0].initial_value == 1.57

    def test_common_parameter_patterns(self):
        """Test common parameter usage patterns."""
        # Common rotation angles
        pi_2 = Parameter(math.pi / 2, "pi_2")
        pi_4 = Parameter(math.pi / 4, "pi_4")
        pi = Parameter(math.pi, "pi")

        circuit = QuantumCircuit(3)
        circuit.rx(0, pi_2)  # 90° rotation
        circuit.ry(1, pi_4)  # 45° rotation
        circuit.rz(2, pi)  # 180° rotation

        # Check values
        assert abs(circuit.gates[0].parameters[0].initial_value - math.pi / 2) < 1e-10
        assert abs(circuit.gates[1].parameters[0].initial_value - math.pi / 4) < 1e-10
        assert abs(circuit.gates[2].parameters[0].initial_value - math.pi) < 1e-10


class TestParameterCollection:
    """Test parameter collection and deduplication logic."""

    def test_parameter_deduplication(self):
        """Test parameter collection with deduplication."""
        circuit = QuantumCircuit(2)
        theta = Parameter(1.57, "theta")
        phi = Parameter(0.785, "phi")

        # Create circuit with parameter reuse
        circuit.rx(0, theta)
        circuit.h(1)  # Non-parametric gate
        circuit.ry(1, phi)
        circuit.rz(0, theta)  # Reuse theta

        # Manually test parameter collection logic
        param_index_map = {}
        for gate in circuit.gates:
            for param in gate.parameters:
                if param.id not in param_index_map:
                    param_index_map[param.id] = len(param_index_map)

        # Should have 2 unique parameters
        assert len(param_index_map) == 2
        assert theta.id in param_index_map
        assert phi.id in param_index_map

        # Check deterministic ordering
        assert param_index_map[theta.id] == 0  # First occurrence
        assert param_index_map[phi.id] == 1  # Second occurrence

    def test_parameter_collection_empty_circuit(self):
        """Test parameter collection on circuit with no parametric gates."""
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)

        # Should find no parameters
        param_index_map = {}
        for gate in circuit.gates:
            for param in gate.parameters:
                if param.id not in param_index_map:
                    param_index_map[param.id] = len(param_index_map)

        assert len(param_index_map) == 0

    def test_parameter_collection_single_parameter_multiple_uses(self):
        """Test parameter collection when same parameter is used multiple times."""
        circuit = QuantumCircuit(3)
        theta = Parameter(1.57, "theta")

        # Use same parameter in multiple gates
        circuit.rx(0, theta)
        circuit.ry(1, theta)
        circuit.rz(2, theta)

        # Should only collect one unique parameter
        param_index_map = {}
        for gate in circuit.gates:
            for param in gate.parameters:
                if param.id not in param_index_map:
                    param_index_map[param.id] = len(param_index_map)

        assert len(param_index_map) == 1
        assert theta.id in param_index_map
        assert param_index_map[theta.id] == 0
