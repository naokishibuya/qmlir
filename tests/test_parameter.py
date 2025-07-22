from qmlir.parameter import Parameter


class TestParameter:
    """Test Parameter class."""

    def test_parameter_creation(self):
        """Test basic parameter creation."""
        param = Parameter(0.5, name="theta")
        assert param.value == 0.5
        assert param.name == "theta"
        assert param.id is not None

    def test_parameter_creation_without_name(self):
        """Test parameter creation without explicit name."""
        param = Parameter(0.5)
        assert param.value == 0.5
        assert param.name.startswith("param_")
        assert param.id is not None

    def test_parameter_unique_ids(self):
        """Test that parameters get unique IDs."""
        param1 = Parameter(0.5, name="theta1")
        param2 = Parameter(0.7, name="theta2")
        assert param1.id != param2.id

    def test_parameter_repr(self):
        """Test parameter string representation."""
        param = Parameter(0.5, name="theta")
        assert "theta" in repr(param)
        assert "0.5" in repr(param)

    def test_parameter_str(self):
        """Test parameter string conversion."""
        param = Parameter(0.5, name="theta")
        assert "theta=0.5" in str(param)

    def test_parameter_equality(self):
        """Test parameter equality."""
        param1 = Parameter(0.5, name="theta")
        param2 = Parameter(0.5, name="theta")
        param3 = Parameter(0.7, name="theta")

        # Parameters with same values but different IDs should not be equal
        assert param1 != param2

        # Different values should not be equal
        assert param1 != param3

    def test_parameter_hash(self):
        """Test that parameters can be used in sets/dicts."""
        param1 = Parameter(0.5, name="theta1")
        param2 = Parameter(0.7, name="theta2")

        param_set = {param1, param2}
        assert len(param_set) == 2
        assert param1 in param_set
        assert param2 in param_set

    def test_parameter_float_conversion(self):
        """Test parameter float conversion."""
        param = Parameter(0.5, name="theta")
        assert param.value == 0.5

    def test_parameter_negative_value(self):
        """Test parameter with negative value."""
        param = Parameter(-0.5, name="negative_theta")
        assert param.value == -0.5

    def test_parameter_negative_param(self):
        """Test parameter with negative value."""
        param = -Parameter(0.5, name="negative_theta")
        assert param.value == -0.5

    def test_parameter_zero_value(self):
        """Test parameter with zero value."""
        param = Parameter(0.0, name="zero_theta")
        assert param.value == 0.0
