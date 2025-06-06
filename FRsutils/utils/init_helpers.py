


def assign_allowed_kwargs(instance, kwargs: dict, schema: dict):
    """
    @brief Assigns validated kwargs to instance attributes based on a schema.

    @param instance The object to assign attributes to (usually 'self').
    @param kwargs Dictionary of keyword arguments to extract from.
    @param schema Dictionary of {key: spec} where spec may include:
        - 'type': 'float', 'int', 'str', 'bool' (required)
        - 'required': bool (default False)
        - 'default': default value if missing and not required
        - 'range': (min, max) for floats or ints
        - 'allowed': set of allowed values (for strings)

    @throws ValueError, TypeError if validation fails.
    """
    for key, spec in schema.items():
        # Get value: first from kwargs, then default if available
        value = kwargs.get(key, spec.get('default', None))

        if spec.get('required', False) and key not in kwargs:
            raise ValueError(f"Missing required parameter '{key}'.")

        if value is None:
            continue  # skip assigning if optional and not provided

        # Type checking
        expected_type = spec['type']
        if expected_type == 'float' and not isinstance(value, float):
            raise TypeError(f"Parameter '{key}' must be a float.")
        elif expected_type == 'int' and not isinstance(value, int):
            raise TypeError(f"Parameter '{key}' must be an int.")
        elif expected_type == 'str' and not isinstance(value, str):
            raise TypeError(f"Parameter '{key}' must be a string.")
        elif expected_type == 'bool' and not isinstance(value, bool):
            raise TypeError(f"Parameter '{key}' must be a bool.")

        # Range checking
        if 'range' in spec:
            lo, hi = spec['range']
            if lo is not None and value < lo or hi is not None and value > hi:
                raise ValueError(f"Parameter '{key}' must be in range [{lo}, {hi}].")

        # Allowed values for enums (e.g., strategy type)
        if 'allowed' in spec and value not in spec['allowed']:
            raise ValueError(f"Parameter '{key}' must be one of {sorted(spec['allowed'])}.")

        # Assignment
        setattr(instance, key + '_name', value)
