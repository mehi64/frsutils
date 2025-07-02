
# Good design and programming
  - Since checking input data is time taking and not always necessary, we added a parameter to constructor of classes to turn input checking on/off.
  - default value for this param (checking input data) is True to check when nothing is given. That is more safe.


# Classes inherit RegistryFactoryMixin
  - All __inits__ must call validate_params because create method uses __init__ therefore in just one place, validation is checked. Even in factories, inside create, __init__ function is called.
  - Make sure you implement validate_params in your new class.
  - in case of need make sure to_dict() and from_dict() is implemented as needed.
  - Do not forget to add class-level docstring. This is returned automatically by calling help() function
  - validate_params() must always be overloaded
  - validate_params() gets cls, not self.
  - @property name, needs to be updated inside RegistryFactoryMixin when a new class added (pay attention to naming convention)
  - nameing convention for new classes is ??????????????????????????????????
  


