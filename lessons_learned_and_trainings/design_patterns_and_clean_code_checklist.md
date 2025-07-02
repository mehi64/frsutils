# ✅ FRsutils Design Patterns & Clean Code Strategy Checklist

## 🔁 Design Patterns

| Pattern                  | Application & Enforcement in Code |
|--------------------------|------------------------------------|
| **Factory Method**       | Each pluggable component (e.g., `TNorm`, `Implicator`) defines a `create(name, **kwargs)` method for dynamic instantiation. |
| **Registry Pattern**     | Each abstract base class maintains a `_registry` and `_aliases` to track registered subclasses using `@register()` decorators. |
| **Template Method**      | Abstract base classes define the structure of algorithms (e.g., `__call__`, `reduce`, `_compute_elementwise`) that subclasses implement. |
| **Strategy Pattern**     | Each subclass encapsulates a distinct algorithm (e.g., `MinTNorm`, `GainesImplicator`) behind a common interface. |
| **Decorator Pattern**    | `@register(*names)` dynamically registers each subclass into a central registry. |
| **Adapter Pattern**      | `to_dict()` and `from_dict()` convert between object state and serializable dictionary format. |
| **Reflection/Introspection** | `inspect.signature` + `_filter_args()` ensure dynamic and safe instantiation using only valid constructor parameters. |

## 🧼 Clean Code Principles

| Principle                            | Applied Practices |
|--------------------------------------|-------------------|
| **Single Responsibility Principle**  | Each class/method does one thing well. E.g., `YagerTNorm` computes only Yager logic. |
| **Open/Closed Principle**            | Easily extendable via subclassing and registration—no need to modify existing code. |
| **DRY (Don't Repeat Yourself)**      | Common logic (e.g., validation, filtering, dispatch) is reused and abstracted. |
| **Encapsulation & Abstraction**      | All complexity is hidden behind clearly defined abstract interfaces. |
| **Liskov Substitution Principle**    | Subclasses can be used wherever the base class is expected without altering behavior. |
| **YAGNI & KISS**                     | Only necessary functionality is included; minimalistic but sufficient interfaces. |
| **Fail-Fast Validation**             | Immediate checks on parameter validity, input shapes, and usage errors. |
| **Type Safety & Introspection**      | Explicit parameter typing (`p: float`), dynamic shape handling, and detailed metadata reporting. |
| **Consistent Doxygen-style Docstrings** | All classes and methods follow structured docstring format for IDE & documentation tool compatibility. |
| **Composability & Functional Symmetry** | Support scalar, vector, and matrix computation with identical method names and behaviors. |
| **Readability & Predictability**     | Method names like `apply_pairwise_matrix`, `describe_params_detailed`, and `create` make the code intuitive. |

## 🧱 Architectural Guidelines

| Architecture Element          | Policy |
|------------------------------|--------|
| **Pluggable Component Model** | New behaviors are added by subclassing and decorating, not by modifying existing logic. |
| **Serialization/Deserialization** | All runtime objects that need reconstruction must implement `to_dict()` and `from_dict()`. |
| **Centralized Parameter Validation** | Each class must optionally implement `validate_params(**kwargs)` to ensure runtime safety. |
| **Dynamic Configuration Support** | Subclasses must return runtime parameter details via `describe_params_detailed()` or `_get_params()`. |
| **Uniform Interface Contracts** | Methods like `__call__`, `reduce`, `apply_pairwise_matrix` must behave consistently across all subclasses. |

## 🧪 Testing Compatibility Requirements

- **All components must work with synthetic testing data (`syntetic_data_for_tests.py`).**
- **Each subclass must be unit-testable for:**
  - Scalar inputs
  - 1D vectors
  - 2D matrices
- **Expected behavior must match mathematical definitions.**
- **`create()` and `from_dict()` must return equivalent instances.**
- **Serialization and deserialization must preserve all important state.**