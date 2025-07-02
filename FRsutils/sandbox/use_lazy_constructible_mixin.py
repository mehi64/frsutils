from FRsutils.utils.constructor_utils.lazy_constructible_mixin import LazyConstructibleMixin

obj = LazyConstructibleMixin()

config = {
    "name": "Mehran",
    "age": 28,
    "address": {
        "street": "123 Main St",
        "city": "New York",
        "state": "NY",
        "zip": "10001"
    }
}

obj.configure(**config)
print(obj)