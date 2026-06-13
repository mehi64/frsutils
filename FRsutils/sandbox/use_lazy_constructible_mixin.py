# SPDX-License-Identifier: BSD-3-Clause
"""Exploratory example script for lazy construction utilities.

This module is an exploratory usage script and is not part of the stable public API.
"""

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
