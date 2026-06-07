"""
@file __init__.py
@brief Preprocessing namespace for FRsutils core.

FRsutils no longer hosts fuzzy-rough oversampling algorithms in its core
package. Application-layer oversamplers such as FRSMOTE live in the standalone
`fuzzy_rough_oversampling` package and consume FRsutils through `FRsutils.api`.

##############################################
# ✅ Quick Summary of Features
# - Keeps the historical preprocessing namespace importable.
# - Documents that fuzzy-rough oversampling moved out of FRsutils core.
# - Avoids importing imbalanced-learn or downstream oversampling code.

# ✅ Design Patterns & Clean Code Notes
# - Package Boundary: FRsutils remains the fuzzy-rough core engine.
# - Dependency Inversion: downstream oversamplers depend on FRsutils, not the reverse.
##############################################
"""

__all__: list[str] = []
