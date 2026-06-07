"""
@file __init__.py
@brief Algorithm namespace for fuzzy-rough oversampling methods.

Concrete algorithms such as FRSMOTE and future FRADASYN live in this package.
Importing this namespace registers available algorithms in the package registry.

##############################################
# ✅ Quick Summary of Features
# Feature                              Description
# ----------------------------------------------------------------------------------
# FRSMOTE                              Positive-region guided fuzzy-rough SMOTE

# ✅ Design Patterns & Clean Code Notes
# - Package Namespace: keeps concrete algorithms separate from registry/factory code
# - Registry Pattern: algorithm imports execute registration decorators
# - Open/Closed Principle: future algorithms can be added as new modules
##############################################
"""

from fuzzy_rough_oversampling.algorithms.frsmote import FRSMOTE

__all__ = ["FRSMOTE"]
