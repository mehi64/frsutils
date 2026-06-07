"""
@file registry.py
@brief Registry and factory helpers for fuzzy-rough oversampling algorithms.

This module provides the public registration and construction mechanism used by
FRSMOTE, FRADASYN, and future fuzzy-rough oversamplers. It is intentionally
small so algorithm modules can register themselves without coupling callers to
concrete module paths.

##############################################
# ✅ Quick Summary of Features
# Feature                              Description
# ----------------------------------------------------------------------------------
# register_oversampler                 Register an oversampler class by aliases
# get_oversampler_class                Resolve an oversampler class by alias
# list_oversamplers                    Return public aliases grouped by algorithm
# build_oversampler                    Construct a registered oversampler

# ✅ Design Patterns & Clean Code Notes
# - Registry Pattern: maps public names to algorithm classes
# - Factory Method: build_oversampler constructs registered algorithms
# - SRP: only algorithm lookup/registration lives here
# - Open/Closed Principle: new algorithms register without changing callers
##############################################

##############################################
# ✅ How to Use - Examples
##############################################

# @register_oversampler("frsmote", aliases=("fuzzy_rough_smote",))
# class FRSMOTE(...):
#     ...
#
# sampler = build_oversampler("frsmote", random_state=42)
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, TypeVar

T = TypeVar("T")

_ALGORITHM_REGISTRY: dict[str, type[Any]] = {}
_ALGORITHM_ALIAS_GROUPS: dict[str, set[str]] = defaultdict(set)


def _normalize_alias(alias: str) -> str:
    """
    @brief Normalize a public oversampler alias.

    @param alias: User-facing alias.
    @return: Lowercase stripped alias.
    @raises TypeError: If alias is not a string.
    @raises ValueError: If alias is empty.
    """
    if not isinstance(alias, str):
        raise TypeError("Oversampler alias must be a string.")
    normalized = alias.strip().lower()
    if not normalized:
        raise ValueError("Oversampler alias cannot be empty.")
    return normalized


def register_oversampler(
    name: str,
    *,
    aliases: Iterable[str] = (),
) -> Callable[[type[T]], type[T]]:
    """
    @brief Decorator used by algorithms to register public oversampler aliases.

    @param name: Primary public algorithm name, e.g. "frsmote".
    @param aliases: Optional additional aliases.
    @return: Class decorator that registers and returns the class unchanged.
    @raises ValueError: If any alias is already registered for another class.
    """
    primary_alias = _normalize_alias(name)
    all_aliases = {primary_alias, *(_normalize_alias(alias) for alias in aliases)}

    def _decorator(cls: type[T]) -> type[T]:
        for alias in all_aliases:
            existing = _ALGORITHM_REGISTRY.get(alias)
            if existing is not None and existing is not cls:
                raise ValueError(
                    f"Oversampler alias '{alias}' is already registered for "
                    f"{existing.__name__}."
                )
            _ALGORITHM_REGISTRY[alias] = cls
            _ALGORITHM_ALIAS_GROUPS[primary_alias].add(alias)
        return cls

    return _decorator


def get_oversampler_class(name: str) -> type[Any]:
    """
    @brief Resolve a registered oversampler class by public alias.

    @param name: Registered public name or alias.
    @return: Oversampler class.
    @raises ValueError: If no oversampler is registered for the alias.
    """
    alias = _normalize_alias(name)
    try:
        return _ALGORITHM_REGISTRY[alias]
    except KeyError as exc:
        available = ", ".join(sorted(_ALGORITHM_REGISTRY)) or "<none>"
        raise ValueError(
            f"Unknown fuzzy-rough oversampler '{name}'. Available aliases: {available}."
        ) from exc


def list_oversamplers() -> Dict[str, list[str]]:
    """
    @brief Return registered oversampler aliases grouped by primary algorithm name.

    @return: Mapping from primary algorithm name to sorted public aliases.
    """
    return {
        primary: sorted(aliases)
        for primary, aliases in sorted(_ALGORITHM_ALIAS_GROUPS.items())
    }


def build_oversampler(name: str, **params: Any) -> Any:
    """
    @brief Build a registered fuzzy-rough oversampler by name.

    @param name: Registered algorithm name, e.g. "frsmote".
    @param params: Constructor parameters forwarded to the algorithm class.
    @return: Constructed oversampler instance.
    """
    oversampler_cls = get_oversampler_class(name)
    return oversampler_cls(**params)
