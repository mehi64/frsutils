
# This lets users write:
# from frsutils.core.models import ITFRS

from .itfrs import ITFRS
from .owafrs import OWAFRS
from .vqrs import VQRS


__all__ = ['ITFRS', 'OWAFRS', 'VQRS']
