# hostsub_gp/spec_base.py

from typing import Protocol, TypeVar
from jax._src.typing import Array

class SpecModelProtocol(Protocol):
    """Protocol for the SpecModel class"""
    pixel_scale: float
    center_ra: float
    center_dec: float
    position_angle: float
    spat_resln: float
    spec_resln: float
    spat: Array
    spec: Array
    shape: tuple[int, int]
    spat_filter: dict[str, Array]
    spat_edges: dict[str, tuple[float, float]]
    slit_wid: float
    mask_offset: float

SpecModelP = TypeVar('SpecModelP', bound=SpecModelProtocol)