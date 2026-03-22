"""Kagome lattice builder functions extracted verbatim from the source notebook."""

import numpy as np
import pybinding as pb
from math import sqrt
from shapely.geometry import Polygon

import torch


def numpy_to_tensor(np_array, requires_grad=False, tkwargs=None):
    """Convert a NumPy array to a PyTorch tensor."""
    from app.core.config import TKWARGS
    kw = tkwargs if tkwargs is not None else TKWARGS
    return torch.tensor(np_array, requires_grad=requires_grad, **kw)


def triangle_rot(a, b, c, shx, shy):
    """Create a PyBinding Polygon defining the triangular finite-lattice boundary."""
    return pb.Polygon([[shx, -a / 2 + shy], [shx, b / 2 + shy], [c + shx, shy]])


def Kagome_lattice(d, t_a=-1, t_b=-1, t_nnn=0, v_a=0, v_b=0, v_c=0):
    """Define a distorted Kagome tight-binding lattice with two distinct NN hoppings."""
    a_cc = d
    a = a_cc * 2
    lat = pb.Lattice(
        a1=[a * sqrt(3) / 2, -a / 2],
        a2=[a * sqrt(3) / 2, a / 2],
    )
    lat.add_sublattices(
        ("A", [a * sqrt(3) / 4, -a / 4], v_a),
        ("B", [a * sqrt(3) / 4, a / 4], v_b),
        ("C", [0, 0], v_c),
    )
    lat.add_hoppings(
        ([0, 0], "A", "C", t_a),
        ([1, 0], "A", "C", t_b),
        ([0, 0], "B", "A", t_a),
        ([-1, 1], "B", "A", t_b),
        ([0, 0], "C", "B", t_a),
        ([0, -1], "C", "B", t_b),
        ([1, 0], "A", "B", t_nnn),
        ([1, 0], "B", "C", t_nnn),
        ([0, 1], "B", "A", t_nnn),
        ([0, 1], "A", "C", t_nnn),
        ([-1, 1], "C", "A", t_nnn),
        ([-1, 1], "B", "C", t_nnn),
    )
    return lat


def lattice_model(
    d=0.133,
    t_a=-1.0,
    t_b=-1.0,
    t_nnn=0,
    v_a=0,
    v_b=0,
    v_c=0,
):
    """Build a finite-size Kagome lattice model with a triangular shape boundary.

    Returns:
        (lattice, model) — PyBinding Lattice and Model objects.
    """
    shape = triangle_rot(1.3, 1.3, 1.15, 0.10, d)
    lattice = Kagome_lattice(d, t_a, t_b, t_nnn, v_a, v_b, v_c)
    model = pb.Model(Kagome_lattice(d, t_a, t_b, t_nnn, v_a, v_b, v_c), shape)
    return lattice, model
