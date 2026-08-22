# -*- coding: utf-8 -*-
"""
3D_cube_validation
===================
Thin rectangular plate (genuinely 3D, closed/watertight triangulated
shell) validation and differentiability example for JAX-EP.

Replaces the earlier bounding-box "patch" (a subset of the real,
patient-specific LA mesh) used in the parameter_learning/forward_model
differentiability demonstration, with a purpose-built, larger-than-
the-HD-grid synthetic geometry -- while reusing exactly the same
numerical scheme (Rush-Larsen ionic integration, Crank-Nicolson
diffusion, cotangent-weight FEM) as the patch and the manuscript's
own formally validated 2D flat-plate convergence study.

Modules
-------
mesh :  thin-plate mesh generation, HD-grid electrode placement,
        entire-edge pacing mask construction.
fem :   cotangent-weight FEM operator assembly, anisotropic
        conductivity weights.
solver : JIT-compiled forward solver (matches the patch's exact
        scheme) and omnipolar EGM (OEGM) construction.
differentiability : FD / jax.grad / jax.jacfwd gradient comparison,
        matching the patch-level demonstration's structure.
"""
from .mesh import build_thin_plate_mesh, place_hd_grid, build_edge_pacing_mask, build_patch_style_pacing_mask
from .fem import build_fem_operators, anisotropic_weights
from .solver import make_forward_solver, make_forward_solver_euler, make_forward_solver_atmap, make_forward_solver_traces, make_forward_solver_traces_euler, compute_oegm
from .differentiability import (
    make_loss_fn, make_atmap_loss_fn, fd_gradient, ad_gradient, jacfwd_gradient,
    report_comparison, PNAMES,
)

__all__ = [
    "build_thin_plate_mesh", "place_hd_grid", "build_edge_pacing_mask",
    "build_patch_style_pacing_mask",
    "build_fem_operators", "anisotropic_weights",
    "make_forward_solver", "make_forward_solver_euler", "make_forward_solver_atmap",
    "make_forward_solver_traces", "make_forward_solver_traces_euler", "compute_oegm",
    "make_loss_fn", "make_atmap_loss_fn", "fd_gradient", "ad_gradient", "jacfwd_gradient",
    "report_comparison", "PNAMES",
]
