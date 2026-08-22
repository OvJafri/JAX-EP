# -*- coding: utf-8 -*-
"""
mesh.py
=======
Mesh generation for the 3D thin-plate cube validation example.

Builds a closed, watertight, purely-triangular thin rectangular
plate (top face + bottom face + 4 side walls) -- genuinely 3D
(non-zero thickness) while using exactly the same triangle-based
connectivity as the rest of the JAX-EP codebase (Labelled.elem, the
2D flat-plate benchmark, and the patch builder in
parameter_learning/forward_model), so the existing FEM assembly
code applies unchanged.

Also places a 4x4, 16-electrode HD grid (matching the real Abbott
Advisor HD Grid catheter: 3mm electrode spacing) centred on the top
face, and builds an entire-edge pacing mask (paced from one full
edge of the top face, not a radius-based centroid mask).
"""
import numpy as np


def build_thin_plate_mesh(lx_mm, ly_mm, thickness_mm, dx_mm):
    """
    Build a closed, watertight, triangulated thin rectangular plate.

    Structure: a regular (nx, ny) grid on the top face (z=thickness)
    and an identical grid on the bottom face (z=0), each triangulated
    by splitting quads (same convention as the 2D flat-plate
    benchmark), connected by 4 triangulated side-wall strips around
    the perimeter.

    Parameters
    ----------
    lx_mm, ly_mm : float
        In-plane dimensions of the plate (mm).
    thickness_mm : float
        Plate thickness (mm), i.e. the z-extent.
    dx_mm : float
        In-plane grid spacing (mm).

    Returns
    -------
    Verts : (N, 3) float64 array
        Node coordinates in mm.
    Elems : (M, 3) int64 array
        Triangle connectivity (node indices into Verts).
    top_grid : (ny, nx) int64 array
        Node-index grid for the top face, for electrode/pacing placement.
    bot_grid : (ny, nx) int64 array
        Node-index grid for the bottom face.
    """
    nx = int(round(lx_mm / dx_mm)) + 1
    ny = int(round(ly_mm / dx_mm)) + 1
    x = np.linspace(0, lx_mm, nx)
    y = np.linspace(0, ly_mm, ny)
    xx, yy = np.meshgrid(x, y)  # shape (ny, nx)

    def grid_face(z_val, node_offset, flip):
        verts = np.column_stack([xx.ravel(), yy.ravel(),
                                  np.full(xx.size, z_val)])
        node_grid = np.arange(xx.size).reshape(ny, nx) + node_offset
        tris = []
        for iy in range(ny - 1):
            for ix in range(nx - 1):
                bl = node_grid[iy, ix]
                br = node_grid[iy, ix + 1]
                tl = node_grid[iy + 1, ix]
                tr = node_grid[iy + 1, ix + 1]
                if not flip:
                    tris.append([bl, br, tr]); tris.append([bl, tr, tl])
                else:
                    tris.append([bl, tr, br]); tris.append([bl, tl, tr])
        return verts, np.array(tris, dtype=np.int64), node_grid

    n_per_face = nx * ny
    top_verts, top_tris, top_grid = grid_face(thickness_mm, 0, flip=False)
    bot_verts, bot_tris, bot_grid = grid_face(0.0, n_per_face, flip=True)

    def wall_strip(top_row, bot_row):
        tris = []
        for i in range(len(top_row) - 1):
            t0, t1 = top_row[i], top_row[i + 1]
            b0, b1 = bot_row[i], bot_row[i + 1]
            tris.append([t0, t1, b1]); tris.append([t0, b1, b0])
        return np.array(tris, dtype=np.int64)

    walls = [
        wall_strip(top_grid[0, :], bot_grid[0, :]),
        wall_strip(top_grid[ny - 1, :], bot_grid[ny - 1, :]),
        wall_strip(top_grid[:, 0], bot_grid[:, 0]),
        wall_strip(top_grid[:, nx - 1], bot_grid[:, nx - 1]),
    ]

    Verts = np.vstack([top_verts, bot_verts])
    Elems = np.vstack([top_tris, bot_tris] + walls)
    return Verts, Elems, top_grid, bot_grid


def place_hd_grid(Verts, top_grid, spacing_mm=3.0, n_side=4,
                    xy_jitter_frac=0.5):
    """
    Place a 4x4 (n_side x n_side), 16-electrode HD grid, ON the top
    surface (z = thickness exactly, no vertical offset), with the
    given electrode-to-electrode spacing -- matching the real Abbott
    Advisor HD Grid catheter (3mm spacing, 16 electrodes) used
    clinically in the HEAT-AF trial data this codebase is built
    around.

    Electrode positions are CONTINUOUS physical coordinates (not
    snapped to mesh nodes), matching the existing lead-field
    convention used throughout the codebase (1/r weighting from
    continuous electrode position to each node).

    IMPORTANT, CORRECTED: electrodes sit genuinely ON the surface
    (z = thickness), matching how the real, original clinical data
    (a genuine contact-mapping catheter, physically against the
    endocardial surface) was used in the source Forward_LA.py code --
    NOT offset above it by an artificial vertical standoff (an
    earlier version of this function added standoff_mm to z; this
    was corrected after review confirmed the original code placed
    electrodes on the surface, not elevated above it).

    To avoid the SEPARATE, purely numerical issue this raises (a
    centred, regularly-spaced electrode grid on this structured mesh
    lands EXACTLY on mesh grid lines in x/y -- confirmed: for a
    19mm plate, dx=0.2mm, 3mm spacing, every electrode x/y coordinate
    is an exact multiple of dx, causing exact node coincidence, r=0,
    and a ~50,000x lead-field blowup), the electrode grid's centroid
    is shifted by HALF a mesh cell in x and y (xy_jitter_frac=0.5,
    i.e. dx/2) -- small enough to be well within a single mesh cell
    (physiologically negligible), but enough to guarantee no
    electrode exactly coincides with a node, while keeping every
    electrode genuinely ON the tissue surface, not floating above it.

    Parameters
    ----------
    Verts : (N, 3) array
        Full mesh node coordinates (mm).
    top_grid : (ny, nx) array
        Node-index grid for the top face (from build_thin_plate_mesh).
    spacing_mm : float
        Electrode-to-electrode spacing (default 3.0mm).
    n_side : int
        Electrodes per side (default 4, giving 16 total).
    xy_jitter_frac : float
        Fraction of the local mesh spacing (dx) to shift the grid's
        centroid by in x and y, to avoid exact node coincidence
        (default 0.5, i.e. half a mesh cell).

    Returns
    -------
    HD16 : (16, 3) array
        Electrode coordinates (mm), ordered row-major (matches the
        CLIQUES indexing convention used elsewhere in the codebase).
    """
    top_node_ids = top_grid.ravel()
    top_coords = Verts[top_node_ids]
    centre = top_coords.mean(axis=0)

    # Determine the local mesh spacing directly from the top_grid's
    # own node coordinates, rather than assuming a value -- robust to
    # whatever dx the mesh was actually built with.
    ny, nx = top_grid.shape
    dx_local = abs(Verts[top_grid[0, 1], 0] - Verts[top_grid[0, 0], 0]) \
        if nx > 1 else spacing_mm
    jitter = xy_jitter_frac * dx_local

    half_extent = (n_side - 1) * spacing_mm / 2.0
    offsets = np.linspace(-half_extent, half_extent, n_side)

    HD16 = []
    for oy in offsets:
        for ox in offsets:
            HD16.append([centre[0] + ox + jitter, centre[1] + oy + jitter,
                         centre[2]])
    return np.array(HD16, dtype=np.float64)


def build_edge_pacing_mask(Verts, top_grid, bot_grid, edge="LEFT"):
    """
    Build a pacing mask covering an ENTIRE edge of the plate (both
    top and bottom face nodes along that edge, plus the connecting
    side-wall nodes are automatically included since they reuse the
    same perimeter node indices) -- simpler and more physically
    direct than a radius-based centroid mask.

    Parameters
    ----------
    edge : str
        One of "LEFT", "RIGHT", "BOTTOM", "TOP" -- matches the
        EDGE_ELEC convention already used in the patch-builder code
        elsewhere in this repository.

    Returns
    -------
    mask : (N,) float64 array
        1.0 at paced nodes, 0.0 elsewhere.
    paced_node_ids : (K,) int64 array
        The node indices that are paced (for writing a .vtx-style
        file if desired).
    """
    ny, nx = top_grid.shape
    if edge == "LEFT":
        top_ids, bot_ids = top_grid[:, 0], bot_grid[:, 0]
    elif edge == "RIGHT":
        top_ids, bot_ids = top_grid[:, nx - 1], bot_grid[:, nx - 1]
    elif edge == "BOTTOM":
        top_ids, bot_ids = top_grid[0, :], bot_grid[0, :]
    elif edge == "TOP":
        top_ids, bot_ids = top_grid[ny - 1, :], bot_grid[ny - 1, :]
    else:
        raise ValueError(f"Unknown edge '{edge}', expected one of "
                          f"LEFT/RIGHT/BOTTOM/TOP")

    paced_node_ids = np.concatenate([top_ids, bot_ids]).astype(np.int64)
    mask = np.zeros(len(Verts), dtype=np.float64)
    mask[paced_node_ids] = 1.0
    return mask, paced_node_ids


def build_patch_style_pacing_mask(Verts, HD16, cliques, edge="LEFT",
                                    edge_elec=None, offset_um=3000.0,
                                    radius_um=2000.0):
    """
    Build a NARROW, LOCALIZED pacing mask, faithfully reproducing the
    ORIGINAL patch's own pacing geometry -- NOT the entire-edge mask
    (build_edge_pacing_mask), which produces a wide, planar wavefront
    quite different from a real, localized clinical pacing site.

    This directly matters for OEGM morphology: the original patch
    paces from a narrow, near-point-source site only ~3mm beyond the
    HD grid's own edge electrodes (radius_um=2mm perpendicular band),
    producing a curved, radially-expanding wavefront through the
    grid -- genuinely different spatial gradient patterns across the
    16 electrodes than a flat, planar wavefront from an entire-edge
    pacing site would produce. Confirmed as a likely real contributor
    to less-physiologic-looking OEGM shapes when using the wider,
    entire-edge mask instead.

    Faithfully reproduces the original code's exact geometry:
      1. Take the 4 electrodes along one edge of the HD grid.
      2. Their centroid (_epc).
      3. The unit vector from that centroid TOWARD the centre
         clique's own centroid (_wf) -- i.e. the direction "into"
         the grid.
      4. The pacing line's centre (_lc) is offset_um AWAY from the
         edge electrodes, in the direction AWAY from the grid
         (opposite to _wf).
      5. The pacing line's orientation (_ld) is the dominant
         direction of the edge electrodes themselves (via SVD),
         i.e. parallel to that edge.
      6. The pacing mask includes all mesh nodes within radius_um
         perpendicular distance of this line, and within the line's
         own extent (the edge electrodes' spatial extent + a 2mm
         margin) along its length.

    Parameters
    ----------
    Verts : (N, 3) array
        Full mesh node coordinates (mm).
    HD16 : (16, 3) array
        HD-grid electrode coordinates (mm), from place_hd_grid.
    cliques : (9, 4) int array
        The CLIQUES convention array (0-indexed).
    edge : str
        Which edge of the HD GRID (not the plate) to pace from --
        "LEFT", "RIGHT", "BOTTOM", "TOP". Matches the original code's
        own EDGE_ELEC convention exactly.
    edge_elec : dict, optional
        Override the default EDGE_ELEC electrode-index mapping if
        needed. Defaults to the original code's own convention:
        LEFT=[0,4,8,12], RIGHT=[3,7,11,15], BOTTOM=[0,1,2,3],
        TOP=[12,13,14,15] (0-indexed, row-major, matching HD16's own
        ordering from place_hd_grid).
    offset_um : float
        Distance (um) the pacing line sits beyond the edge
        electrodes, away from the grid (default 3000um = 3mm,
        matching the original code exactly).
    radius_um : float
        Perpendicular distance (um) defining the pacing band's width
        (default 2000um = 2mm, matching the original code exactly).

    Returns
    -------
    mask : (N,) float64 array
        1.0 at paced nodes, 0.0 elsewhere.
    paced_node_ids : (K,) int64 array
        The node indices that are paced.
    """
    if edge_elec is None:
        edge_elec = {"LEFT": np.array([0, 4, 8, 12]),
                     "RIGHT": np.array([3, 7, 11, 15]),
                     "BOTTOM": np.array([0, 1, 2, 3]),
                     "TOP": np.array([12, 13, 14, 15])}

    ep = HD16[edge_elec[edge]]
    epc = ep.mean(0)
    c5c = HD16[cliques[4]].mean(0)
    wf = (c5c - epc) / (np.linalg.norm(c5c - epc) + 1e-12)
    lc = epc - wf * offset_um * 1e-3  # um -> mm (Verts/HD16 are in mm)

    _, _, Ve = np.linalg.svd(ep - epc, full_matrices=False)
    ld = Ve[0]
    projs = (ep - epc) @ ld
    # The 2000um (2mm) margin on hl is a HARDCODED constant in the
    # real, confirmed source code, independent of radius_um --
    # NOT tied to the radius parameter (verified directly against
    # parameter_estimation_kaggle.py: "hl=max(np.ptp(projs)/2.,1.)+2000.").
    margin_um = 2000.0
    hl = max(np.ptp(projs) / 2.0, 1.0) + margin_um * 1e-3

    d = Verts - lc
    pal = d @ ld
    perp = d - np.outer(pal, ld)
    dist = np.linalg.norm(perp, axis=1)
    mk = (dist <= radius_um * 1e-3) & (np.abs(pal) <= hl)
    if mk.sum() < 10:
        # Exact fallback from the real code: doubles the radius AND
        # extends the length constraint by the same margin -- the
        # length (pal) constraint is NOT dropped, only widened.
        mk = (dist <= radius_um * 1e-3 * 2.0) & \
             (np.abs(pal) <= hl + margin_um * 1e-3)

    paced_node_ids = np.where(mk)[0].astype(np.int64)
    mask = np.zeros(len(Verts), dtype=np.float64)
    mask[paced_node_ids] = 1.0
    return mask, paced_node_ids
