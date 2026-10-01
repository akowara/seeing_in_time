"""
Scene definitions for the Isaac Sim policy test (pure numpy, no Isaac imports).

A scene is a dict:
    balls   (N, 4)  x, y, z, r                      spheres
    voxels  (N, 6)  cx, cy, cz, hx, hy, hz          axis-aligned boxes (half sizes)
    cyl     (N, 3)  x, y, r                         vertical cylinders (infinite height)
    cyl_h   (N, 3)  x, z, r                         horizontal cylinders along world y
    p0      (3,)    start position (m, z up)
    goal    (3,)    goal position
    max_speed  float  (m/s)
    margin     float  (m), the drone radius term in the policy state
    trap_x     float or None   (U-trap only) x of the back wall's far face

Same world convention as the training sim and Isaac/Pegasus: x forward, y left, z up.
"""

import numpy as np

KEYS = ("balls", "voxels", "cyl", "cyl_h")


def empty_scene():
    return {"balls": np.zeros((0, 4)), "voxels": np.zeros((0, 6)),
            "cyl": np.zeros((0, 3)), "cyl_h": np.zeros((0, 3)),
            "p0": np.array([0.0, 0.0, 1.5]), "goal": np.array([40.0, 0.0, 1.5]),
            "max_speed": 6.0, "margin": 0.2, "trap_x": None}


# -----------------------------------------------------------------------------
# Scenes exported from the training sim (export_scenes.py)
# -----------------------------------------------------------------------------

def load_exported(path):
    """Load scenes saved by export_scenes.py -> list of scene dicts."""
    data = np.load(path, allow_pickle=True)
    B = int(data["p0"].shape[0])
    scenes = []
    for i in range(B):
        s = empty_scene()
        for k in KEYS:
            if k in data.files:
                arr = np.asarray(data[k][i], dtype=np.float64)
                s[k] = arr.reshape(-1, arr.shape[-1]) if arr.size else s[k]
        s["p0"] = np.asarray(data["p0"][i], dtype=np.float64)
        s["goal"] = np.asarray(data["goal"][i], dtype=np.float64)
        s["max_speed"] = float(np.ravel(data["max_speed"][i])[0])
        s["margin"] = float(np.ravel(data["margin"][i])[0])
        scenes.append(s)
    return scenes


# -----------------------------------------------------------------------------
# Built-in scenes
# -----------------------------------------------------------------------------

def make_utrap(rng, alt=1.5, max_speed=6.0, margin=0.2, thickness=0.3, wall_top=7.0):
    """U-trap matching the training-sim --utrap spec: cup 5-8 m wide, 3-6 m deep,
    back wall 12-14 m ahead, goal 8 m behind it, walls from floor to 7 m."""
    s = empty_scene()
    D = rng.uniform(12, 14)          # back wall (near face) distance
    w = rng.uniform(5, 8)            # cup width
    d = rng.uniform(3, 6)            # cup depth
    hz = wall_top / 2
    t = thickness / 2
    back = [D + t, 0.0, hz, t, w / 2 + thickness, hz]
    left = [D - d / 2, w / 2 + t, hz, d / 2, t, hz]
    right = [D - d / 2, -w / 2 - t, hz, d / 2, t, hz]
    s["voxels"] = np.array([back, left, right])
    s["p0"] = np.array([0.0, 0.0, alt])
    s["goal"] = np.array([D + 8.0, 0.0, alt])
    s["max_speed"] = max_speed
    s["margin"] = margin
    s["trap_x"] = D + thickness
    return s


def make_forest(rng, alt=1.5, length=40.0, half_width=8.0, density=0.08,
                r_range=(0.15, 0.45), max_speed=6.0, margin=0.2):
    """Generic tree field (vertical cylinders). Use exported scenes for an exact
    match to training; this is only a fallback."""
    s = empty_scene()
    area = (length - 8.0) * 2 * half_width
    n = rng.poisson(density * area)
    x = rng.uniform(4.0, length - 4.0, n)
    y = rng.uniform(-half_width, half_width, n)
    r = rng.uniform(*r_range, n)
    s["cyl"] = np.stack([x, y, r], 1) if n else np.zeros((0, 3))
    s["p0"] = np.array([0.0, 0.0, alt])
    s["goal"] = np.array([length, 0.0, alt])
    s["max_speed"] = max_speed
    s["margin"] = margin
    return s


# -----------------------------------------------------------------------------
# Distance to the nearest obstacle (same quantity the training metric uses)
# -----------------------------------------------------------------------------

def nearest_distance(scene, p, ground=True):
    """Distance (m) from point p to the nearest obstacle surface (negative inside)."""
    d = [np.inf]
    b = scene["balls"]
    if len(b):
        d.append(np.min(np.linalg.norm(b[:, :3] - p, axis=1) - b[:, 3]))
    c = scene["cyl"]
    if len(c):
        d.append(np.min(np.linalg.norm(c[:, :2] - p[:2], axis=1) - c[:, 2]))
    ch = scene["cyl_h"]
    if len(ch):
        d.append(np.min(np.hypot(ch[:, 0] - p[0], ch[:, 1] - p[2]) - ch[:, 2]))
    v = scene["voxels"]
    if len(v):
        q = np.abs(p - v[:, :3]) - v[:, 3:]
        outside = np.linalg.norm(np.maximum(q, 0), axis=1)
        inside = np.minimum(np.max(q, axis=1), 0)
        d.append(np.min(outside + inside))
    if ground:
        d.append(p[2])
    return float(min(d))