"""
Dump obstacle scenes from the training sim so Isaac Sim can rebuild the exact same
geometry. Run in your normal training environment (NOT inside Isaac Sim), from the
repo root:

    python seeing_in_time/export_scenes.py $(cat configs/single_agent.args) \
        --num_scenes 64 --out isaac_scenes_seed0.npz

It prints every tensor the env holds after reset(). If the obstacle names below don't
match your env_cuda.py, the printout shows the right ones; edit OBSTACLE_ATTRS.
"""

import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from env_cuda import Env  # noqa: E402

# env attribute -> key in the exported file (see isaac_scenes.py for formats)
OBSTACLE_ATTRS = {"balls": "balls", "voxels": "voxels", "cyl": "cyl", "cyl_h": "cyl_h"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num_scenes", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="isaac_scenes.npz")
    # Same env flags as eval_planner.py, so $(cat configs/single_agent.args) works
    p.add_argument("--grad_decay", type=float, default=0.4)
    p.add_argument("--speed_mtp", type=float, default=1.0)
    p.add_argument("--fov_x_half_tan", type=float, default=0.53)
    p.add_argument("--cam_angle", type=int, default=10)
    for f in ["single", "gate", "ground_voxels", "scaffold", "random_rotation",
              "gimbal_stabilized"]:
        p.add_argument(f"--{f}", default=False, action="store_true")
    args, unknown = p.parse_known_args()
    if unknown:
        print("ignoring flags:", " ".join(unknown))
    return args


@torch.no_grad()
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda")
    B = args.num_scenes
    env = Env(B, 64, 48, args.grad_decay, device,
              fov_x_half_tan=args.fov_x_half_tan, single=args.single,
              gate=args.gate, ground_voxels=args.ground_voxels,
              scaffold=args.scaffold, speed_mtp=args.speed_mtp,
              random_rotation=args.random_rotation, cam_angle=args.cam_angle,
              gimbal_stabilized=args.gimbal_stabilized)
    env.reset()

    print("\nTensors on the env after reset():")
    for k, v in sorted(vars(env).items()):
        if torch.is_tensor(v):
            print(f"  {k:24s} {tuple(v.shape)}")

    out = {}
    for attr, key in OBSTACLE_ATTRS.items():
        v = getattr(env, attr, None)
        if torch.is_tensor(v) and v.shape[0] == B:
            out[key] = v.float().cpu().numpy()
            print(f"exported {attr} -> {key} {tuple(v.shape)}")
        else:
            print(f"WARNING: env has no per-scene tensor '{attr}'; skipped")

    def per_drone(name):
        v = getattr(env, name)
        v = v.float().cpu().numpy()
        return v.reshape(B, -1)

    out["p0"] = per_drone("p")
    out["goal"] = per_drone("p_target")
    out["max_speed"] = per_drone("max_speed")[:, :1]
    out["margin"] = per_drone("margin")[:, :1]
    np.savez(args.out, **out)
    print(f"\nsaved {B} scenes to {args.out}")
    print(f"start z range {out['p0'][:, 2].min():.2f}..{out['p0'][:, 2].max():.2f} m, "
          f"goal distance {np.linalg.norm(out['goal'] - out['p0'], axis=1).mean():.1f} m, "
          f"max_speed {out['max_speed'].min():.1f}..{out['max_speed'].max():.1f} m/s")


if __name__ == "__main__":
    main()