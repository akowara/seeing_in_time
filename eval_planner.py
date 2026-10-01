"""
Evaluate a trained policy in the training sim, with or without the TTC planner.

Same env, model and observation pipeline as training; no losses, no optimizer,
no gradients. Run twice with the same seed to compare on identical scenes:

    python eval_planner.py --resume checkpoints/ttc/checkpoint0004.pth --use_ttc
    python eval_planner.py --resume checkpoints/ttc/checkpoint0004.pth --use_ttc --use_planner

Pass the same observation flags (--use_ttc, --no_odom, --use_harmonic, ...)
that the checkpoint was trained with.

Needs ttc_planner.py and planner_bridge.py next to this file.
"""

import argparse
import math
import random
from random import normalvariate

import numpy as np
import torch
import torch.nn.functional as F

import quadsim_cuda
from env_cuda import Env
from model import Model
from planner_bridge import TTCPlannerBridge
from layer_b import LayerB


@torch.no_grad()
def memory_rays(depth, R_cam_world, fov_x_half_tan, pool=4, max_depth=90.0):
    """
    Camera rays for Layer B's memory: min-pooled depth -> world-frame ray directions
    (B, P, 3) and distance along each ray (B, P), inf where nothing was hit.
    """
    d = -F.max_pool2d(-depth[:, None], pool)[:, 0]                 # closest obstacle per cell
    B, h, w = d.shape
    fov_y = fov_x_half_tan * h / w
    fv = (2 * (torch.arange(w, device=d.device, dtype=d.dtype) + 0.5) / w - 1) * fov_x_half_tan
    fu = (2 * (torch.arange(h, device=d.device, dtype=d.dtype) + 0.5) / h - 1) * fov_y
    fu, fv = torch.meshgrid(fu, fv, indexing='ij')
    rays_flu = torch.stack([torch.ones_like(fu), -fv, -fu], -1)      # camera FLU, (h, w, 3)
    rays_world = torch.einsum('bij,hwj->bhwi', R_cam_world, rays_flu)
    dist = d * rays_flu.norm(dim=-1)[None]                         # depth is along the camera axis
    dist = torch.where(d < max_depth, dist, torch.full_like(dist, float('inf')))
    return rays_world.reshape(B, -1, 3), dist.reshape(B, -1)


# -----------------------------------------------------------------------------
# U-trap scene
# -----------------------------------------------------------------------------

@torch.no_grad()
def setup_utrap(env, alt=1.0):
    """
    Replace the random scene with a U-shaped cup facing the drone, goal behind it.

        start ----->   |‾‾‾‾‾‾‾‾‾|        goal
                       |  cup    |  wall
                       |_________|

    Drone starts at x=0 flying +x; the cup's opening faces it, its back wall
    blocks the straight line to the goal. Sizes are randomized per drone.
    Walls are tall (floor to 7 m) so going over isn't an easy way out.
    """
    B, dev = env.batch_size, env.device
    r = lambda: torch.rand(B, device=dev)
    W = 5.0 + 3.0 * r()            # cup inner width (m)
    L = 3.0 + 3.0 * r()            # cup depth (m)
    Xw = 12.0 + 2.0 * r()          # back wall x (m)
    th, zc, zh = 0.1, 3.0, 4.0     # wall half-thickness, height centre, half-height
    z = torch.zeros(B, device=dev)
    o = torch.ones(B, device=dev)

    back = torch.stack([Xw, z, zc * o, th * o, W / 2 + th, zh * o], -1)
    left = torch.stack([Xw - L / 2, W / 2, zc * o, L / 2, th * o, zh * o], -1)
    right = torch.stack([Xw - L / 2, -W / 2, zc * o, L / 2, th * o, zh * o], -1)
    floor = torch.stack([Xw, z, -2.0 * o, 100 * o, 100 * o, 1.0 * o], -1)   # top at z = -1, as in training
    env.voxels = torch.stack([back, left, right, floor], 1).contiguous()
    env.utrap_geom = (Xw, W, L)        # kept for the end-of-episode breakdown

    # Move every random obstacle far away
    env.balls[..., :3] = 1e3
    env.cyl[..., :2] = 1e3
    env.cyl_h[..., :2] = 1e3

    # Start on the cup's axis, goal straight behind the back wall
    env.p = torch.stack([z, 0.1 * torch.randn(B, device=dev), alt * o], -1)
    env.p_old = env.p
    env.p_target = torch.stack([Xw + 8.0, 0.1 * torch.randn(B, device=dev), alt * o], -1)
    env.v = torch.randn((B, 3), device=dev) * 0.2

    # Re-aim the initial attitude at the new goal (same as env.reset)
    R = torch.zeros((B, 3, 3), device=dev)
    env.R = quadsim_cuda.update_state_vec(
        R, env.act, torch.randn((B, 3), device=dev) * 0.2 + F.normalize(env.p_target - env.p),
        torch.zeros_like(env.yaw_ctl_delay), 5)
    env.R_old = env.R.clone()
    env.R_gimbal = env._level(env.R)
    env.R_gimbal_old = env.R_gimbal.clone()


@torch.no_grad()
def setup_open(env, alt=1.0):
    """Empty scene (floor only) with the goal far straight ahead, for step-response tests."""
    B, dev = env.batch_size, env.device
    z = torch.zeros(B, device=dev)
    o = torch.ones(B, device=dev)
    floor = torch.stack([z, z, -2.0 * o, 500 * o, 500 * o, 1.0 * o], -1)   # top at z = -1
    env.voxels = floor[:, None].contiguous()
    env.balls[..., :3] = 1e3
    env.cyl[..., :2] = 1e3
    env.cyl_h[..., :2] = 1e3

    env.p = torch.stack([z, z, alt * o], -1)
    env.p_old = env.p
    env.p_target = torch.stack([300.0 * o, z, alt * o], -1)
    env.v = torch.randn((B, 3), device=dev) * 0.2

    R = torch.zeros((B, 3, 3), device=dev)
    env.R = quadsim_cuda.update_state_vec(
        R, env.act, torch.randn((B, 3), device=dev) * 0.2 + F.normalize(env.p_target - env.p),
        torch.zeros_like(env.yaw_ctl_delay), 5)
    env.R_old = env.R.clone()
    env.R_gimbal = env._level(env.R)
    env.R_gimbal_old = env.R_gimbal.clone()


def step_response_times(t, signal, t0, thresholds=(0.632, 0.9)):
    """
    t: (T,) times; signal: (T,) normalized response, 0 before the step and 1 when fully
    followed. Returns the time after t0 at which the signal first reaches each threshold
    (nan if never).
    """
    out = []
    after = t >= t0
    for th in thresholds:
        hit = after & (signal >= th)
        idx = torch.nonzero(hit).flatten()
        out.append((t[idx[0]] - t0).item() if len(idx) else float('nan'))
    return out


# -----------------------------------------------------------------------------
# Observation pipeline (copied unchanged from the training script)
# -----------------------------------------------------------------------------

def compute_scene_flow(depth, flow, R, R_old, pos, pos_old, ctl_dt, fov_x_half_tan,
                       prev_depth=None, ttc=None, valid_max_depth=90.0):
    assert prev_depth is not None or ttc is not None, \
        "need either prev_depth (preferred) or ttc (fallback) to recover old-frame depth"

    B, H, W = depth.shape
    device = depth.device
    fov_y_half_tan = fov_x_half_tan / W * H

    u = torch.arange(H, device=device, dtype=depth.dtype)
    v = torch.arange(W, device=device, dtype=depth.dtype)
    uu, vv = torch.meshgrid(u, v, indexing='ij')
    uu = uu.expand(B, H, W)
    vv = vv.expand(B, H, W)

    fu = (2 * (uu + 0.5) / H - 1) * fov_y_half_tan - 1e-5
    fv = (2 * (vv + 0.5) / W - 1) * fov_x_half_tan - 1e-5

    dir_local = torch.stack([torch.ones_like(fu), -fv, -fu], dim=-1)
    dir_world = torch.einsum('bij,bhwj->bhwi', R, dir_local)
    P_cur = pos[:, None, None, :] + depth[..., None] * dir_world

    flow_u = flow[..., 0]
    flow_v = flow[..., 1]
    u_old = uu - flow_u
    v_old = vv - flow_v

    if prev_depth is not None:
        norm_x = 2 * v_old / (W - 1) - 1
        norm_y = 2 * u_old / (H - 1) - 1
        grid = torch.stack([norm_x, norm_y], dim=-1)
        Z_old = F.grid_sample(prev_depth[:, None], grid, mode='bilinear',
                              padding_mode='border', align_corners=True)[:, 0]
        out_of_bounds = (u_old < 0) | (u_old > H - 1) | (v_old < 0) | (v_old > W - 1)
    else:
        Z_old = depth * (1 + ctl_dt / ttc.clamp_min(1e-3))
        out_of_bounds = torch.zeros_like(depth, dtype=torch.bool)

    fu_old = (2 * (u_old + 0.5) / H - 1) * fov_y_half_tan - 1e-5
    fv_old = (2 * (v_old + 0.5) / W - 1) * fov_x_half_tan - 1e-5
    dir_old_local = torch.stack([torch.ones_like(fu_old), -fv_old, -fu_old], dim=-1)
    dir_old_world = torch.einsum('bij,bhwj->bhwi', R_old, dir_old_local)
    P_old = pos_old[:, None, None, :] + Z_old[..., None] * dir_old_world

    scene_flow = (P_cur - P_old) / ctl_dt
    valid_mask = ((depth < valid_max_depth) & (Z_old > 0) &
                  (Z_old < valid_max_depth) & (~out_of_bounds))
    scene_flow = torch.where(valid_mask[..., None], scene_flow, torch.zeros_like(scene_flow))
    return scene_flow, valid_mask


def compute_input_dims(args):
    if args.use_sceneflow_only:
        return 3
    if args.use_flow_only:
        return 2
    if args.use_sceneflow:
        return 6
    if args.use_flow:
        return 3
    return 1


def compute_dim_obs(args):
    dim_obs = 7
    if not args.no_odom:
        dim_obs += 3
    if args.use_harmonic:
        dim_obs += 3
    return dim_obs


def build_model_input(args, env, depth, flow, ttc, prev_depth, ctl_dt):
    if args.use_ttc or args.use_sceneflow:
        ttc_scale = 10.0
        x = 3 / ttc.clamp(0.3, ttc_scale) - 0.6 + torch.randn_like(ttc) * 0.02
        x = F.max_pool2d(x[:, None], 4, 4)
    else:
        x = 3 / depth.clamp_(0.3, 48) - 0.6 + torch.randn_like(depth) * 0.02
        x = F.max_pool2d(x[:, None], 4, 4)

    if args.use_flow or args.use_sceneflow:
        flow_scale = 20.0
        flow_clamped = flow.clamp(-flow_scale, flow_scale)
        flow_clamped = flow_clamped + torch.randn_like(flow_clamped) * 0.5
        flow_chw = flow_clamped.permute(0, 3, 1, 2)
        flow_x = F.avg_pool2d(flow_chw, 4, 4) / flow_scale
        x = flow_x if args.use_flow_only else torch.cat([x, flow_x], 1)

    if args.use_sceneflow or args.use_sceneflow_only:
        scene_flow, _ = compute_scene_flow(
            depth, flow, env.R_cam_world, env.R_cam_world_old, env.p_cam, env.p_cam_old,
            ctl_dt, args.fov_x_half_tan, prev_depth=prev_depth, ttc=ttc)
        scene_flow_chw = scene_flow.permute(0, 3, 1, 2)
        scene_flow_x = F.avg_pool2d(scene_flow_chw, 4, 4) / 10.0
        x = scene_flow_x if args.use_sceneflow_only else torch.cat([x, scene_flow_x], 1)

    return x


# -----------------------------------------------------------------------------
# Arguments
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--resume', required=True, help='trained policy checkpoint')
    p.add_argument('--seed', type=int, default=0, help='same seed -> same scenes')
    p.add_argument('--num_episodes', type=int, default=10, help='batches of --batch_size drones')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--timesteps', type=int, default=150)

    # Planner
    p.add_argument('--use_planner', default=False, action='store_true')
    p.add_argument('--plan_every', type=int, default=3, help='control steps per plan (3 -> 5 Hz)')
    p.add_argument('--planner_t_min', type=float, default=0.5, help='time-headway margin (s)')
    p.add_argument('--planner_T', type=float, default=0.3, help='policy velocity time constant (s)')
    p.add_argument('--planner_horizon', type=float, default=1.5, help='lookahead at current speed (s)')
    p.add_argument('--planner_w_c', type=float, default=100.0, help='collision weight (0 = ignore obstacles)')
    p.add_argument('--planner_w_s', type=float, default=0.2, help='smoothness weight')
    p.add_argument('--planner_w_u', type=float, default=2.0, help='unseen-space weight (0 = off)')
    p.add_argument('--planner_speeds', default='0.5,0.75,1.0',
                   help="candidate speed scales; '1.0' = steer only, never slow down")
    p.add_argument('--speed_scale', type=float, default=1.0,
                   help='control: scale target_v speed by this everywhere (uniform slowdown)')
    p.add_argument('--brake', default=False, action='store_true',
                   help='hybrid braking: when the warning fires, slow target_v down (no steering)')
    p.add_argument('--brake_signal', choices=['cur', 'goal'], default='cur',
                   help="warning used for braking: current path ('cur') or goal direction ('goal')")
    p.add_argument('--brake_t', type=float, default=0.05, help='braking threshold (s)')
    p.add_argument('--brake_factor', type=float, default=0.3, help='target speed multiplier while braking')
    p.add_argument('--planner_shadow', default=False, action='store_true',
                   help='run the planner but ignore its output; report whether it saw crashes coming')
    p.add_argument('--warn_window', type=float, default=1.0,
                   help='shadow mode: a crash counts as warned if flagged within this many seconds before it')
    p.add_argument('--roc_thresholds', default='0.05,0.1,0.15,0.2,0.3',
                   help='shadow mode: warning thresholds (s) to compare')
    p.add_argument('--planner_gate', default=False, action='store_true',
                   help='only intervene when flying straight at the goal is blocked')
    p.add_argument('--utrap', default=False, action='store_true',
                   help='replace the random scene with a U-shaped dead end (forces --yaw_drift off)')
    p.add_argument('--reach_radius', type=float, default=1.5, help='goal counts as reached within this (m)')
    p.add_argument('--layer_b', default=False, action='store_true',
                   help='Layer B: heading memory + trap detection + escape + trap avoidance '
                        '(forces --yaw_drift off). Combine with --use_planner for A+B')
    p.add_argument('--no_yaw_drift', default=False, action='store_true',
                   help='turn --yaw_drift off (for baselines comparable to --layer_b runs)')
    p.add_argument('--lb_window', type=float, default=2.0, help='trap detector: progress window (s)')
    p.add_argument('--lb_min_progress', type=float, default=1.0,
                   help='trap detector: metres of goal progress expected per window')
    p.add_argument('--lb_speed_frac', type=float, default=0.5, help='escape speed, fraction of normal')
    p.add_argument('--lb_exit_dist', type=float, default=8.0, help='escape ends after this far (m)')
    p.add_argument('--lb_timeout', type=float, default=8.0, help='escape ends after this long (s)')
    p.add_argument('--lb_radius', type=float, default=6.0, help='keep-out radius around a trap (m)')
    p.add_argument('--lb_progress', choices=['goal', 'command'], default='command',
                   help="trap detector progress: toward the goal, or along the commanded direction")
    p.add_argument('--step_test', default=False, action='store_true',
                   help='empty scene; at --step_time give half the drones a direction step and half a '
                        'speed step, and time how fast velocity follows (measures the policy time constant T)')
    p.add_argument('--step_time', type=float, default=3.0, help='when the step command starts (s)')
    p.add_argument('--step_yaw_deg', type=float, default=45.0, help='size of the direction step (deg)')
    p.add_argument('--step_speed', type=float, default=0.3, help='speed step: new speed as a fraction')
    p.add_argument('--heading_mode', choices=['plan', 'goal'], default='plan',
                   help="camera faces the planned direction ('plan') or the goal, as in training ('goal')")

    # Env / model settings: must match the checkpoint's training run
    p.add_argument('--grad_decay', type=float, default=0.4)
    p.add_argument('--speed_mtp', type=float, default=1.0)
    p.add_argument('--fov_x_half_tan', type=float, default=0.53)
    p.add_argument('--cam_angle', type=int, default=10)
    p.add_argument('--single', default=False, action='store_true')
    p.add_argument('--gate', default=False, action='store_true')
    p.add_argument('--ground_voxels', default=False, action='store_true')
    p.add_argument('--scaffold', default=False, action='store_true')
    p.add_argument('--random_rotation', default=False, action='store_true')
    p.add_argument('--yaw_drift', default=False, action='store_true')
    p.add_argument('--gimbal_stabilized', default=False, action='store_true')
    p.add_argument('--active_yaw', default=False, action='store_true')
    p.add_argument('--yaw_offset_gain', type=float, default=1.0)
    p.add_argument('--no_odom', default=False, action='store_true')
    p.add_argument('--use_flow', default=False, action='store_true')
    p.add_argument('--use_ttc', default=False, action='store_true')
    p.add_argument('--use_harmonic', default=False, action='store_true')
    p.add_argument('--use_sceneflow', default=False, action='store_true')
    p.add_argument('--use_sceneflow_only', default=False, action='store_true')
    p.add_argument('--use_flow_only', default=False, action='store_true')
    p.add_argument('--harmonic_max_iterations', type=int, default=500)
    p.add_argument('--harmonic_process', choices=['cpu', 'gpu'], default='gpu')
    p.add_argument('--avoid_margin_cells', type=int, default=0)
    p.add_argument('--harmonic_eps', type=float, default=1e-3)
    p.add_argument('--harmonic_min_forward', type=float, default=0.3)

    # Accept the full training command: training-only flags (loss coefficients,
    # lr, num_iters, ...) don't affect evaluation, so they're ignored.
    args, ignored = p.parse_known_args()
    if ignored:
        print("ignoring training-only args:", " ".join(ignored))
    return args


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------

@torch.no_grad()
def main():
    args = parse_args()
    print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda')
    B = args.batch_size

    env = Env(B, 64, 48, args.grad_decay, device,
              fov_x_half_tan=args.fov_x_half_tan, single=args.single,
              gate=args.gate, ground_voxels=args.ground_voxels,
              scaffold=args.scaffold, speed_mtp=args.speed_mtp,
              random_rotation=args.random_rotation, cam_angle=args.cam_angle,
              gimbal_stabilized=args.gimbal_stabilized)

    dim_action = 9 if args.active_yaw else 6
    model = Model(compute_dim_obs(args), dim_action, compute_input_dims(args)).to(device)
    state_dict = torch.load(args.resume, map_location=device)
    missing, unexpected = model.load_state_dict(state_dict, False)
    if missing:
        print("missing_keys:", missing)
    if unexpected:
        print("unexpected_keys:", unexpected)
    model.eval()

    planner = None
    if args.use_planner and args.brake:
        raise SystemExit("use either --use_planner (steering) or --brake, not both")
    if args.use_planner or args.planner_shadow or args.brake:
        # Shadow mode always gates, so last_intervened = "planner says the goal direction is blocked"
        gated = args.planner_gate or args.planner_shadow or args.brake
        planner = TTCPlannerBridge(args.fov_x_half_tan, device,
                                   T=args.planner_T, horizon=args.planner_horizon,
                                   t_min=args.planner_t_min,
                                   w_c=args.planner_w_c, w_s=args.planner_w_s,
                                   w_u=args.planner_w_u,
                                   gate_t=args.planner_t_min if gated else None,
                                   speed_scales=[float(s) for s in args.planner_speeds.split(',')])
    use_plan = planner is not None and not args.planner_shadow and not args.brake

    # Shadow-mode records, pooled over all episodes
    thresholds = [float(x) for x in args.roc_thresholds.split(',')]
    signals = ['cur', 'goal']   # current velocity path / straight-to-goal path
    shadow = {
        'roc': {s: {th: {'warned': [], 'lead': [], 'fa': 0.0, 'fa_n': 0.0} for th in thresholds}
                for s in signals},
        'seen_recent': [], 'seen_3s': [], 'where': {},
    }

    def warning_lead(flags_b, times, tc, window):
        """How long the warning had been continuously on before impact (None if not on in window)."""
        idx = [i for i, t in enumerate(times) if t <= tc and flags_b[i]]
        if not idx or times[idx[-1]] < tc - window:
            return None
        start = idx[-1]
        while start > 0 and flags_b[start - 1]:
            start -= 1
        return tc - times[start]

    if args.utrap and args.step_test:
        raise SystemExit("use either --utrap or --step_test, not both")
    if (args.utrap or args.step_test or args.layer_b or args.no_yaw_drift) and args.yaw_drift:
        print("turning off --yaw_drift so target_v keeps pointing at the goal")
        args.yaw_drift = False

    lb = None
    if args.layer_b:
        lb = LayerB(B, device,
                    detector_kw=dict(window=args.lb_window, min_progress=args.lb_min_progress,
                                     progress=args.lb_progress),
                    escape_kw=dict(speed_frac=args.lb_speed_frac, exit_dist=args.lb_exit_dist,
                                   timeout=args.lb_timeout),
                    avoider_kw=dict(radius=args.lb_radius))
    step_results = {'dir': {'t63': [], 't90': []}, 'speed': {'t63': [], 't90': []}}

    # U-trap breakdown of drones that didn't make it, pooled over episodes
    speed_bins = [(0.0, 6.0, '3-6 m/s'), (6.0, 10.0, '6-10 m/s'), (10.0, 99.0, '10-13 m/s')]
    utrap_diag = {
        'n_stuck': 0,
        'closest': [],               # closest approach to the goal of each stuck drone (m)
        'region': {'inside cup': 0, 'in front of cup': 0, 'beside cup': 0, 'behind cup': 0},
        'state': {'escaping': 0, 'going around': 0, 'normal': 0},
        'speed': {name: {'n': 0, 'reached': 0, 'stuck': 0, 'crashed': 0} for _, _, name in speed_bins},
    }

    if args.use_harmonic:
        from Harmonic_FN.mapping import depth_map_to_action_batch

    totals = {k: [] for k in ['success', 'avg_speed', 'max_speed', 'ar',
                              'final_goal_dist', 'planner_headway', 'deviation_deg',
                              'yaw_dev_deg', 'pitch_dev_deg', 'coll_active_frac',
                              'all_blocked_frac', 'unseen_frac',
                              'toward_obstacle_frac', 'side_flip_frac', 'intervene_frac',
                              'reached', 'crashed', 'stuck', 'time_to_goal_s', 'brake_frac',
                              'escapes_per_drone', 'escaped_frac', 'escaping_frac', 'avoiding_frac',
                              'retraps_while_avoid', 'cleared_trap']}

    def elevation(v):
        return torch.rad2deg(torch.atan2(v[:, 2], v[:, :2].norm(dim=-1)))

    def heading(v):
        return torch.atan2(v[:, 1], v[:, 0])

    for ep in range(args.num_episodes):
        env.reset()
        if args.utrap:
            setup_utrap(env)
        if args.step_test:
            setup_open(env)
        # Step test: even drones get a direction step (alternating left/right), odd ones a speed step
        idx = torch.arange(B, device=device)
        step_is_dir = idx % 2 == 0
        step_sign = 1.0 - 2.0 * ((idx // 2) % 2).float()           # +1, +1, -1, -1, ... (left/right)
        step_v = None              # commanded velocity after the step
        step_t0 = None             # time the step command started
        step_psi = None            # goal heading at the step
        step_times = []
        model.reset()
        if planner is not None:
            planner.reset()
        crashed = torch.zeros(B, dtype=torch.bool, device=device)
        reached = torch.zeros(B, dtype=torch.bool, device=device)
        reach_step = torch.full((B,), float('nan'), device=device)
        crash_time = torch.full((B,), float('nan'), device=device)
        cleared = torch.zeros(B, dtype=torch.bool, device=device)       # U-trap: got past the back wall
        min_goal_dist = torch.full((B,), float('inf'), device=device)   # closest approach to the goal
        flag_hist, flag_times, moving_hist = [], [], []    # shadow: "goal direction blocked" per plan
        cur_flag_hist = []                                 # shadow: "current velocity heading into something"
        pose_hist, pcam_hist, pose_times = [], [], []      # shadow: camera pose every step
        hit_pt = torch.zeros(B, 3, device=device)          # where each drone hit, its position and velocity then
        hit_pos = torch.zeros(B, 3, device=device)
        hit_vel = torch.zeros(B, 3, device=device)
        sim_time = 0.0
        intervene_log = []
        h = None
        yaw_log, pitch_log, coll_log, blocked_log, unseen_log = [], [], [], [], []
        toward_log, flip_log = [], []
        prev_side = torch.zeros(B, device=device)
        prev_depth = None
        planned = None
        brake_mask = torch.zeros(B, dtype=torch.bool, device=device)
        brake_log = []
        if lb is not None:
            lb.reset()
        lb_desired = None            # Layer B's output, held between plans
        lb_blocked = None            # Layer A's "nothing clear" flag from the last plan
        lb_escaping_log, lb_avoiding_log = [], []

        act_lag = 1
        act_buffer = [env.act] * (act_lag + 1)
        yaw_offset_buffer = [torch.zeros_like(env.v)] * (act_lag + 1)

        v_history, vec_to_pt_history = [], []
        headway_log, deviation_log = [], []

        target_v_raw = env.p_target - env.p
        if args.yaw_drift:
            drift_av = torch.randn(B, device=device) * (5 * math.pi / 180 / 15)
            zeros, ones = torch.zeros_like(drift_av), torch.ones_like(drift_av)
            R_drift = torch.stack([torch.cos(drift_av), -torch.sin(drift_av), zeros,
                                   torch.sin(drift_av), torch.cos(drift_av), zeros,
                                   zeros, zeros, ones], -1).reshape(B, 3, 3)

        for t in range(args.timesteps):
            ctl_dt = normalvariate(1 / 15, 0.1 / 15)
            depth, flow = env.render(ctl_dt)
            vec_to_pt_history.append(env.find_vec_to_nearest_pt())

            # Outcome tracking: reached counts only if it happened before any crash
            vec = vec_to_pt_history[-1]                                   # (10, B, 3)
            gap = vec.norm(2, -1) - env.margin                            # (10, B)
            was_crashed = crashed.clone()
            crashed |= (gap <= 0).any(0)
            newly_crashed = crashed & ~was_crashed
            crash_time = torch.where(newly_crashed,
                                     torch.full_like(crash_time, sim_time), crash_time)

            if args.planner_shadow:
                # Record the exact impact point, plus the camera pose every step
                if newly_crashed.any():
                    k = gap.argmin(0)                                     # sub-step of first contact
                    p_sub = env.p[None] + env.v[None] * env.sub_div       # (10, B, 3)
                    hit = (p_sub + vec)[k, torch.arange(B, device=device)]
                    m = newly_crashed[:, None]
                    hit_pt = torch.where(m, hit, hit_pt)
                    hit_pos = torch.where(m, env.p, hit_pos)
                    hit_vel = torch.where(m, env.v, hit_vel)
                pose_hist.append(env.R_cam_world.clone())
                pcam_hist.append(env.p_cam.clone())
                pose_times.append(sim_time)
            goal_dist_now = (env.p_target - env.p).norm(dim=-1)
            min_goal_dist = torch.minimum(min_goal_dist, goal_dist_now)
            if args.utrap:
                cleared |= (env.p[:, 0] > env.utrap_geom[0] + 0.5) & ~crashed
            newly = (~reached) & (~crashed) & (goal_dist_now < args.reach_radius)
            reach_step = torch.where(newly, torch.full_like(reach_step, sim_time), reach_step)
            reached |= newly
            sim_time += ctl_dt

            if args.yaw_drift:
                target_v_raw = torch.squeeze(target_v_raw[:, None] @ R_drift, 1)
            else:
                target_v_raw = env.p_target - env.p

            heading_cmd = target_v_raw

            # ---------------- LAYER B ----------------
            n_raw = target_v_raw.norm(dim=-1, keepdim=True)
            goal_v_normal = target_v_raw / n_raw * torch.minimum(n_raw, env.max_speed)
            if lb is not None:
                if t % args.plan_every == 0:
                    rays_w, dist_w = memory_rays(depth, env.R_cam_world, env._fov_x_half_tan)
                    lb_desired, lb_info = lb.step(sim_time, env.p, env.p_target, goal_v_normal,
                                                  rays_w, dist_w, lb_blocked)
                    lb_escaping_log.append(lb_info['escaping'].float())
                    lb_avoiding_log.append(lb_info['avoiding'].float())
                heading_cmd = lb_desired     # camera follows Layer B (overridden by the planner below)
            # -----------------------------------------

            # ---------------- STEP TEST ----------------
            if args.step_test and sim_time >= args.step_time:
                if step_v is None:
                    g = target_v_raw
                    n = g.norm(dim=-1, keepdim=True)
                    speed0 = torch.minimum(n, env.max_speed)[:, 0]                 # normal target speed
                    step_psi = torch.atan2(g[:, 1], g[:, 0])
                    yaw = step_psi + step_is_dir * step_sign * math.radians(args.step_yaw_deg)
                    dirn = torch.stack([torch.cos(yaw), torch.sin(yaw), torch.zeros_like(yaw)], -1)
                    speed = speed0 * (step_is_dir.float() + (~step_is_dir).float() * args.step_speed)
                    step_v = dirn * speed[:, None]
                    step_t0 = sim_time - ctl_dt        # command acts from the start of this step
                heading_cmd = step_v                   # camera follows the command, as with the planner
            # -------------------------------------------

            # ---------------- PLANNER ----------------
            if planner is not None:
                if t % args.plan_every == 0:
                    # Layer A plans toward Layer B's output when Layer B is on
                    goal_v = lb_desired if lb is not None else goal_v_normal
                    planned = planner(depth, env.v, goal_v, env.R_cam_world)
                    if lb is not None:
                        lb_blocked = ~planner.last_any_free & (env.v.norm(dim=-1) >= planner.min_speed)

                    headway_log.append(planner.last_headway)
                    cos_dev = F.cosine_similarity(planned, goal_v, dim=-1).clamp(-1, 1)
                    deviation_log.append(torch.rad2deg(torch.acos(cos_dev)))

                    dyaw = heading(planned) - heading(goal_v)
                    dyaw = torch.atan2(torch.sin(dyaw), torch.cos(dyaw)).abs()
                    yaw_log.append(torch.rad2deg(dyaw))
                    pitch_log.append((elevation(planned) - elevation(goal_v)).abs())
                    moving = env.v.norm(dim=-1) >= planner.min_speed
                    coll_log.append((planner.last_coll_active & moving).float())
                    blocked_log.append((~planner.last_any_free & moving).float())
                    unseen_log.append(planner.last_unseen[moving])
                    intervene_log.append(planner.last_intervened.float())
                    flag_hist.append(planner.last_goal_headway.clone())
                    cur_flag_hist.append(planner.last_cur_headway.clone())

                    if args.brake:
                        hw_sig = planner.last_cur_headway if args.brake_signal == 'cur' \
                            else planner.last_goal_headway
                        brake_mask = (hw_sig < args.brake_t) & moving   # held until the next plan
                        brake_log.append(brake_mask[moving].float())
                    flag_times.append(sim_time)
                    moving_hist.append(moving)

                    # Ground-truth checks, only when the planner actually steers (>3 deg)
                    steering = moving & (cos_dev < math.cos(math.radians(3)))
                    # 1. Does it steer toward or away from the nearest real obstacle (within 5 m)?
                    vec_obs = vec_to_pt_history[-1][0]     # (B, 3), current position
                    near_obs = steering & (vec_obs.norm(dim=-1) < 5.0)
                    steer = F.normalize(planned, dim=-1) - F.normalize(goal_v, dim=-1)
                    toward = (steer * vec_obs).sum(-1) > 0
                    if near_obs.any():
                        toward_log.append(toward[near_obs].float())
                    # 2. Does it keep switching sides of the goal from one plan to the next?
                    side = torch.sign(goal_v[:, 0] * planned[:, 1] - goal_v[:, 1] * planned[:, 0])
                    side = torch.where(steering, side, torch.zeros_like(side))
                    both = (side != 0) & (prev_side != 0)
                    if both.any():
                        flip_log.append((side[both] != prev_side[both]).float())
                    prev_side = side
                if use_plan and args.heading_mode == 'plan':
                    heading_cmd = planned  # camera turns toward where the planner is going
            # -----------------------------------------

            if args.active_yaw:
                heading_cmd = heading_cmd + args.yaw_offset_gain * yaw_offset_buffer[t]
            env.run(act_buffer[t], ctl_dt, heading_cmd)

            fwd = env.R[:, :, 0].clone()
            up = torch.zeros_like(fwd)
            fwd[:, 2] = 0
            up[:, 2] = 1
            fwd = F.normalize(fwd, 2, -1)
            R = torch.stack([fwd, torch.cross(up, fwd, dim=-1), up], -1)

            target_v_norm = torch.norm(target_v_raw, 2, -1, keepdim=True)
            target_v = target_v_raw / target_v_norm * torch.minimum(target_v_norm, env.max_speed)

            # ---------------- PLANNER ----------------
            if lb is not None:
                target_v = lb_desired      # Layer B's goal (replaced by Layer A's plan if it's on)
            if use_plan:
                target_v = planned         # the policy flies the planner's direction
            if args.speed_scale != 1.0:
                target_v = target_v * args.speed_scale   # uniform slowdown (control)
            if args.step_test and step_v is not None:
                target_v = step_v                         # step command
            if args.brake:
                # same direction, slower target speed while the warning is on
                target_v = torch.where(brake_mask[:, None], target_v * args.brake_factor, target_v)
            # -----------------------------------------

            if args.use_ttc or args.use_sceneflow:
                local_v = torch.squeeze(env.v[:, None] @ R, 1)
                radial_v = local_v[:, 0].clamp_min(1e-3)
                ttc = depth / radial_v[:, None, None]
            else:
                ttc = None
            x = build_model_input(args, env, depth, flow, ttc, prev_depth, ctl_dt)
            prev_depth = depth.clone()

            state = [torch.squeeze(target_v[:, None] @ R, 1),
                     env.R[:, 2],
                     env.margin[:, None]]

            if args.use_harmonic:
                R_np = R.cpu().numpy()
                goal_flu = np.squeeze(R_np.transpose(0, 2, 1) @ target_v_raw.cpu().numpy()[..., None], -1)
                flip = np.array([1., -1., -1.], dtype=np.float32)
                harmonic_frd = depth_map_to_action_batch(
                    ttc.cpu().numpy(), goal_flu * flip, fov_x_half_tan=env._fov_x_half_tan,
                    max_iterations=args.harmonic_max_iterations, process=args.harmonic_process,
                    safety_margin_cells=args.avoid_margin_cells, smooth=True,
                    harmonic_eps=args.harmonic_eps, harmonic_min_forward=args.harmonic_min_forward)
                state.append(torch.from_numpy(harmonic_frd * flip).to(device))

            local_v = torch.squeeze(env.v[:, None] @ R, 1)
            if not args.no_odom:
                state.insert(0, local_v)
            state = torch.cat(state, -1)

            act, values, h = model(x, state, h)
            act_vecs = (R @ act.reshape(B, 3, -1)).unbind(-1)
            a_pred, v_pred = act_vecs[0], act_vecs[1]
            if args.active_yaw:
                yaw_offset_buffer.append(act_vecs[2])
            act = (a_pred - v_pred - env.g_std) * env.thr_est_error[:, None] + env.g_std
            act_buffer.append(act)

            v_history.append(env.v)
            step_times.append(sim_time)

        # ---------------- metrics (same definitions as training) ----------------
        v_history = torch.stack(v_history)
        speed = v_history.norm(2, -1)                                   # (T, B)
        distance = torch.stack(vec_to_pt_history).norm(2, -1) - env.margin
        success = torch.all(distance.flatten(0, 1) > 0, 0)             # (B,) never collided
        avg_speed = speed.mean(0)

        totals['success'].append(success.float().mean().item())
        totals['avg_speed'].append(avg_speed.mean().item())
        totals['max_speed'].append(speed.max(0).values.mean().item())
        totals['ar'].append((success * avg_speed).mean().item())
        totals['final_goal_dist'].append((env.p_target - env.p).norm(dim=-1).mean().item())
        if headway_log:
            hw = torch.stack(headway_log)
            totals['planner_headway'].append(hw[torch.isfinite(hw)].mean().item())
            totals['deviation_deg'].append(torch.stack(deviation_log).mean().item())
            totals['yaw_dev_deg'].append(torch.stack(yaw_log).mean().item())
            totals['pitch_dev_deg'].append(torch.stack(pitch_log).mean().item())
            totals['coll_active_frac'].append(torch.stack(coll_log).mean().item())
            totals['all_blocked_frac'].append(torch.stack(blocked_log).mean().item())
            uns = torch.cat(unseen_log)
            if uns.numel():
                totals['unseen_frac'].append(uns.mean().item())
            if toward_log:
                totals['toward_obstacle_frac'].append(torch.cat(toward_log).mean().item())
            if flip_log:
                totals['side_flip_frac'].append(torch.cat(flip_log).mean().item())
            if intervene_log:
                totals['intervene_frac'].append(torch.stack(intervene_log).mean().item())
            if brake_log:
                bl = torch.cat(brake_log)
                if bl.numel():
                    totals['brake_frac'].append(bl.mean().item())
        if lb is not None:
            n_esc = lb.escape.n_escapes.float()
            totals['escapes_per_drone'].append(n_esc.mean().item())
            totals['escaped_frac'].append((n_esc > 0).float().mean().item())
            totals['retraps_while_avoid'].append(lb.n_retrap_avoiding.float().mean().item())
            if lb_escaping_log:
                totals['escaping_frac'].append(torch.stack(lb_escaping_log).mean().item())
                totals['avoiding_frac'].append(torch.stack(lb_avoiding_log).mean().item())

        # ---------------- step-response analysis ----------------
        if args.step_test and step_v is not None:
            tt = torch.tensor(step_times, device=device)                  # (T,)
            pre = (tt >= step_t0 - 0.5) & (tt < step_t0)                  # 0.5 s before the step
            for b in range(B):
                if crashed[b]:
                    continue
                vb = v_history[:, b]                                      # (T, 3)
                if step_is_dir[b]:
                    # heading change toward the commanded side, normalized to the step size
                    ang = torch.atan2(vb[:, 1], vb[:, 0]) - step_psi[b]
                    ang = torch.atan2(torch.sin(ang), torch.cos(ang)) * step_sign[b]
                    a0 = ang[pre].mean()
                    sig = (ang - a0) / (math.radians(args.step_yaw_deg) - a0)
                    kind = 'dir'
                else:
                    # speed drop, normalized to the commanded drop
                    s = vb.norm(dim=-1)
                    s0 = s[pre].mean()
                    s1 = step_v[b].norm()
                    if s0 - s1 < 0.5:          # barely moving before the step: nothing to measure
                        continue
                    sig = (s0 - s) / (s0 - s1)
                    kind = 'speed'
                t63, t90 = step_response_times(tt, sig, step_t0)
                step_results[kind]['t63'].append(t63)
                step_results[kind]['t90'].append(t90)

        # ---------------- shadow-mode crash analysis ----------------
        if args.planner_shadow and flag_hist:
            hw = {'goal': torch.stack(flag_hist), 'cur': torch.stack(cur_flag_hist)}      # (Tp, B) headways
            mov = torch.stack(moving_hist)                                 # (Tp, B)
            poses = torch.stack(pose_hist)                                 # (T, B, 3, 3)
            pcams = torch.stack(pcam_hist)                                 # (T, B, 3)
            pt = torch.tensor(pose_times, device=device)                   # (T,)
            fx = env._fov_x_half_tan
            fy = fx * depth.shape[1] / depth.shape[2]

            for b in torch.nonzero(crashed).flatten().tolist():
                tc = crash_time[b].item()
                if tc < 0.5:            # crashed at spawn: nothing anyone could do
                    continue

                # Each signal at each threshold: warned before impact, and for how long
                for s in signals:
                    for th in thresholds:
                        fl_b = ((hw[s][:, b] < th) & mov[:, b]).tolist()
                        lead = warning_lead(fl_b, flag_times, tc, args.warn_window)
                        rec = shadow['roc'][s][th]
                        rec['warned'].append(float(lead is not None))
                        if lead is not None:
                            rec['lead'].append(lead)

                # Was the exact impact point inside the camera's view earlier? (ignores occlusion)
                rel = hit_pt[b] - pcams[:, b]                              # (T, 3)
                cam = torch.einsum('tji,tj->ti', poses[:, b], rel)         # world -> camera FLU
                inview = ((cam[:, 0] > 0) & (cam[:, 1].abs() < fx * cam[:, 0])
                          & (cam[:, 2].abs() < fy * cam[:, 0]) & (rel.norm(dim=-1) < 90))
                recent = (pt <= tc - 0.3) & (pt >= tc - args.warn_window)
                earlier = (pt <= tc - 0.3) & (pt >= tc - 3.0)
                shadow['seen_recent'].append(float((inview & recent).any()))
                shadow['seen_3s'].append(float((inview & earlier).any()))

                # Where the impact point was relative to the drone's direction of travel
                r = hit_pt[b] - hit_pos[b]
                elev = math.degrees(math.atan2(r[2].item(), r[:2].norm().item()))
                if elev < -30:
                    where = 'below'
                elif elev > 30:
                    where = 'above'
                else:
                    f = hit_vel[b, :2]
                    f = f / f.norm() if f.norm() > 1e-3 else torch.tensor([1.0, 0.0], device=device)
                    az = abs(math.degrees(torch.atan2(f[0] * r[1] - f[1] * r[0], (f * r[:2]).sum()).item()))
                    where = 'ahead' if az < 45 else ('side' if az < 135 else 'behind')
                shadow['where'][where] = shadow['where'].get(where, 0) + 1

            # False alarms: flag rate on drones that never crashed, while moving
            ok = (~crashed)[None] & mov
            for s in signals:
                for th in thresholds:
                    rec = shadow['roc'][s][th]
                    rec['fa'] += ((hw[s] < th) & ok).float().sum().item()
                    rec['fa_n'] += ok.float().sum().item()

        # ---------------- U-trap breakdown ----------------
        if args.utrap:
            stuck_m = ~reached & ~crashed
            Xw, Wc, Lc = env.utrap_geom
            x, y = env.p[:, 0], env.p[:, 1]
            front = x <= Xw - Lc
            behind = x >= Xw
            inside = ~front & ~behind & (y.abs() < Wc / 2)
            beside = ~front & ~behind & ~inside
            for name, m in [('inside cup', inside), ('in front of cup', front),
                            ('beside cup', beside), ('behind cup', behind)]:
                utrap_diag['region'][name] += (stuck_m & m).sum().item()
            if lb is not None:
                esc_m = lb.escape.escaping
                avo_m = lb.avoider.active & ~esc_m
            else:
                esc_m = avo_m = torch.zeros_like(stuck_m)
            for name, m in [('escaping', esc_m), ('going around', avo_m), ('normal', ~esc_m & ~avo_m)]:
                utrap_diag['state'][name] += (stuck_m & m).sum().item()
            utrap_diag['n_stuck'] += stuck_m.sum().item()
            utrap_diag['closest'] += min_goal_dist[stuck_m].tolist()
            totals['cleared_trap'].append(cleared.float().mean().item())
            ms = env.max_speed[:, 0]
            for lo, hi, name in speed_bins:
                inb = (ms >= lo) & (ms < hi)
                rec = utrap_diag['speed'][name]
                rec['n'] += inb.sum().item()
                rec['reached'] += (inb & reached).sum().item()
                rec['stuck'] += (inb & stuck_m).sum().item()
                rec['crashed'] += (inb & crashed & ~reached).sum().item()

        totals['reached'].append(reached.float().mean().item())
        totals['crashed'].append((crashed & ~reached).float().mean().item())
        totals['stuck'].append((~reached & ~crashed).float().mean().item())
        if reached.any():
            totals['time_to_goal_s'].append(reach_step[reached].mean().item())

        print(f"episode {ep + 1}/{args.num_episodes}: "
              f"success {totals['success'][-1]:.3f}  "
              f"avg_speed {totals['avg_speed'][-1]:.2f}  "
              f"final_goal_dist {totals['final_goal_dist'][-1]:.1f}  "
              f"reached {totals['reached'][-1]:.2f}  stuck {totals['stuck'][-1]:.2f}  "
              f"crashed {totals['crashed'][-1]:.2f}")

    if use_plan:
        mode = "(planner steering)"
    elif args.brake:
        mode = f"(hybrid braking: {args.brake_signal} < {args.brake_t} s -> x{args.brake_factor})"
    elif planner is not None:
        mode = "(reactive only, planner in shadow)"
    else:
        mode = "(reactive only)"
    print("\n==== summary", mode, "====")
    for k, vals in totals.items():
        if vals:
            print(f"{k:>20}: {sum(vals) / len(vals):.3f}")

    if args.planner_shadow:
        mean = lambda xs: sum(xs) / len(xs) if xs else float('nan')
        n = len(shadow['seen_3s'])
        print(f"\n==== shadow: could the planner see the crashes coming? ====")
        print(f"crashes analysed: {n}")
        if n:
            names = {'cur': 'current path blocked', 'goal': 'goal direction blocked'}
            for s in signals:
                print(f"\nwarning signal: {names[s]}")
                print(f"  threshold(s)   warned   lead(s)   false-alarm")
                for th in thresholds:
                    rec = shadow['roc'][s][th]
                    print(f"     {th:5.2f}      {mean(rec['warned']):6.3f}   {mean(rec['lead']):6.2f}    "
                          f"{rec['fa'] / max(rec['fa_n'], 1):6.3f}")

            print(f"\nimpact point in camera view")
            print(f"  0.3-{args.warn_window:.1f} s before impact:  {mean(shadow['seen_recent']):.3f}")
            print(f"  0.3-3 s before impact:    {mean(shadow['seen_3s']):.3f}")
            print(f"  never seen (3 s):         {1 - mean(shadow['seen_3s']):.3f}")

            print(f"\nwhere the impact was, relative to direction of travel")
            for k in ['ahead', 'side', 'behind', 'below', 'above']:
                print(f"  {k:>7}: {shadow['where'].get(k, 0) / n:.3f}")

    if args.utrap:
        n = utrap_diag['n_stuck']
        print(f"\n==== U-trap: drones that didn't reach the goal ({n} stuck) ====")
        if n:
            print("where they ended up:")
            for k, v in utrap_diag['region'].items():
                print(f"  {k:>16}: {v / n:.0%}")
            print("what Layer B was doing at the end:")
            for k, v in utrap_diag['state'].items():
                print(f"  {k:>16}: {v / n:.0%}")
            c = torch.tensor(utrap_diag['closest'])
            print("closest they ever got to the goal:")
            print(f"  median {c.median().item():.1f} m   "
                  f"within 3 m: {(c < 3).float().mean().item():.0%}   "
                  f"within 5 m: {(c < 5).float().mean().item():.0%}")
        print("\nby max speed:")
        print(f"  {'max speed':>10}  {'drones':>6}  {'reached':>7}  {'stuck':>6}  {'crashed':>7}")
        for name, r in utrap_diag['speed'].items():
            if r['n']:
                print(f"  {name:>10}  {r['n']:>6}  {r['reached'] / r['n']:>7.0%}  "
                      f"{r['stuck'] / r['n']:>6.0%}  {r['crashed'] / r['n']:>7.0%}")

    if args.step_test:
        def stats(xs):
            ok = torch.tensor([x for x in xs if x == x])          # drop nan (never reached)
            if len(ok) == 0:
                return float('nan'), float('nan')
            return ok.median().item(), ok.mean().item()

        print(f"\n==== step response (time after the command, in seconds) ====")
        names = {'dir': f"direction step ({args.step_yaw_deg:.0f} deg)",
                 'speed': f"speed step (to {args.step_speed:.0%})"}
        for kind in ['dir', 'speed']:
            r = step_results[kind]
            n = len(r['t63'])
            if n == 0:
                continue
            m63, a63 = stats(r['t63'])
            m90, a90 = stats(r['t90'])
            got63 = sum(1 for x in r['t63'] if x == x) / n
            got90 = sum(1 for x in r['t90'] if x == x) / n
            print(f"\n{names[kind]}  ({n} drones)")
            print(f"  T63 (time constant): median {m63:.2f}  mean {a63:.2f}   reached by {got63:.0%}")
            print(f"  T90:                 median {m90:.2f}  mean {a90:.2f}   reached by {got90:.0%}")


if __name__ == '__main__':
    main()