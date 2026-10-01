"""
Fly the trained TTC policy in Isaac Sim 5.1 + Pegasus (sim-to-sim check).

What is reproduced exactly from the training eval (eval_planner.py):
  - 64x48 depth camera, fov_x_half_tan, cam_angle tilt, rigidly mounted on the body
  - TTC = planar depth / forward speed in the gravity-level frame
  - model input 3/clamp(ttc, 0.3, 10) - 0.6 + noise, max-pooled by 4
  - state [local_v, target_v (level frame), body-z row of R, margin]
  - action decoding: act = a_pred - v_pred (world), heading = goal direction
  - 15 Hz control, act_buffer lag (computed at step t, applied at t+2), depth rendered
    before the physics step and read with the post-step state
What is different on purpose (the point of the test):
  - rendering (RTX depth), rigid-body physics (PhysX), Pegasus Iris rotors + linear drag,
  - a geometric attitude controller turning the commanded acceleration into rotor speeds.

Run from the repo root with Pegasus' launcher:

    isaac_run seeing_in_time/isaac_eval.py $(cat configs/single_agent.args) --use_ttc \
        --resume checkpoints/hynbrid/ttc/cmd_checkpoint0000.pth --scene utrap --episodes 5

Layer B:  add --layer_b to escape traps (without it the reactive policy stalls in the U,
          exactly like the 0% baseline in the training sim)
Viewport: --view fpv (default, through the policy camera) | chase | free

Scenes: --scene exported --scenes_file isaac_scenes.npz   (exact training geometry)
        --scene utrap | forest                              (built in)
"""

import argparse
import json
import math
import os
import sys
import time
from collections import deque

# ----------------------------------------------------------------------------- args
p = argparse.ArgumentParser()
p.add_argument("--resume", required=True)
p.add_argument("--repo_root", default=None, help="folder with model.py (default: parent of this file's folder)")
p.add_argument("--scene", choices=["exported", "utrap", "forest"], default="utrap")
p.add_argument("--scenes_file", default="isaac_scenes.npz")
p.add_argument("--episodes", type=int, default=5)
p.add_argument("--first_episode", type=int, default=0, help="index into the exported scenes")
p.add_argument("--timesteps", type=int, default=None, help="control steps (default 600 utrap, 150 otherwise)")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--max_speed", type=float, default=6.0, help="built-in scenes only")
p.add_argument("--margin", type=float, default=0.2, help="built-in scenes only")
p.add_argument("--alt", type=float, default=1.5, help="built-in scenes: flight altitude (m)")
p.add_argument("--goal_radius", type=float, default=1.5)
p.add_argument("--no_ground", action="store_true", help="no ground plane (render and collision)")
p.add_argument("--headless", action="store_true")
p.add_argument("--view", choices=["fpv", "chase", "free"], default="fpv",
               help="viewport: fpv = through the policy camera, chase = behind the drone, free = normal editor camera")
p.add_argument("--out", default="isaac_results")
# timing / control
p.add_argument("--physics_hz", type=int, default=240)
p.add_argument("--ctl_hz", type=int, default=15)
p.add_argument("--act_lag", type=int, default=1, help="control steps of action delay (training: 1)")
p.add_argument("--depth_lag", type=int, default=0, help="extra steps of depth staleness beyond training (0 = same as training)")
p.add_argument("--mass", type=float, default=1.5, help="Pegasus Iris mass (kg)")
p.add_argument("--kr", type=float, default=3.5)
p.add_argument("--kw", type=float, default=0.5)
p.add_argument("--no_drag", action="store_true", help="turn off Pegasus' linear body drag")
# must match the checkpoint
p.add_argument("--fov_x_half_tan", type=float, default=0.53)
p.add_argument("--cam_angle", type=float, default=10.0, help="camera pitch-up (deg)")
p.add_argument("--img_w", type=int, default=64)
p.add_argument("--img_h", type=int, default=48)
p.add_argument("--use_ttc", action="store_true")
p.add_argument("--no_odom", action="store_true")
p.add_argument("--active_yaw", action="store_true")
# Layer B (trap escape); same defaults as eval_planner.py
p.add_argument("--layer_b", action="store_true", help="run Layer B (memory + trap detector + escape + avoider)")
p.add_argument("--plan_every", type=int, default=3, help="control steps per Layer B update (3 -> 5 Hz)")
p.add_argument("--lb_window", type=float, default=2.0)
p.add_argument("--lb_min_progress", type=float, default=1.0)
p.add_argument("--lb_speed_frac", type=float, default=0.5)
p.add_argument("--lb_exit_dist", type=float, default=8.0)
p.add_argument("--lb_timeout", type=float, default=8.0)
p.add_argument("--lb_radius", type=float, default=6.0)
p.add_argument("--lb_progress", choices=["goal", "command"], default="command",
               help="trap detector progress: toward the goal, or along the commanded direction (eval_planner default)")
args, _unknown = p.parse_known_args()

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": args.headless})

# ----------------------------------------------------------------------------- imports needing Isaac
import carb  # noqa: E402
import numpy as np  # noqa: E402
import omni.timeline  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from omni.isaac.core.world import World  # noqa: E402
from pxr import Gf, UsdGeom  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from pegasus.simulator.logic.backends import Backend  # noqa: E402
from pegasus.simulator.logic.dynamics import LinearDrag  # noqa: E402
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface  # noqa: E402
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig  # noqa: E402
from pegasus.simulator.params import ROBOTS  # noqa: E402

try:
    from isaacsim.sensors.camera import Camera
except ImportError:  # older module name
    from omni.isaac.sensor import Camera

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, args.repo_root or os.path.dirname(HERE))
sys.path.insert(0, os.getcwd())
from model import Model  # noqa: E402
import isaac_scenes  # noqa: E402

G = 9.80665
VEHICLE = "/World/quadrotor"
OBST_ROOT = "/World/Obstacles"
CAM_PATH = f"{VEHICLE}/body/ttc_camera"
VIEW_CAM = "/OmniverseKit_Persp"   # Isaac's default viewport camera


# ============================================================================= controller backend
class AccelYawBackend(Backend):
    """Tracks a world-frame net acceleration (what the policy commands, gravity included)
    and a yaw angle, with a geometric attitude controller (same structure as Pegasus'
    NonlinearController, minus the position loop)."""

    def __init__(self, mass, kr, kw):
        super().__init__(None)
        self.m = mass
        self.Kr = np.diag([kr, kr, kr])
        self.Kw = np.diag([kw, kw, kw])
        self.a_cmd = np.zeros(3)
        self.yaw_cmd = 0.0
        self.input_ref = [0.0] * 4
        self.R = np.eye(3)
        self.w = np.zeros(3)
        self.have_state = False

    def set_command(self, a_cmd, yaw):
        self.a_cmd = np.asarray(a_cmd, dtype=np.float64)
        self.yaw_cmd = float(yaw)

    def update_state(self, state):
        self.R = Rotation.from_quat(state.attitude).as_matrix()
        self.w = np.asarray(state.angular_velocity)
        self.have_state = True

    def input_reference(self):
        return self.input_ref

    def update(self, dt):
        if not self.have_state or self.vehicle is None:
            return
        f = self.m * (self.a_cmd + np.array([0.0, 0.0, G]))
        f[2] = max(f[2], 0.1 * self.m * G)            # never command upside-down thrust
        z_b = self.R[:, 2]
        u1 = max(float(f @ z_b), 0.0)
        z_des = f / np.linalg.norm(f)
        x_c = np.array([math.cos(self.yaw_cmd), math.sin(self.yaw_cmd), 0.0])
        y_des = np.cross(z_des, x_c)
        y_des /= max(np.linalg.norm(y_des), 1e-6)
        x_des = np.cross(y_des, z_des)
        R_des = np.c_[x_des, y_des, z_des]
        S = R_des.T @ self.R - self.R.T @ R_des
        e_R = 0.5 * np.array([-S[1, 2], S[0, 2], -S[0, 1]])
        tau = -(self.Kr @ e_R) - (self.Kw @ self.w)
        self.input_ref = self.vehicle.force_and_torques_to_velocities(u1, tau)

    def update_sensor(self, sensor_type, data):
        pass

    def update_graphical_sensor(self, sensor_type, data):
        pass

    def start(self):
        pass

    def stop(self):
        pass

    def reset(self):
        self.a_cmd = np.zeros(3)


# ============================================================================= scene building
def build_obstacles(stage, scene, ground):
    if stage.GetPrimAtPath(OBST_ROOT).IsValid():
        stage.RemovePrim(OBST_ROOT)
    UsdGeom.Xform.Define(stage, OBST_ROOT)
    n = 0

    def xform(prim, t, s=(1, 1, 1)):
        api = UsdGeom.XformCommonAPI(prim)
        api.SetTranslate(Gf.Vec3d(*map(float, t)))
        api.SetScale(Gf.Vec3f(*map(float, s)))

    for b in scene["balls"]:
        g = UsdGeom.Sphere.Define(stage, f"{OBST_ROOT}/ball_{n}"); n += 1
        g.GetRadiusAttr().Set(float(b[3]))
        xform(g, b[:3])
    for v in scene["voxels"]:
        g = UsdGeom.Cube.Define(stage, f"{OBST_ROOT}/box_{n}"); n += 1
        g.GetSizeAttr().Set(2.0)
        xform(g, v[:3], v[3:6])
    for c in scene["cyl"]:            # vertical, effectively infinite
        g = UsdGeom.Cylinder.Define(stage, f"{OBST_ROOT}/cyl_{n}"); n += 1
        g.GetRadiusAttr().Set(float(c[2])); g.GetHeightAttr().Set(60.0); g.GetAxisAttr().Set("Z")
        xform(g, (c[0], c[1], 20.0))
    for c in scene["cyl_h"]:          # horizontal along y
        g = UsdGeom.Cylinder.Define(stage, f"{OBST_ROOT}/cylh_{n}"); n += 1
        g.GetRadiusAttr().Set(float(c[2])); g.GetHeightAttr().Set(100.0); g.GetAxisAttr().Set("Y")
        xform(g, (c[0], 0.0, c[1]))
    return n


def setup_camera(stage):
    path = CAM_PATH
    cam = Camera(prim_path=path, resolution=(args.img_w, args.img_h))
    prim = stage.GetPrimAtPath(path)
    h_ap = 20.955
    prim.GetAttribute("horizontalAperture").Set(h_ap)
    prim.GetAttribute("verticalAperture").Set(h_ap * args.img_h / args.img_w)   # square pixels
    prim.GetAttribute("focalLength").Set(h_ap / (2.0 * args.fov_x_half_tan))
    prim.GetAttribute("clippingRange").Set(Gf.Vec2f(0.05, 1000.0))
    return cam


def mount_camera(cam):
    # world camera axes: +x forward, +z up. Rotation about body y by -angle tilts the view up.
    q = Rotation.from_euler("y", -args.cam_angle, degrees=True).as_quat()      # xyzw
    cam.set_local_pose(translation=np.array([0.1, 0.0, 0.0]),
                       orientation=np.array([q[3], q[0], q[1], q[2]]), camera_axes="world")


def read_depth(cam):
    d = None
    try:
        d = cam.get_depth()
    except Exception:
        pass
    if d is None or np.size(d) == 0:
        d = cam.get_current_frame().get("distance_to_image_plane")
    if d is None or np.size(d) == 0:
        return None
    d = np.asarray(d, dtype=np.float32).reshape(args.img_h, args.img_w)
    d = np.where(np.isfinite(d) & (d > 0), d, 1000.0)
    return d


# ============================================================================= viewport
def setup_view():
    """Show the drone's own camera (fpv) in the main viewport, or leave it for the chase cam."""
    if args.headless or args.view != "fpv":
        return
    from omni.kit.viewport.utility import get_active_viewport
    vp = get_active_viewport()
    if vp is None:
        carb.log_warn("no active viewport; --view fpv ignored")
        return
    try:
        vp.camera_path = CAM_PATH
    except Exception:
        vp.set_active_camera(CAM_PATH)
    carb.log_warn(f"viewport now looking through {CAM_PATH}")


def update_chase(p, Rb):
    """Chase cam: 4 m behind, 1.5 m above, looking ahead along the level heading (no roll)."""
    if args.headless or args.view != "chase":
        return
    try:
        from isaacsim.core.utils.viewports import set_camera_view
    except ImportError:
        from omni.isaac.core.utils.viewports import set_camera_view
    fwd = level_frame(Rb)[:, 0]
    eye = p - 4.0 * fwd + np.array([0.0, 0.0, 1.5])
    set_camera_view(eye=eye, target=p + 3.0 * fwd, camera_prim_path=VIEW_CAM)


# ============================================================================= Layer B glue
def make_layer_b(device):
    from layer_b import LayerB
    return LayerB(1, device,
                  detector_kw=dict(window=args.lb_window, min_progress=args.lb_min_progress,
                                   progress=args.lb_progress),
                  escape_kw=dict(speed_frac=args.lb_speed_frac, exit_dist=args.lb_exit_dist,
                                 timeout=args.lb_timeout),
                  avoider_kw=dict(radius=args.lb_radius))


_R_MOUNT = Rotation.from_euler("y", -args.cam_angle, degrees=True).as_matrix()   # camera FLU -> body FLU


def camera_rays(depth, Rb, pool=4, max_depth=90.0):
    """Min-pooled depth -> world-frame rays (1, P, 3) and hit distances (1, P), as in eval_planner.
    Rays use the training camera model: direction (1, -fv, -fu) in camera forward-left-up."""
    H, W = depth.shape
    d = depth.reshape(H // pool, pool, W // pool, pool).min(axis=(1, 3))
    h, w = d.shape
    fov_y = args.fov_x_half_tan / W * H
    fu = (2 * (np.arange(h) + 0.5) / h - 1) * fov_y
    fv = (2 * (np.arange(w) + 0.5) / w - 1) * args.fov_x_half_tan
    FU, FV = np.meshgrid(fu, fv, indexing="ij")
    rays_cam = np.stack([np.ones_like(FU), -FV, -FU], -1).reshape(-1, 3)
    rays_w = rays_cam @ (Rb @ _R_MOUNT).T
    dist = d.reshape(-1) * np.linalg.norm(rays_cam, axis=1)            # planar depth -> range
    dist = np.where(d.reshape(-1) < max_depth, dist, np.inf)           # no hit = open
    return rays_w, dist


# ============================================================================= policy glue
def level_frame(Rb):
    fwd = Rb[:, 0].copy()
    fwd[2] = 0.0
    fwd /= max(np.linalg.norm(fwd), 1e-6)
    up = np.array([0.0, 0.0, 1.0])
    return np.stack([fwd, np.cross(up, fwd), up], 1)        # columns: fwd, left, up


class Policy:
    def __init__(self, device):
        self.device = device
        dim_obs = 7 + (0 if args.no_odom else 3)
        dim_act = 9 if args.active_yaw else 6
        self.model = Model(dim_obs, dim_act, 1).to(device)
        sd = torch.load(args.resume, map_location=device)
        missing, unexpected = self.model.load_state_dict(sd, False)
        if missing or unexpected:
            print("!! checkpoint mismatch  missing:", missing, " unexpected:", unexpected)
        self.model.eval()

    def reset(self):
        if hasattr(self.model, "reset"):
            self.model.reset()
        self.h = None

    @torch.no_grad()
    def act(self, depth, v, Rb, target_v, margin):
        R = level_frame(Rb)
        dev = self.device
        local_v = R.T @ v
        radial_v = max(local_v[0], 1e-3)
        ttc = torch.from_numpy(depth).to(dev)[None] / radial_v
        x = 3 / ttc.clamp(0.3, 10.0) - 0.6 + torch.randn_like(ttc) * 0.02
        x = F.max_pool2d(x[:, None], 4, 4)
        state = [R.T @ target_v, Rb[2, :], [margin]]
        if not args.no_odom:
            state.insert(0, local_v)
        state = torch.tensor(np.concatenate(state), dtype=torch.float32, device=dev)[None]
        act, _, self.h = self.model(x.float(), state, self.h)
        Rt = torch.tensor(R, dtype=torch.float32, device=dev)[None]
        vecs = (Rt @ act.reshape(1, 3, -1)).unbind(-1)
        a_cmd = (vecs[0] - vecs[1])[0].cpu().numpy().astype(np.float64)
        return a_cmd, x


# ============================================================================= episode
def run_episode(world, vehicle, backend, cam, policy, scene, idx, T):
    stage = world.stage
    n_obs = build_obstacles(stage, scene, not args.no_ground)
    p0, goal = scene["p0"], scene["goal"]
    yaw0 = math.atan2(goal[1] - p0[1], goal[0] - p0[0])
    q = Rotation.from_euler("z", yaw0).as_quat()
    q_wxyz = np.array([q[3], q[0], q[1], q[2]])

    # Hold the drone at the start while the controller and renderer settle
    backend.set_command(np.zeros(3), yaw0)
    for _ in range(int(0.5 * args.physics_hz)):
        vehicle.set_world_pose(position=p0, orientation=q_wxyz)
        vehicle.set_linear_velocity(np.zeros(3))
        vehicle.set_angular_velocity(np.zeros(3))
        world.step(render=False)
    for _ in range(3):
        world.step(render=True)

    policy.reset()
    lb = getattr(policy, "lb", None)
    if lb is not None:
        lb.reset()
    lb_desired = None
    lb_log = {"escaping": [], "avoiding": []}
    sub = args.physics_hz // args.ctl_hz
    act_q = deque([np.zeros(3)] * (args.act_lag + 1))
    depth_q = deque(maxlen=args.depth_lag + 1)
    log = {"p": [], "v": [], "a_cmd": [], "t": []}
    crashed = reached = False
    min_clear = np.inf
    t_reach = None
    s = vehicle.state
    for k in range(T):
        d = read_depth(cam)
        if d is None:
            d = depth_q[-1] if depth_q else np.full((args.img_h, args.img_w), 1000.0, np.float32)
        depth_q.append(d)

        # heading follows the goal direction, as in training (yaw drift off)
        to_goal = goal - s.position
        head = lb_desired if lb_desired is not None else to_goal     # camera follows Layer B
        a_cmd = act_q.popleft()
        backend.set_command(a_cmd, math.atan2(head[1], head[0]))
        for j in range(sub):
            world.step(render=(j == sub - 1))
            dist = isaac_scenes.nearest_distance(scene, vehicle.state.position, not args.no_ground)
            min_clear = min(min_clear, dist - scene["margin"])
            if dist - scene["margin"] < 0:
                crashed = True
                break
        s = vehicle.state
        p, v = s.position.copy(), s.linear_velocity.copy()
        log["p"].append(p); log["v"].append(v); log["a_cmd"].append(a_cmd); log["t"].append(k / args.ctl_hz)
        if crashed:
            break
        if np.linalg.norm(goal - p) <= args.goal_radius:
            reached, t_reach = True, k / args.ctl_hz
            break

        target_raw = goal - p
        n = np.linalg.norm(target_raw)
        target_v = target_raw / max(n, 1e-6) * min(n, scene["max_speed"])
        Rb = Rotation.from_quat(s.attitude).as_matrix()
        if lb is not None:
            if k % args.plan_every == 0:
                dev = policy.device
                tt = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32, device=dev)[None]
                rays_w, dist = camera_rays(d, Rb)
                out, info = lb.step((k + 1) / args.ctl_hz, tt(p), tt(goal), tt(target_v),
                                    tt(rays_w), tt(dist), None)
                lb_desired = out[0].cpu().numpy().astype(np.float64)
                lb_log["escaping"].append(bool(info["escaping"][0]))
                lb_log["avoiding"].append(bool(info["avoiding"][0]))
                if lb_log["escaping"][-1] and (len(lb_log["escaping"]) == 1 or not lb_log["escaping"][-2]):
                    print(f"    t={k / args.ctl_hz:5.1f}s  trap detected -> escaping")
                if lb_log["avoiding"][-1] and (len(lb_log["avoiding"]) == 1 or not lb_log["avoiding"][-2]):
                    print(f"    t={k / args.ctl_hz:5.1f}s  escape done -> going around")
            target_v = lb_desired
        update_chase(p, Rb)
        a_next, _ = policy.act(depth_q[0], v, Rb, target_v, scene["margin"])   # depth_q[0] = rendered before this step
        act_q.append(a_next)

    P = np.array(log["p"]); V = np.array(log["v"])
    speed = np.linalg.norm(V, axis=1) if len(V) else np.zeros(1)
    res = {
        "episode": idx, "obstacles": n_obs,
        "crashed": crashed, "reached": reached,
        "success": not crashed,
        "time_to_goal": t_reach,
        "avg_speed": float(speed.mean()), "max_speed": float(speed.max()),
        "max_speed_cmd": scene["max_speed"],
        "final_goal_dist": float(np.linalg.norm(goal - P[-1])) if len(P) else None,
        "closest_goal_dist": float(np.min(np.linalg.norm(P - goal, axis=1))) if len(P) else None,
        "min_clearance": float(min_clear),
        "duration": len(P) / args.ctl_hz,
    }
    if lb is not None:
        e = lb_log["escaping"]
        res["escapes"] = int(sum(1 for i, x in enumerate(e) if x and (i == 0 or not e[i - 1])))
        res["escaping_frac"] = float(np.mean(e)) if e else 0.0
        res["avoiding_frac"] = float(np.mean(lb_log["avoiding"])) if e else 0.0
    if scene.get("trap_x") is not None:
        res["cleared_trap"] = bool(not crashed and len(P) and P[:, 0].max() > scene["trap_x"])
    os.makedirs(args.out, exist_ok=True)
    np.savez(os.path.join(args.out, f"ep{idx:03d}.npz"), goal=goal, p0=p0,
             **{k: np.array(v) for k, v in log.items()})
    return res


# ============================================================================= main
def main():
    T = args.timesteps or (600 if args.scene == "utrap" else 150)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    if args.scene == "exported":
        scenes = isaac_scenes.load_exported(args.scenes_file)
        scenes = scenes[args.first_episode:args.first_episode + args.episodes]
    elif args.scene == "utrap":
        scenes = [isaac_scenes.make_utrap(rng, args.alt, args.max_speed, args.margin)
                  for _ in range(args.episodes)]
    else:
        scenes = [isaac_scenes.make_forest(rng, args.alt, max_speed=args.max_speed, margin=args.margin)
                  for _ in range(args.episodes)]

    timeline = omni.timeline.get_timeline_interface()
    pg = PegasusInterface()
    settings = dict(pg._world_settings)
    settings.update(physics_dt=1.0 / args.physics_hz, rendering_dt=1.0 / args.physics_hz,
                    stage_units_in_meters=1.0)
    pg._world = World(**settings)
    world = pg.world
    if not args.no_ground:
        world.scene.add_default_ground_plane()

    backend = AccelYawBackend(args.mass, args.kr, args.kw)
    cfg = MultirotorConfig()
    cfg.backends = [backend]
    if args.no_drag:
        cfg.drag = LinearDrag([0.0, 0.0, 0.0])
    p0 = scenes[0]["p0"]
    vehicle = Multirotor(VEHICLE, ROBOTS["Iris"], 0, list(map(float, p0)),
                         [0.0, 0.0, 0.0, 1.0], config=cfg)
    cam = setup_camera(world.stage)
    world.reset()
    cam.initialize()
    cam.add_distance_to_image_plane_to_frame()
    mount_camera(cam)
    setup_view()
    timeline.play()
    for _ in range(10):
        world.step(render=True)

    policy = Policy(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    policy.lb = make_layer_b(policy.device) if args.layer_b else None
    print("Layer B:", "ON" if args.layer_b else "off (reactive policy only: expect it to get stuck in the U-trap)")
    results = []
    for i, scene in enumerate(scenes):
        if not simulation_app.is_running():
            break
        t0 = time.time()
        r = run_episode(world, vehicle, backend, cam, policy, scene, args.first_episode + i, T)
        results.append(r)
        tag = "CRASH" if r["crashed"] else ("GOAL" if r["reached"] else "timeout")
        extra = f" cleared={r['cleared_trap']}" if "cleared_trap" in r else ""
        if "escapes" in r:
            extra += f" escapes={r['escapes']}"
        print(f"[ep {r['episode']:3d}] {tag:7s} avg {r['avg_speed']:.2f} m/s  max {r['max_speed']:.2f}  "
              f"closest-to-goal {r['closest_goal_dist']:.2f} m  min-clear {r['min_clearance']:.2f} m"
              f"{extra}  ({time.time() - t0:.0f}s wall)")

    if results:
        def frac(k):
            return float(np.mean([bool(r.get(k)) for r in results]))
        summary = {"scene": args.scene, "episodes": len(results),
                   "success (no crash)": frac("success"), "reached": frac("reached"),
                   "crashed": frac("crashed"),
                   "avg_speed": float(np.mean([r["avg_speed"] for r in results]))}
        summary["layer_b"] = args.layer_b
        if "cleared_trap" in results[0]:
            summary["cleared_trap"] = frac("cleared_trap")
        print("\n" + json.dumps(summary, indent=2))
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "summary.json"), "w") as f:
            json.dump({"args": vars(args), "summary": summary, "episodes": results}, f, indent=2)

    carb.log_warn("Isaac policy eval finished.")
    timeline.stop()
    simulation_app.close()


if __name__ == "__main__":
    main()