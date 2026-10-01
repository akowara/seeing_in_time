"""
Bridge between the training sim and the TTC planner (Layer A), batched over drones.

Planning frame (per drone, rebuilt every plan):
    gravity-leveled, x right, y down, z = horizontal direction of current flight.
So "straight ahead" means level flight along the current heading, and the
candidate cone's pitch is measured from the horizon, independent of camera
tilt (cam_angle) or body pitch.

Points are time-normalized by the current speed |v|: q = P / |v|, in seconds.
On the real system this is TTC lifted with the FoE correction; in sim we use
the true depth, pose and velocity, which carry the same information.

Sim conventions: world z up. Camera frame is FLU with pixel ray (1, -fv, -fu);
env.R_cam_world maps camera FLU -> world.
"""

import math
import torch
import torch.nn.functional as F

from seeing_in_time.ttc_planner import make_candidates


class TTCPlannerBridge:
    def __init__(self, fov_x_half_tan, device, pool=4, ttc_max=24.0,
                 T=0.3, horizon=1.5, dt=0.1, t_min=0.5,
                 w_c=100.0, w_g=1.0, w_s=0.2, w_u=2.0,
                 lam=0.02, max_angle=math.radians(20), min_speed=0.5,
                 max_depth=90.0, gate_t=None, speed_scales=(0.5, 0.75, 1.0)):
        self.fov_x = fov_x_half_tan
        self.pool = pool
        self.ttc_max = ttc_max
        self.T = T
        self.t_min, self.w_c, self.w_g, self.w_s, self.w_u = t_min, w_c, w_g, w_s, w_u
        self.lam, self.cos_max = lam, math.cos(max_angle)
        self.min_speed = min_speed
        self.max_depth = max_depth
        # Gating: if set, only intervene when flying straight at the goal would
        # come within gate_t seconds of an obstacle. Otherwise pass the goal through.
        self.gate_t = gate_t

        # Candidates (planning frame) and their per-candidate time grids
        self.cands = make_candidates(speed_scales=tuple(speed_scales)).to(device)   # (N, 3)
        self.dirs = F.normalize(self.cands, dim=-1)
        base_t = torch.arange(dt, horizon + 1e-6, dt, device=device)  # (K,)
        self.base_t = base_t
        self.t = base_t[None, :] / self.cands.norm(dim=-1, keepdim=True)   # (N, K): same distance ahead
        self.N, self.K = self.t.shape
        self.last_headway = None
        self.reset()

    def reset(self):
        """Call at the start of every episode."""
        self.prev_world = None      # previous plan: unit direction (world) * relative speed

    def _rays_flu(self, h, w, device, dtype):
        fov_y = self.fov_x * h / w
        fv = (2 * (torch.arange(w, device=device, dtype=dtype) + 0.5) / w - 1) * self.fov_x
        fu = (2 * (torch.arange(h, device=device, dtype=dtype) + 0.5) / h - 1) * fov_y
        fu, fv = torch.meshgrid(fu, fv, indexing='ij')                 # (h, w)
        return torch.stack([torch.ones_like(fu), -fv, -fu], -1)       # (h, w, 3)

    @staticmethod
    def _planning_frame(v_world, goal_world):
        """Rows map world -> planning frame (right, down, forward-level)."""
        f = v_world.clone()
        f[:, 2] = 0
        g = goal_world.clone()
        g[:, 2] = 0
        f = torch.where(f.norm(dim=-1, keepdim=True) > 1e-3, f, g)     # hovering: face the goal
        f = F.normalize(f, dim=-1)
        up = torch.zeros_like(f)
        up[:, 2] = 1
        left = torch.cross(up, f, dim=-1)
        return torch.stack([-left, -up, f], dim=1)                     # (B, 3, 3)

    @torch.no_grad()
    def __call__(self, depth, v_world, goal_world, R_cam_world):
        """
        depth:       (B, H, W) rendered depth along the camera axis
        v_world:     (B, 3) drone velocity (env.v)
        goal_world:  (B, 3) desired velocity toward the goal (speed-capped target_v)
        R_cam_world: (B, 3, 3) env.R_cam_world

        Returns (B, 3) planned desired velocity in world frame: the planner's
        direction, at the goal's speed times the planner's relative speed.
        """
        B = depth.shape[0]
        speed = v_world.norm(dim=-1)                                   # (B,)
        sp = speed.clamp_min(1e-3)
        R_pw = self._planning_frame(v_world, goal_world)               # world -> plan

        # ---- lift: closest obstacle per pooled cell, into the planning frame
        d = -F.max_pool2d(-depth[:, None], self.pool)[:, 0]            # (B, h, w)
        h, w = d.shape[1:]
        rays_world = torch.einsum('bij,hwj->bhwi', R_cam_world,
                                  self._rays_flu(h, w, d.device, d.dtype))
        P = d[..., None] * rays_world                                  # (B, h, w, 3) world offsets
        q = torch.einsum('bij,bhwj->bhwi', R_pw, P) / sp[:, None, None, None]
        valid = (torch.isfinite(d) & (d < self.max_depth)
                 & (d / sp[:, None, None] < self.ttc_max))
        q = torch.where(valid[..., None], q, torch.full_like(q, 1e4))
        pts = q.reshape(B, -1, 3)                                      # (B, P, 3)

        # ---- rollout from each drone's actual velocity (first-order response)
        v_now = (R_pw @ v_world[..., None])[..., 0] / sp[:, None]      # (B, 3), unit length
        # Headway if the drone simply keeps its current velocity (a warning signal)
        D_cur = v_now[:, None, :] * self.base_t[None, :, None]         # (B, K, 3)
        self.last_cur_headway = torch.cdist(D_cur, pts).amin(dim=(1, 2))

        vn = v_now[:, None, None, :]                                   # (B, 1, 1, 3)
        vc = self.cands[None, :, None, :]                              # (1, N, 1, 3)
        tt = self.t[None, :, :, None]                                  # (1, N, K, 1)
        decay = torch.exp(-tt / self.T)
        D = vc * tt + (vn - vc) * self.T * (1 - decay)                 # (B, N, K, 3)
        V = vc + (vn - vc) * decay                                     # (B, N, K, 3)

        # ---- score
        dist = torch.cdist(D.reshape(B, -1, 3), pts)                   # (B, N*K, P)
        dist = dist.view(B, self.N, self.K, -1) / V.norm(dim=-1)[..., None]
        headway = dist.amin(dim=(2, 3))                                # (B, N)
        C_coll = (self.t_min - headway).clamp_min(0) ** 2

        goal_dir = F.normalize((R_pw @ goal_world[..., None])[..., 0], dim=-1)
        C_goal = 1 - goal_dir @ self.dirs.T                            # (B, N)
        # Smoothness: change in *desired* velocity since the last plan (brief's ||dv_d||),
        # not distance from the current velocity, which would make the plan lag the goal.
        if self.prev_world is None:
            prev = v_now
        else:
            prev = (R_pw @ self.prev_world[..., None])[..., 0]
        C_smooth = (self.cands[None] - prev[:, None]).norm(dim=-1)     # (B, N)

        # Unseen: fraction of each path outside the camera's current view.
        # Space the camera can't see has no points, so without this it looks free.
        D_world = torch.einsum('bji,bnkj->bnki', R_pw, D)              # plan -> world
        D_cam = torch.einsum('bji,bnkj->bnki', R_cam_world, D_world)   # world -> camera FLU
        fwd_c = D_cam[..., 0]
        fov_y = self.fov_x * h / w
        inside = ((fwd_c > 1e-6)
                  & (D_cam[..., 1].abs() < self.fov_x * fwd_c)
                  & (D_cam[..., 2].abs() < fov_y * fwd_c))
        C_unseen = (~inside).float().mean(-1)                          # (B, N)

        J = (self.w_c * C_coll + self.w_g * C_goal + self.w_s * C_smooth
             + self.w_u * C_unseen)

        # ---- select_local
        best_idx = J.argmin(1)
        best = self.dirs[best_idx]
        near = (best @ self.dirs.T) >= self.cos_max
        Jm = torch.where(near, J, torch.full_like(J, float('inf')))
        wts = torch.softmax(-(Jm - J.amin(1, keepdim=True)) / self.lam, dim=1)
        v_plan = wts @ self.cands                                      # (B, 3)

        # ---- back to world
        dir_world = (R_pw.transpose(1, 2) @ F.normalize(v_plan, dim=-1)[..., None])[..., 0]
        planned = dir_world * goal_world.norm(dim=-1, keepdim=True) * v_plan.norm(dim=-1, keepdim=True)

        self.last_headway = headway.gather(1, best_idx[:, None])[:, 0].clamp(max=10.0)
        self.last_coll_active = C_coll.gather(1, best_idx[:, None])[:, 0] > 0
        self.last_any_free = (C_coll == 0).any(dim=1)
        self.last_unseen = C_unseen.gather(1, best_idx[:, None])[:, 0]

        # Gate: is the straight-to-goal candidate (full speed) blocked?
        align = goal_dir @ self.dirs.T + 1e-3 * self.cands.norm(dim=-1)[None]
        g_idx = align.argmax(1)
        self.last_goal_headway = headway.gather(1, g_idx[:, None])[:, 0]
        if self.gate_t is not None:
            blocked = self.last_goal_headway < self.gate_t
        else:
            blocked = torch.ones(B, dtype=torch.bool, device=depth.device)

        slow = speed < self.min_speed                                  # TTC meaningless: head for goal
        use_goal = (slow | ~blocked)[:, None]
        self.last_intervened = ~use_goal[:, 0]
        out = torch.where(use_goal, goal_world, planned)

        prev_new = dir_world * v_plan.norm(dim=-1, keepdim=True)
        self.prev_world = torch.where(use_goal, F.normalize(goal_world, dim=-1), prev_new)
        return out