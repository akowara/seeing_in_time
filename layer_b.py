"""
Layer B, step 4 (trap avoider) and the LayerB wrapper that runs all four parts.

TrapAvoider ("no going back"):
  After an escape, keep a circle of radius R around where the drone got stuck.
  While active, if the goal direction points into that circle, turn it to the
  circle's tangent edge on a fixed side (chosen once, at activation). The closer the
  drone gets, the wider the turn, so it curves around the trap. Released once the
  drone is closer to the goal than the trap is (by release_margin), or on timeout.

LayerB:
  Each step: update memory -> update trap detector -> apply trap avoidance to the
  goal velocity -> let the escape controller override it while escaping. The output
  replaces the goal velocity given to Layer A and the policy.

Batched over B drones; world frame, z up. Distances in metres.
"""

import math

import torch

from heading_memory import HeadingMemory, wrap
from trap_detector import TrapDetector
from escape_controller import EscapeController


class TrapAvoider:
    def __init__(self, batch_size, device, radius=6.0, release_margin=2.0, timeout=30.0):
        self.B, self.device = batch_size, device
        self.radius = radius                  # keep-out radius around the trap (m)
        self.release_margin = release_margin  # how far past the trap before releasing (m)
        self.timeout = timeout                # give up avoiding after this long (s)
        self.reset()

    def reset(self):
        B, dev = self.B, self.device
        self.active = torch.zeros(B, dtype=torch.bool, device=dev)
        self.trap_pos = torch.zeros(B, 3, device=dev)
        self.side = torch.ones(B, device=dev)             # +1: pass counter-clockwise, -1: clockwise
        self.start_t = torch.full((B,), float('nan'), device=dev)

    def _geometry(self, pos):
        to_trap = (self.trap_pos - pos)[:, :2]
        d = to_trap.norm(dim=-1).clamp_min(1e-6)
        trap_yaw = torch.atan2(to_trap[:, 1], to_trap[:, 0])
        half = torch.asin((self.radius / d).clamp(max=1.0))   # tangent half-angle; 90 deg inside
        return trap_yaw, half

    @torch.no_grad()
    def activate(self, mask, t, trap_pos, pos, goal, memory=None):
        """Start avoiding trap_pos for these drones, and pick which side to pass on."""
        self.active |= mask
        self.trap_pos = torch.where(mask[:, None], trap_pos, self.trap_pos)
        self.start_t = torch.where(mask, torch.full_like(self.start_t, t), self.start_t)

        trap_yaw, half = self._geometry(pos)
        yaw_p, yaw_m = trap_yaw + half, trap_yaw - half
        if memory is not None:
            free_p = torch.nan_to_num(memory.free_toward(wrap(yaw_p)), nan=0.0)
            free_m = torch.nan_to_num(memory.free_toward(wrap(yaw_m)), nan=0.0)
        else:
            free_p = free_m = torch.zeros(self.B, device=self.device)
        g = (goal - pos)[:, :2]
        goal_yaw = torch.atan2(g[:, 1], g[:, 0])
        close_p = wrap(yaw_p - goal_yaw).abs()
        close_m = wrap(yaw_m - goal_yaw).abs()
        # more open side wins; on a tie, the side closer to the goal; then +1
        pick_minus = (free_m > free_p) | ((free_m == free_p) & (close_m < close_p))
        side = 1.0 - 2.0 * pick_minus.float()
        self.side = torch.where(mask, side, self.side)

    @torch.no_grad()
    def step(self, t, pos, goal, goal_vel):
        """Returns the (possibly turned) goal velocity (B, 3) and which drones are avoiding (B,)."""
        # Release: past the trap, or timed out
        d_goal = (goal - pos)[:, :2].norm(dim=-1)
        d_trap_goal = (goal - self.trap_pos)[:, :2].norm(dim=-1)
        past = d_goal < d_trap_goal - self.release_margin
        expired = (t - self.start_t) >= self.timeout
        self.active &= ~(past | expired)

        trap_yaw, half = self._geometry(pos)
        g_yaw = torch.atan2(goal_vel[:, 1], goal_vel[:, 0])
        in_cone = self.active & (wrap(g_yaw - trap_yaw).abs() < half)

        new_yaw = trap_yaw + self.side * half
        speed = goal_vel[:, :2].norm(dim=-1)
        turned = torch.stack([torch.cos(new_yaw) * speed, torch.sin(new_yaw) * speed,
                              goal_vel[:, 2]], -1)
        return torch.where(in_cone[:, None], turned, goal_vel), self.active.clone()


class LayerB:
    """Memory + trap detector + escape + trap avoidance, in the right order."""

    def __init__(self, batch_size, device, memory_kw=None, detector_kw=None,
                 escape_kw=None, avoider_kw=None):
        self.memory = HeadingMemory(batch_size, device, **(memory_kw or {}))
        self.detector = TrapDetector(batch_size, device, **(detector_kw or {}))
        self.escape = EscapeController(batch_size, device, **(escape_kw or {}))
        self.avoider = TrapAvoider(batch_size, device, **(avoider_kw or {}))
        self.reset()

    def reset(self):
        self.memory.reset()
        self.detector.reset()
        self.escape.reset()
        self.avoider.reset()
        self.last_desired = None
        # traps that started while going around an earlier trap (re-traps)
        self.n_retrap_avoiding = torch.zeros_like(self.escape.n_escapes)

    @torch.no_grad()
    def step(self, t, pos, goal, goal_vel, rays_world=None, dist=None, all_blocked=None):
        """
        t:           current time (s)
        pos, goal:   (B, 3) positions, world frame
        goal_vel:    (B, 3) normal desired velocity toward the goal
        rays_world, dist: camera rays (B, P, 3) and hit distances (B, P), optional
        all_blocked: optional (B,) bool from Layer A
        Returns the desired velocity (B, 3) to hand to Layer A / the policy, and a dict
        with 'escaping' and 'avoiding' masks.
        """
        self.memory.update(t, pos, rays_world, dist)
        cmd = self.last_desired if self.last_desired is not None else goal_vel
        new_trap = self.detector.update(t, pos, goal, all_blocked, cmd_vel=cmd)
        self.n_retrap_avoiding += (new_trap & self.avoider.active).long()

        was_escaping = self.escape.escaping.clone()
        avoid_v, _ = self.avoider.step(t, pos, goal, goal_vel)
        desired, escaping = self.escape.step(t, pos, avoid_v, self.memory, self.detector)

        finished = was_escaping & ~escaping
        if finished.any():
            # escape just ended: start avoiding that trap, and apply it right away
            self.avoider.activate(finished, t, self.escape.last_trap_pos, pos, goal, self.memory)
            avoid_v, _ = self.avoider.step(t, pos, goal, goal_vel)
            desired = torch.where(finished[:, None], avoid_v, desired)

        avoiding = self.avoider.active & ~escaping
        self.last_desired = desired.clone()
        return desired, {'escaping': escaping, 'avoiding': avoiding}