"""
Layer B, step 3: escape controller.

When the trap detector flags a drone, replace its goal direction with an escape
direction until it's clear:

  1. Pick the most open remembered heading (HeadingMemory.best_heading), excluding
     +- avoid_halfwidth around the blocked goal direction. If memory has nothing
     usable, go straight back the way it came (blocked direction + 180 deg).
  2. Fly it at speed_frac of the normal target speed (heading follows the command,
     so the camera looks where the drone is going and memory fills in).
  3. If memory now shows the chosen heading is blocked (< min_free m), re-pick,
     at most every repick_s seconds.
  4. Stop escaping after moving exit_dist metres from where the escape started, or
     after timeout seconds. Then clear the detector and remember the trap (step 4
     uses it to avoid flying straight back in).

The controller doesn't fly the drone itself: it only changes the goal velocity that
Layer A and the policy are given, so obstacle avoidance keeps working during escape.
Batched over B drones; world frame, z up.
"""

import math

import torch

from heading_memory import wrap


class EscapeController:
    def __init__(self, batch_size, device, speed_frac=0.5, avoid_halfwidth_deg=60.0,
                 min_free=2.0, exit_dist=8.0, timeout=8.0, repick_s=0.5):
        self.B, self.device = batch_size, device
        self.speed_frac = speed_frac
        self.avoid_hw = avoid_halfwidth_deg
        self.min_free = min_free
        self.exit_dist = exit_dist
        self.timeout = timeout
        self.repick_s = repick_s
        self.reset()

    def reset(self):
        """Call at the start of every episode."""
        B, dev = self.B, self.device
        self.escaping = torch.zeros(B, dtype=torch.bool, device=dev)
        self.esc_yaw = torch.zeros(B, device=dev)
        self.start_t = torch.full((B,), float('nan'), device=dev)
        self.start_pos = torch.zeros(B, 3, device=dev)
        self.last_pick = torch.full((B,), -float('inf'), device=dev)
        self.n_escapes = torch.zeros(B, dtype=torch.long, device=dev)
        self.n_repicks = torch.zeros(B, dtype=torch.long, device=dev)
        # The most recent trap each drone escaped from (used by step 4)
        self.has_last_trap = torch.zeros(B, dtype=torch.bool, device=dev)
        self.last_trap_pos = torch.zeros(B, 3, device=dev)
        self.last_trap_yaw = torch.zeros(B, device=dev)
        self.last_exit_t = torch.full((B,), float('nan'), device=dev)

    def _pick(self, t, memory, avoid_yaw, mask):
        yaw, free = memory.best_heading(t, avoid_yaw=avoid_yaw, avoid_halfwidth_deg=self.avoid_hw)
        yaw = torch.where(free > 0, yaw, wrap(avoid_yaw + math.pi))   # nothing usable: go back
        self.esc_yaw = torch.where(mask, yaw, self.esc_yaw)
        self.last_pick = torch.where(mask, torch.full_like(self.last_pick, t), self.last_pick)

    @torch.no_grad()
    def step(self, t, pos, goal_vel, memory, detector):
        """
        t:         current time (s), a float
        pos:       (B, 3) drone positions
        goal_vel:  (B, 3) the normal desired velocity (toward the goal)
        memory:    HeadingMemory, already updated this step
        detector:  TrapDetector, already updated this step
        Returns the desired velocity to use (B, 3) and which drones are escaping (B,).
        """
        # 1. Start escaping when newly trapped
        start = detector.trapped & ~self.escaping
        if start.any():
            self.escaping |= start
            self.start_t = torch.where(start, torch.full_like(self.start_t, t), self.start_t)
            self.start_pos = torch.where(start[:, None], pos, self.start_pos)
            self.n_escapes += start.long()
            self._pick(t, memory, detector.trap_yaw, start)

        # 2. Re-pick if the chosen heading turned out to be blocked
        if self.escaping.any():
            f = memory.free_toward(self.esc_yaw)          # nan if never seen: not "blocked"
            blocked = self.escaping & (f < self.min_free) & (t - self.last_pick >= self.repick_s)
            if blocked.any():
                self.n_repicks += blocked.long()
                self._pick(t, memory, detector.trap_yaw, blocked)

        # 3. Finish: far enough from where the escape started, or out of time
        moved = (pos - self.start_pos)[:, :2].norm(dim=-1)
        done = self.escaping & ((moved >= self.exit_dist) | (t - self.start_t >= self.timeout))
        if done.any():
            self.escaping &= ~done
            self.has_last_trap |= done
            self.last_trap_pos = torch.where(done[:, None], detector.trap_pos, self.last_trap_pos)
            self.last_trap_yaw = torch.where(done, detector.trap_yaw, self.last_trap_yaw)
            self.last_exit_t = torch.where(done, torch.full_like(self.last_exit_t, t), self.last_exit_t)
            detector.clear(done, t)

        # 4. Desired velocity: escape direction (level) at reduced speed, else unchanged
        speed = goal_vel.norm(dim=-1, keepdim=True) * self.speed_frac
        d = torch.stack([torch.cos(self.esc_yaw), torch.sin(self.esc_yaw),
                         torch.zeros_like(self.esc_yaw)], -1)
        desired = torch.where(self.escaping[:, None], d * speed, goal_vel)
        return desired, self.escaping.clone()