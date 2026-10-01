"""
Layer B, step 2: trap detector.

Flags a drone as trapped when either
  1. progress stalls: over the last `window` seconds the distance to the goal
     shrank by less than `min_progress` metres (and the drone isn't near the goal), or
  2. everything is blocked: Layer A reports no clear candidate for `blocked_s` seconds.

Once flagged, a drone stays trapped until clear() is called (the escape decides when
it's over). On detection it records where the trap was and which way the goal was
(the blocked direction). Batched over B drones; world frame, z up.
"""

import math
from collections import deque

import torch


class TrapDetector:
    def __init__(self, batch_size, device, window=2.0, min_progress=1.0,
                 near_goal=3.0, blocked_s=1.0, progress='goal'):
        self.B, self.device = batch_size, device
        self.window = window              # seconds of history used to judge progress
        self.min_progress = min_progress  # metres of progress expected per window
        self.near_goal = near_goal        # never trapped this close to the goal (m)
        self.blocked_s = blocked_s        # all-blocked this long counts as trapped (s)
        # 'goal':    progress = reduction in distance to the goal
        # 'command': progress = distance moved along the commanded direction, so a
        #            deliberate detour (e.g. going around a trap) doesn't look stuck
        assert progress in ('goal', 'command')
        self.progress_mode = progress
        self.reset()

    def reset(self):
        """Call at the start of every episode."""
        B, dev = self.B, self.device
        self.history = deque()                                    # (time, progress coordinate (B,))
        self.along = torch.zeros(B, device=dev)                   # metres moved along commands
        self.prev_pos = None
        self.start_t = torch.full((B,), -float('inf'), device=dev)  # history valid from here
        self.started = False
        self.last_t = None
        self.blocked_time = torch.zeros(B, device=dev)
        self.prev_blocked = torch.zeros(B, dtype=torch.bool, device=dev)
        self.trapped = torch.zeros(B, dtype=torch.bool, device=dev)
        self.trap_pos = torch.zeros(B, 3, device=dev)            # where it was detected
        self.trap_yaw = torch.zeros(B, device=dev)                # goal direction then (the blocked way)
        self.trap_time = torch.full((B,), float('nan'), device=dev)
        self.reason = torch.zeros(B, dtype=torch.long, device=dev)  # 1 = stalled, 2 = all blocked

    @torch.no_grad()
    def update(self, t, pos, goal, all_blocked=None, cmd_vel=None):
        """
        t:           current time (s), a float
        pos, goal:   (B, 3) drone and goal positions, world frame
        all_blocked: optional (B,) bool, Layer A found no clear candidate this plan
        cmd_vel:     optional (B, 3) the command that was active since the last call
                     (needed for progress='command')
        Returns (B,) bool: drones that became trapped on this call.
        """
        if not self.started:
            self.start_t = torch.full_like(self.start_t, t)
            self.started = True
        dt = 0.0 if self.last_t is None else t - self.last_t
        self.last_t = t

        to_goal = goal - pos
        dist = to_goal.norm(dim=-1)

        # Distance moved along the command since the last call
        if self.prev_pos is not None and cmd_vel is not None:
            u = cmd_vel[:, :2]
            u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            self.along = self.along + ((pos - self.prev_pos)[:, :2] * u).sum(-1)
        self.prev_pos = pos.clone()

        # 1. Progress over the last `window` seconds (a coordinate that grows with progress)
        if self.progress_mode == 'command' and cmd_vel is not None:
            coord = self.along
        else:
            coord = -dist
        self.history.append((t, coord.clone()))
        while len(self.history) > 1 and self.history[1][0] <= t - self.window:
            self.history.popleft()
        t_then, c_then = self.history[0]
        enough_history = (t - t_then >= self.window - 1e-6) & (t - self.start_t >= self.window - 1e-6)
        stalled = enough_history & (coord - c_then < self.min_progress) & (dist > self.near_goal)

        # 2. Continuously all-blocked
        if all_blocked is not None:
            # time since blocking started: counts only while blocked on consecutive calls
            self.blocked_time = torch.where(all_blocked & self.prev_blocked, self.blocked_time + dt,
                                            torch.zeros_like(self.blocked_time))
            self.prev_blocked = all_blocked.clone()
        blocked = self.blocked_time >= self.blocked_s - 1e-6

        new = (stalled | blocked) & ~self.trapped
        if new.any():
            self.trapped |= new
            m = new[:, None]
            self.trap_pos = torch.where(m, pos, self.trap_pos)
            self.trap_yaw = torch.where(new, torch.atan2(to_goal[:, 1], to_goal[:, 0]), self.trap_yaw)
            self.trap_time = torch.where(new, torch.full_like(self.trap_time, t), self.trap_time)
            self.reason = torch.where(new, 2 - stalled.long(), self.reason)   # 1 stalled, 2 blocked
        return new

    def clear(self, mask, t):
        """Escape finished for these drones: un-trap them and require fresh history."""
        self.trapped &= ~mask
        self.start_t = torch.where(mask, torch.full_like(self.start_t, t), self.start_t)
        self.blocked_time = torch.where(mask, torch.zeros_like(self.blocked_time), self.blocked_time)
        self.prev_blocked &= ~mask