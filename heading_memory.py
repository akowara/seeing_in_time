"""
Layer B, step 1: heading memory.

A ring of bins over world yaw. Each bin stores the free distance (m) last known in
that direction and the time it was recorded. Two sources write to it:

  1. Camera: for every bin the camera currently covers, the distance to the closest
     obstacle (d_cap if nothing is hit). Bins outside the view keep their old values.
  2. Trail: the direction back toward where the drone was ~trail_s seconds ago is
     free by construction (it just flew through it), at least as far as that point.

World frame: z up, yaw = atan2(y, x). Batched over B drones.
"""

import math
from collections import deque

import torch


def wrap(a):
    """Wrap angles to (-pi, pi]."""
    return torch.atan2(torch.sin(a), torch.cos(a))


class HeadingMemory:
    def __init__(self, batch_size, device, n_bins=36, elev_band_deg=10.0, d_cap=30.0,
                 trail_s=3.0, min_trail=0.5):
        self.B, self.n, self.device = batch_size, n_bins, device
        self.band = math.radians(elev_band_deg)   # only rays this close to level count
        self.d_cap = d_cap                         # "nothing hit" is stored as this distance
        self.trail_s = trail_s                     # how far back the trail looks (s)
        self.min_trail = min_trail                 # ignore trails shorter than this (m)
        # centre yaw of each bin: -180, -170, ..., 170 deg, so 0 / +-90 / 180 are bin centres
        self.bin_yaw = (torch.arange(n_bins, device=device, dtype=torch.float32)
                        * (2 * math.pi / n_bins) - math.pi)
        self.reset()

    def reset(self):
        """Forget everything (call at the start of every episode)."""
        self.free = torch.zeros(self.B, self.n, device=self.device)
        self.last_seen = torch.full((self.B, self.n), -float('inf'), device=self.device)
        self.trail = deque()                       # (time, positions (B, 3))

    # ------------------------------------------------------------------ helpers

    def yaw_to_bin(self, yaw):
        # shift by half a bin so each bin is centred on bin_yaw; +-180 both land in bin 0
        half = math.pi / self.n
        return torch.floor((yaw + math.pi + half) / (2 * math.pi) * self.n).long() % self.n

    @property
    def seen(self):
        return torch.isfinite(self.last_seen)

    def age(self, t):
        """Seconds since each bin was last updated (inf if never)."""
        return t - self.last_seen

    # ------------------------------------------------------------------ update

    @torch.no_grad()
    def update(self, t, pos, rays_world=None, dist=None):
        """
        t:          current time (s), a float
        pos:        (B, 3) drone positions, world frame
        rays_world: (B, P, 3) pixel ray directions in world frame (any length), optional
        dist:       (B, P) distance along each ray to the hit; inf or nan = nothing hit
        """
        rows = torch.arange(self.B, device=self.device)

        # 1. Trail: the way back to where we were ~trail_s ago is free.
        self.trail.append((t, pos.clone()))
        while len(self.trail) > 1 and self.trail[1][0] <= t - self.trail_s:
            self.trail.popleft()
        _, p_then = self.trail[0]
        back = p_then - pos
        back_len = back[:, :2].norm(dim=-1)
        has_trail = back_len > self.min_trail
        b_trail = self.yaw_to_bin(torch.atan2(back[:, 1], back[:, 0]))
        old = self.free[rows, b_trail]
        self.free[rows, b_trail] = torch.where(has_trail, back_len.clamp(max=self.d_cap), old)
        old_t = self.last_seen[rows, b_trail]
        self.last_seen[rows, b_trail] = torch.where(has_trail, torch.full_like(old_t, t), old_t)

        # 2. Camera: closest obstacle per bin, for bins in view. Overrides the trail.
        if rays_world is not None:
            horiz = rays_world[..., :2].norm(dim=-1)
            elev = torch.atan2(rays_world[..., 2], horiz)
            in_band = elev.abs() <= self.band
            b = self.yaw_to_bin(torch.atan2(rays_world[..., 1], rays_world[..., 0]))   # (B, P)
            d = torch.nan_to_num(dist.float(), nan=self.d_cap, posinf=self.d_cap).clamp(max=self.d_cap)
            d = torch.where(in_band, d, torch.full_like(d, float('inf')))  # out-of-band can't win

            closest = torch.full((self.B, self.n), float('inf'), device=self.device)
            closest = closest.scatter_reduce(1, b, d, reduce='amin', include_self=True)
            count = torch.zeros(self.B, self.n, device=self.device).scatter_add(1, b, in_band.float())
            observed = count > 0

            self.free = torch.where(observed, closest, self.free)
            self.last_seen = torch.where(observed, torch.full_like(self.last_seen, t), self.last_seen)

    # ------------------------------------------------------------------ queries

    def free_toward(self, yaw):
        """Remembered free distance toward each drone's yaw (B,) -> (B,); nan if never seen."""
        idx = self.yaw_to_bin(yaw)[:, None]
        val = self.free.gather(1, idx)[:, 0]
        return torch.where(self.seen.gather(1, idx)[:, 0], val, torch.full_like(val, float('nan')))

    def best_heading(self, t, avoid_yaw=None, avoid_halfwidth_deg=45.0, max_age=None):
        """
        Most open remembered direction for each drone.

        avoid_yaw:  optional (B,) yaw to exclude, +- avoid_halfwidth_deg (e.g. the trap)
        max_age:    optional, ignore bins older than this (s)
        Returns yaw (B,) and its free distance (B,); free = -1 means nothing usable.
        """
        score = torch.where(self.seen, self.free, torch.full_like(self.free, -1.0))
        if max_age is not None:
            score = torch.where(self.age(t) <= max_age, score, torch.full_like(score, -1.0))
        if avoid_yaw is not None:
            diff = wrap(self.bin_yaw[None] - avoid_yaw[:, None])
            score = torch.where(diff.abs() < math.radians(avoid_halfwidth_deg),
                                torch.full_like(score, -1.0), score)
        best = score.argmax(1)
        return self.bin_yaw[best], score.gather(1, best[:, None])[:, 0]