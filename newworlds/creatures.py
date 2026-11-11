"""Creatures stored as a structure of numpy arrays (no per-creature Python objects).

Genes (all floats in [0, 1]):
    0 speed, 1 size, 2 vision, 3 metabolism, 4 aggression, 5 hue
"""
from __future__ import annotations

import numpy as np

GENE_NAMES = ("speed", "size", "vision", "metabolism", "aggression", "hue")
SPEED, SIZE, VISION, METAB, AGGR, HUE = range(6)
N_GENES = 6

MAX_POP = 1500
MUTATION_SIGMA = 0.04
HUE_SIGMA = 0.02
MACRO_RATE = 0.01        # chance of a large-effect mutation per birth
MACRO_SIGMA = 0.3
BUCKET = 10.0           # spatial hash bucket size (cells) >= max vision radius
BUCKET_K = 10           # max creatures sampled per bucket (keeps sensing O(N))
MAX_AGE = 900

# Energy rules -----------------------------------------------------------
BASAL_COST = 0.012
SIZE_COST = 0.018
MOVE_COST = 0.045
VISION_COST = 0.006
METAB_COST = 0.015
AGGR_COST = 0.02        # upkeep of claws and muscle
START_ENERGY = 4.0
PLANT_EFF = 1.3         # energy per unit of food for a pure herbivore
PLANT_AGGR_PENALTY = 0.9
PREY_ENERGY_FRAC = 0.6  # share of the prey's energy a kill yields
PREY_BODY_ENERGY = 4.5  # extra energy per unit of prey body size
CATCH_BASE = 0.2       # chance per step a predator in range lands a kill
CATCH_AGGR = 0.5
INTERFERENCE = 0.5


def body_size(size_gene):
    return 0.5 + size_gene                     # 0.5 .. 1.5


def move_speed(speed_gene):
    return 0.25 + 1.0 * speed_gene             # cells per step


def vision_radius(vision_gene):
    return 2.0 + 8.0 * vision_gene             # 2 .. 10 cells


def repro_threshold(size_gene):
    return 6.0 + 6.0 * body_size(size_gene)


def step_cost(genes: np.ndarray, moved: np.ndarray) -> np.ndarray:
    """Energy cost for one step. `moved` is distance travelled (cells)."""
    s = body_size(genes[:, SIZE])
    return (BASAL_COST
            + SIZE_COST * s
            + MOVE_COST * moved ** 2 * s
            + VISION_COST * genes[:, VISION]
            + METAB_COST * genes[:, METAB]
            + AGGR_COST * genes[:, AGGR])


def mutate(genes: np.ndarray, rng: np.random.Generator,
           sigma: float = MUTATION_SIGMA, macro_rate: float = MACRO_RATE) -> np.ndarray:
    """Return a mutated copy of `genes` (N, 6).

    Every gene gets a small Gaussian nudge; with probability `macro_rate` one
    random gene also takes a large jump. Hue drifts slowly and wraps around.
    """
    out = genes + rng.normal(0.0, sigma, genes.shape)
    if macro_rate > 0 and len(genes):
        big = np.nonzero(rng.random(len(genes)) < macro_rate)[0]
        if len(big):
            which = rng.integers(0, HUE, len(big))
            out[big, which] += rng.normal(0.0, MACRO_SIGMA, len(big))
    out[:, HUE] = (genes[:, HUE] + rng.normal(0.0, HUE_SIGMA, len(genes))) % 1.0
    out[:, :HUE] = np.clip(out[:, :HUE], 0.0, 1.0)
    return out


class Population:
    def __init__(self, rng: np.random.Generator, width: int, height: int):
        self.rng = rng
        self.w = width
        self.h = height
        self.x = np.zeros(0)
        self.y = np.zeros(0)
        self.energy = np.zeros(0)
        self.genes = np.zeros((0, N_GENES))
        self.age = np.zeros(0, dtype=np.int32)
        self.species = np.zeros(0, dtype=np.int32)
        self.heading = np.zeros(0)
        self.births = 0
        self.deaths = 0
        self.kills = 0

    def __len__(self) -> int:
        return len(self.x)

    # ------------------------------------------------------------------
    def add(self, x, y, genes, energy=None, species=None) -> None:
        n = len(x)
        if n == 0:
            return
        self.x = np.concatenate([self.x, np.asarray(x, float)])
        self.y = np.concatenate([self.y, np.asarray(y, float)])
        self.genes = np.concatenate([self.genes, np.asarray(genes, float)])
        e = np.full(n, START_ENERGY) if energy is None else np.asarray(energy, float)
        self.energy = np.concatenate([self.energy, e])
        self.age = np.concatenate([self.age, np.zeros(n, np.int32)])
        sp = np.zeros(n, np.int32) if species is None else np.asarray(species, np.int32)
        self.species = np.concatenate([self.species, sp])
        self.heading = np.concatenate([self.heading, self.rng.uniform(0, 2 * np.pi, n)])

    def keep(self, mask: np.ndarray) -> None:
        for name in ("x", "y", "energy", "genes", "age", "species", "heading"):
            setattr(self, name, getattr(self, name)[mask])

    def kill_in_disk(self, cx, cy, r) -> int:
        dead = (self.x - cx) ** 2 + (self.y - cy) ** 2 <= r * r
        n = int(dead.sum())
        self.deaths += n
        self.keep(~dead)
        return n

    # ------------------------------------------------------------------
    def _neighbours(self):
        """Spatial hash: candidate neighbour indices (N, 9*K), -1 = empty."""
        n = len(self)
        nbx = int(np.ceil(self.w / BUCKET))
        nby = int(np.ceil(self.h / BUCKET))
        bx = np.clip((self.x / BUCKET).astype(int), 0, nbx - 1)
        by = np.clip((self.y / BUCKET).astype(int), 0, nby - 1)
        b = bx * nby + by
        # random order so truncation to K per bucket is unbiased
        order = self.rng.permutation(n)
        order = order[np.argsort(b[order], kind="stable")]
        sb = b[order]
        starts = np.searchsorted(sb, np.arange(nbx * nby))
        rank = np.arange(n) - starts[sb]
        ok = rank < BUCKET_K
        table = np.full((nbx * nby, BUCKET_K), -1, dtype=np.int64)
        table[sb[ok], rank[ok]] = order[ok]
        offs = np.array([(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1)])
        nx_ = bx[:, None] + offs[None, :, 0]
        ny_ = by[:, None] + offs[None, :, 1]
        valid = (nx_ >= 0) & (nx_ < nbx) & (ny_ >= 0) & (ny_ < nby)
        nb = np.where(valid, nx_ * nby + ny_, 0)
        cand = table[nb]                                    # (N, 9, K)
        cand[~valid] = -1
        return cand.reshape(n, -1)

    def step(self, world) -> None:
        n = len(self)
        if n == 0:
            return
        rng = self.rng
        g = self.genes
        size = body_size(g[:, SIZE])
        speed = move_speed(g[:, SPEED])
        vis = vision_radius(g[:, VISION])
        aggr = g[:, AGGR]

        # --- sense food: sample 8 directions at two ranges --------------
        angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        ca, sa = np.cos(angles), np.sin(angles)
        best_val = world.food_at(self.x, self.y) * 0.9
        dir_x = np.zeros(n)
        dir_y = np.zeros(n)
        for frac in (0.4, 1.0):
            px = self.x[:, None] + ca[None, :] * vis[:, None] * frac
            py = self.y[:, None] + sa[None, :] * vis[:, None] * frac
            inside = (px >= 0) & (px < world.w) & (py >= 0) & (py < world.h)
            val = world.food_at(px, py) * inside * (1.0 - 0.15 * frac)
            j = np.argmax(val, axis=1)
            v = val[np.arange(n), j]
            better = v > best_val
            best_val = np.where(better, v, best_val)
            dir_x = np.where(better, ca[j], dir_x)
            dir_y = np.where(better, sa[j], dir_y)
        # herbivores weight food highly; pure predators barely care
        food_w = (1.0 - 0.8 * aggr) * np.minimum(best_val * 4, 1.0)
        vx = dir_x * food_w
        vy = dir_y * food_w

        # --- sense creatures: flee larger predators, chase smaller prey --
        cand = self._neighbours()
        valid = cand >= 0
        ci = np.where(valid, cand, 0)
        x32 = self.x.astype(np.float32)
        y32 = self.y.astype(np.float32)
        dx = x32[ci] - x32[:, None]
        dy = y32[ci] - y32[:, None]
        d2 = dx * dx + dy * dy
        me = np.arange(n)[:, None]
        valid &= (ci != me) & (d2 <= (vis * vis).astype(np.float32)[:, None])
        size32 = size.astype(np.float32)
        osize = size32[ci]
        threat = valid & (osize > size32[:, None] * 1.2) & (aggr[ci] > 0.5)
        tw = threat / (d2 + 1e-3)              # unit vector / distance
        fx = -(dx * tw).sum(1)
        fy = -(dy * tw).sum(1)
        fn = np.hypot(fx, fy) + 1e-9
        fleeing = fn > 1e-6
        vx += np.where(fleeing, fx / fn * 1.5, 0)
        vy += np.where(fleeing, fy / fn * 1.5, 0)

        # satiated predators (energy above 60% of the breeding threshold) don't hunt
        hunter = (aggr > 0.5) & (self.energy < 0.6 * repro_threshold(g[:, SIZE]))
        prey_ok = valid & hunter[:, None] & (osize < size32[:, None] * 0.85)
        pd2 = np.where(prey_ok, d2, np.inf)
        pj = np.argmin(pd2, axis=1)
        pdist2 = pd2[np.arange(n), pj]
        has_prey = np.isfinite(pdist2)
        prey_idx = ci[np.arange(n), pj]
        pdd = np.sqrt(np.where(has_prey, pdist2, 1.0)) + 1e-6
        chase_w = 2.0 * aggr
        vx += np.where(has_prey, dx[np.arange(n), pj] / pdd * chase_w, 0)
        vy += np.where(has_prey, dy[np.arange(n), pj] / pdd * chase_w, 0)

        # --- wander: smooth random heading -------------------------------
        self.heading += rng.normal(0, 0.5, n)
        vx += np.cos(self.heading) * 0.35
        vy += np.sin(self.heading) * 0.35

        # --- move ---------------------------------------------------------
        vn = np.hypot(vx, vy) + 1e-9
        # slow down when sitting on good food and not threatened
        here = world.food_at(self.x, self.y)
        throttle = np.where(fleeing | has_prey, 1.0, np.clip(1.2 - here * 1.5, 0.25, 1.0))
        step_len = speed * throttle
        nxp = self.x + vx / vn * step_len
        nyp = self.y + vy / vn * step_len
        # bounce off walls
        hitx = (nxp < 0) | (nxp >= self.w)
        hity = (nyp < 0) | (nyp >= self.h)
        self.heading = np.where(hitx, np.pi - self.heading, self.heading)
        self.heading = np.where(hity, -self.heading, self.heading)
        self.x = np.clip(nxp, 0, self.w - 1e-3)
        self.y = np.clip(nyp, 0, self.h - 1e-3)

        # --- eat plants ---------------------------------------------------
        ix = self.x.astype(int)
        iy = self.y.astype(int)
        bite = (0.15 + 0.45 * g[:, METAB]) * size
        avail = world.food[ix, iy]
        # several creatures on one cell: share the cell's food fairly
        cell = ix * world.h + iy
        crowd = np.bincount(cell, minlength=world.w * world.h)[cell]
        eaten = np.minimum(bite, avail / crowd)
        np.subtract.at(world.food, (ix, iy), eaten)
        np.maximum(world.food, 0.0, out=world.food)
        plant_eff = PLANT_EFF * (1.0 - PLANT_AGGR_PENALTY * aggr)
        self.energy += eaten * plant_eff

        # --- energy cost --------------------------------------------------
        self.energy -= step_cost(g, step_len)
        self.age += 1

        # --- predation ----------------------------------------------------
        catch = has_prey & (pdist2 <= (0.6 + 0.5 * size) ** 2)
        # interference: hunters crowding the same patch get in each other's way
        rivals = (valid & (aggr[ci] > 0.5)).sum(1)
        # fast prey can outrun slow hunters
        outrun = np.clip(speed / np.maximum(speed[prey_idx], 1e-3), 0.2, 1.0)
        p_catch = (CATCH_BASE + CATCH_AGGR * aggr) * outrun / (1.0 + INTERFERENCE * rivals)
        catch &= rng.random(n) < p_catch
        dead = np.zeros(n, dtype=bool)
        if catch.any():
            preds = np.nonzero(catch)[0]
            preys = prey_idx[preds]
            # hunters that are themselves eaten this step cannot eat
            preds = preds[rng.permutation(len(preds))]
            preys = prey_idx[preds]
            _, first = np.unique(preys, return_index=True)
            preds, preys = preds[first], preys[first]
            ok = ~np.isin(preds, preys)
            preds, preys = preds[ok], preys[ok]
            gain = PREY_ENERGY_FRAC * np.maximum(self.energy[preys], 0) + PREY_BODY_ENERGY * size[preys]
            self.energy[preds] += gain * (0.4 + 0.6 * aggr[preds])
            dead[preys] = True
            self.kills += len(preys)

        # --- deaths -------------------------------------------------------
        old = self.age > MAX_AGE * (0.7 + 0.6 * rng.random(n))
        dead |= (self.energy <= 0) | old
        if dead.any():
            self.deaths += int(dead.sum())
            self.keep(~dead)

        self.reproduce()

    def reproduce(self) -> None:
        if len(self) == 0:
            return
        ready = self.energy >= repro_threshold(self.genes[:, SIZE])
        idx = np.nonzero(ready)[0]
        room = MAX_POP - len(self)
        if room <= 0 or len(idx) == 0:
            # crowded: cap energy so stored energy does not explode
            np.minimum(self.energy, repro_threshold(self.genes[:, SIZE]), out=self.energy)
            return
        if len(idx) > room:
            idx = self.rng.choice(idx, room, replace=False)
        self.energy[idx] *= 0.5
        child_genes = mutate(self.genes[idx], self.rng)
        jitter = self.rng.normal(0, 0.7, (len(idx), 2))
        cx = np.clip(self.x[idx] + jitter[:, 0], 0, self.w - 1e-3)
        cy = np.clip(self.y[idx] + jitter[:, 1], 0, self.h - 1e-3)
        self.add(cx, cy, child_genes, energy=self.energy[idx].copy(),
                 species=self.species[idx].copy())
        self.births += len(idx)
        np.minimum(self.energy, repro_threshold(self.genes[:, SIZE]), out=self.energy)
