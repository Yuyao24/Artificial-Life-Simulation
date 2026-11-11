"""Top-level simulation: world + creatures + species + user events. Headless."""
from __future__ import annotations

import numpy as np

from .creatures import GENE_NAMES, Population
from .species import SpeciesTracker
from .world import World

HISTORY_EVERY = 10

# Founder archetypes: (speed, size, vision, metabolism, aggression, hue)
FOUNDERS = [
    ("grazer", [0.55, 0.35, 0.40, 0.55, 0.10, 0.33]),
    ("hulk",   [0.25, 0.75, 0.35, 0.35, 0.15, 0.58]),
    ("scout",  [0.70, 0.30, 0.75, 0.45, 0.20, 0.14]),
    ("hunter", [0.60, 0.70, 0.65, 0.40, 0.80, 0.98]),
]


class Simulation:
    def __init__(self, seed: int | None = None, width: int = 160, height: int = 100,
                 n_start: int = 600, species_interval: int = 100):
        if seed is None:
            seed = int(np.random.SeedSequence().entropy % 1_000_000)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.world = World(width, height, rng=self.rng)
        self.pop = Population(self.rng, width, height)
        self.tracker = SpeciesTracker(self.rng, interval=species_interval)
        self.t = 0
        self.history_t: list[int] = []
        self.history_total: list[int] = []
        self.history_species: list[dict[int, int]] = []
        self.history_temp: list[float] = []
        self.event_log: list[tuple[int, str]] = []
        self._seed_founders(n_start)
        self._record()

    # ------------------------------------------------------------------
    def _seed_founders(self, n_start: int) -> None:
        w, h = self.world.w, self.world.h
        per = n_start // len(FOUNDERS)
        centres = [(0.25, 0.3), (0.75, 0.3), (0.25, 0.7), (0.75, 0.7)]
        order = self.rng.permutation(len(FOUNDERS))
        ref = np.mean([g for _, g in FOUNDERS], axis=0)
        for (cx, cy), k in zip(centres, order):
            _, genes = FOUNDERS[k]
            g = np.clip(np.array(genes) + self.rng.normal(0, 0.03, (per, 6)), 0, 1)
            g[:, 5] %= 1.0
            n = per if FOUNDERS[k][0] != "hunter" else per // 5
            g = g[:n]
            sp = self.tracker.new_species(g.mean(0), 0, reference=ref)
            x = np.clip(self.rng.normal(cx * w, w * 0.08, n), 0, w - 1e-3)
            y = np.clip(self.rng.normal(cy * h, h * 0.08, n), 0, h - 1e-3)
            self.pop.add(x, y, g, species=np.full(n, sp.id))
        self.tracker.refresh(self.pop.genes, self.pop.species, 0)

    def _record(self) -> None:
        self.history_t.append(self.t)
        self.history_total.append(len(self.pop))
        ids, counts = np.unique(self.pop.species, return_counts=True)
        self.history_species.append(dict(zip(ids.tolist(), counts.tolist())))
        self.history_temp.append(self.world.temperature)

    def log(self, msg: str) -> None:
        self.event_log.append((self.t, msg))

    # ------------------------------------------------------------------
    def step(self) -> None:
        self.t += 1
        self.world.step()
        self.pop.step(self.world)
        if self.t % self.tracker.interval == 0:
            self.pop.species = self.tracker.update(self.pop.genes, self.pop.species, self.t)
        elif self.t % HISTORY_EVERY == 0:
            self.tracker.refresh(self.pop.genes, self.pop.species, self.t)
        if self.t % HISTORY_EVERY == 0:
            self._record()

    def run(self, steps: int) -> None:
        for _ in range(steps):
            self.step()

    # --- user events ----------------------------------------------------
    def ice_age(self) -> None:
        self.world.shock_temperature(-0.35)
        self.log("Ice age")

    def heatwave(self) -> None:
        self.world.shock_temperature(+0.35)
        self.log("Heatwave")

    def meteor(self, x: float | None = None, y: float | None = None, r: float = 16.0):
        if x is None:
            # aim at the densest region for maximum drama
            if len(self.pop):
                i = self.rng.integers(len(self.pop))
                x, y = self.pop.x[i], self.pop.y[i]
            else:
                x, y = self.world.w / 2, self.world.h / 2
        killed = self.pop.kill_in_disk(x, y, r)
        self.world.scorch(x, y, r)
        self.log(f"Meteor strike ({killed} killed)")
        return x, y, r, killed

    def drop_food(self, x: float, y: float) -> None:
        self.world.add_food(x, y)

    # --- summaries ------------------------------------------------------
    def mean_genes(self) -> dict[str, float]:
        if len(self.pop) == 0:
            return {g: 0.0 for g in GENE_NAMES}
        m = self.pop.genes.mean(0)
        return dict(zip(GENE_NAMES, m.tolist()))
