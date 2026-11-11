"""The world grid: climate map, global temperature and food.

Pure numpy, no rendering. Coordinates are in cell units: x in [0, W), y in [0, H).
Arrays are indexed [x, y] so they map directly onto pygame surfarrays.
"""
from __future__ import annotations

import numpy as np

FOOD_MAX = 1.0          # absolute ceiling for food in a cell
REGROW_RATE = 0.012     # fraction of the gap to the cap regrown per step
DECAY_RATE = 0.05       # fraction of the excess above the cap lost per step
TEMP_SENSITIVITY = 0.9  # how strongly global temperature shifts local climate
TEMP_BASELINE = 0.5


def value_noise(w: int, h: int, rng: np.random.Generator, octaves: int = 4,
                base_cells: int = 4, persistence: float = 0.5) -> np.ndarray:
    """Smooth fractal value noise in [0, 1], shape (w, h)."""
    out = np.zeros((w, h), dtype=np.float64)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        n = base_cells * (2 ** o)
        gx, gy = n + 1, max(2, int(round(n * h / w)) + 1)
        grid = rng.random((gx, gy))
        xs = np.linspace(0, gx - 1, w, endpoint=False)
        ys = np.linspace(0, gy - 1, h, endpoint=False)
        x0 = xs.astype(int)
        y0 = ys.astype(int)
        tx = xs - x0
        ty = ys - y0
        # smoothstep interpolation
        tx = tx * tx * (3 - 2 * tx)
        ty = ty * ty * (3 - 2 * ty)
        x1 = np.minimum(x0 + 1, gx - 1)
        y1 = np.minimum(y0 + 1, gy - 1)
        a = grid[x0][:, y0]
        b = grid[x1][:, y0]
        c = grid[x0][:, y1]
        d = grid[x1][:, y1]
        txc = tx[:, None]
        tyc = ty[None, :]
        layer = (a * (1 - txc) + b * txc) * (1 - tyc) + (c * (1 - txc) + d * txc) * tyc
        out += amp * layer
        total += amp
        amp *= persistence
    out /= total
    lo, hi = out.min(), out.max()
    return (out - lo) / (hi - lo + 1e-12)


class World:
    def __init__(self, width: int = 160, height: int = 100, seed: int | None = None,
                 rng: np.random.Generator | None = None):
        self.w = width
        self.h = height
        self.rng = rng if rng is not None else np.random.default_rng(seed)
        # Local climate: 0 = polar, 1 = tropical. Latitude gradient + noise.
        lat = np.abs(np.linspace(-1, 1, height))[None, :]          # 0 at equator
        noise = value_noise(width, height, self.rng)
        self.climate = np.clip(0.55 * noise + 0.45 * (1 - lat), 0, 1)
        # Fertility: a second noise map so some warm regions are deserts.
        self.fertility = 0.35 + 0.65 * value_noise(width, height, self.rng, octaves=3, base_cells=3)
        self.temperature = TEMP_BASELINE   # global temperature, 0 = frozen, 1 = scorching
        self.temp_target = TEMP_BASELINE   # temperature relaxes toward this
        self.time = 0
        self.cap = np.zeros((width, height))
        self.update_cap()
        self.food = self.cap * self.rng.uniform(0.5, 1.0, self.cap.shape)

    # ------------------------------------------------------------------
    def local_temperature(self) -> np.ndarray:
        return self.climate + TEMP_SENSITIVITY * (self.temperature - TEMP_BASELINE) - 0.1

    def update_cap(self) -> None:
        """Food cap: highest where the local temperature is mild (~0.5)."""
        lt = self.local_temperature()
        suit = np.exp(-((lt - 0.5) / 0.28) ** 2)
        self.cap = FOOD_MAX * suit * self.fertility

    def step(self) -> None:
        self.time += 1
        # Slow natural climate cycle plus relaxation of event shocks.
        cycle = 0.06 * np.sin(self.time / 900.0)
        self.temp_target += (TEMP_BASELINE + cycle - self.temp_target) * 0.002
        self.temperature += (self.temp_target - self.temperature) * 0.01
        self.temperature += self.rng.normal(0, 0.0015)
        self.temperature = float(np.clip(self.temperature, 0.0, 1.0))
        if self.time % 5 == 0:
            self.update_cap()
        gap = self.cap - self.food
        grow = np.where(gap > 0, REGROW_RATE * gap, DECAY_RATE * gap)
        self.food += grow
        np.clip(self.food, 0.0, FOOD_MAX * 2, out=self.food)

    # ------------------------------------------------------------------
    def _disk(self, cx: float, cy: float, r: float) -> np.ndarray:
        xs = np.arange(self.w)[:, None]
        ys = np.arange(self.h)[None, :]
        return (xs - cx) ** 2 + (ys - cy) ** 2 <= r * r

    def add_food(self, cx: float, cy: float, r: float = 6.0, amount: float = 1.5) -> None:
        xs = np.arange(self.w)[:, None]
        ys = np.arange(self.h)[None, :]
        d2 = (xs - cx) ** 2 + (ys - cy) ** 2
        self.food += amount * np.exp(-d2 / (2 * (r / 2) ** 2)) * (d2 <= r * r)
        np.clip(self.food, 0.0, FOOD_MAX * 2, out=self.food)

    def scorch(self, cx: float, cy: float, r: float) -> None:
        self.food[self._disk(cx, cy, r)] = 0.0

    def shock_temperature(self, delta: float) -> None:
        """Ice age (negative) or heatwave (positive)."""
        self.temp_target = float(np.clip(self.temp_target + delta, 0.0, 1.0))
        self.temperature = float(np.clip(self.temperature + delta * 0.35, 0.0, 1.0))
        self.update_cap()

    def food_at(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        ix = np.clip(x.astype(int), 0, self.w - 1)
        iy = np.clip(y.astype(int), 0, self.h - 1)
        return self.food[ix, iy]
