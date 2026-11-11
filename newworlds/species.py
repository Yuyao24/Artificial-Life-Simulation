"""Species detection by clustering gene vectors, plus naming and lineage tracking.

Every `interval` steps the tracker:
  1. tries to split each species in two with 2-means; if the two halves are far
     enough apart in gene space, the smaller half becomes a new daughter species;
  2. re-assigns every creature to the nearest species centroid;
  3. merges species that have converged, and marks empty species extinct.
Children inherit their parent's species label between updates, so lineages are
continuous and the parent links form a phylogenetic tree.
"""
from __future__ import annotations

import colorsys
from dataclasses import dataclass, field

import numpy as np

from .creatures import AGGR, HUE, METAB, SIZE, SPEED, VISION

GENE_WEIGHTS = np.array([1.0, 1.0, 1.0, 1.0, 1.4])   # speed size vision metab aggr
HUE_WEIGHT = 0.12                                    # radius of the hue circle
SPLIT_DIST = 0.26       # min centroid distance for a split
SPLIT_RATIO = 2.5       # min separation / within-cluster spread along the split axis
MERGE_DIST = 0.12
MIN_SPECIES_SIZE = 12
MAX_LIVING = 14


def features(genes: np.ndarray) -> np.ndarray:
    """Weighted feature vectors; hue is mapped onto a circle so 0.99 ~ 0.01."""
    ang = genes[:, HUE] * 2 * np.pi
    return np.column_stack([genes[:, :HUE] * GENE_WEIGHTS,
                            HUE_WEIGHT * np.cos(ang), HUE_WEIGHT * np.sin(ang)])


def kmeans(X: np.ndarray, k: int, rng: np.random.Generator, iters: int = 15):
    """Plain k-means with k-means++ init. Returns (labels, centroids)."""
    n = len(X)
    k = min(k, n)
    cent = [X[rng.integers(n)]]
    for _ in range(1, k):
        d2 = np.min(((X[:, None, :] - np.array(cent)[None]) ** 2).sum(-1), axis=1)
        p = d2 / d2.sum() if d2.sum() > 0 else None
        cent.append(X[rng.choice(n, p=p)])
    cent = np.array(cent)
    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        d2 = ((X[:, None, :] - cent[None]) ** 2).sum(-1)
        new = np.argmin(d2, axis=1)
        if np.array_equal(new, labels) and _ > 0:
            break
        labels = new
        for j in range(k):
            m = labels == j
            if m.any():
                cent[j] = X[m].mean(0)
    return labels, cent


def is_distinct(X: np.ndarray, labels: np.ndarray, cent: np.ndarray) -> bool:
    """Are the two k-means halves far apart *and* separated by a gap?"""
    axis = cent[1] - cent[0]
    d = float(np.linalg.norm(axis))
    if d < SPLIT_DIST:
        return False
    proj = X @ (axis / d)
    spread = np.sqrt(np.mean([proj[labels == j].var() for j in (0, 1)])) + 1e-9
    return d / spread >= SPLIT_RATIO


# --- names ---------------------------------------------------------------
_ONSET = ["b", "c", "d", "f", "g", "k", "l", "m", "n", "p", "r", "s", "t", "v", "z",
          "th", "kr", "dr", "vr", "sk", "x", "qu", "ph", "zh"]
_VOWEL = ["a", "e", "i", "o", "u", "ae", "y", "io", "au"]
_CODA = ["", "", "n", "r", "s", "x", "l", "th", "m"]
_ENDS = ["ia", "on", "us", "ax", "ora", "ix", "ium", "yr", "ex", "ara"]
EPITHETS = {
    (SPEED, 1): "velox", (SPEED, -1): "tardus",
    (SIZE, 1): "magnus", (SIZE, -1): "minimus",
    (VISION, 1): "oculatus", (VISION, -1): "caecus",
    (METAB, 1): "vorax", (METAB, -1): "frugalis",
    (AGGR, 1): "ferox", (AGGR, -1): "placidus",
}


def make_genus(rng: np.random.Generator) -> str:
    s = "".join(rng.choice(_ONSET) + rng.choice(_VOWEL) + rng.choice(_CODA)
                for _ in range(rng.integers(1, 3)))
    return (s + rng.choice(_ENDS)).capitalize()


def epithet(centroid_genes: np.ndarray, reference: np.ndarray) -> str:
    diff = centroid_genes[:HUE] - reference[:HUE]
    diff[AGGR] *= 1.5
    i = int(np.argmax(np.abs(diff)))
    if abs(diff[i]) < 0.04:
        return "vulgaris"
    return EPITHETS[(i, 1 if diff[i] > 0 else -1)]


@dataclass
class Species:
    id: int
    name: str
    genus: str
    color: tuple
    parent: int | None
    born: int
    extinct: int | None = None
    centroid: np.ndarray = field(default_factory=lambda: np.zeros(6))
    count: int = 0
    peak: int = 0
    fate: str = ""

    @property
    def alive(self) -> bool:
        return self.extinct is None


def hue_color(h: float, s: float = 0.75, v: float = 0.95) -> tuple:
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, s, v)
    return int(r * 255), int(g * 255), int(b * 255)


class SpeciesTracker:
    def __init__(self, rng: np.random.Generator, interval: int = 100):
        self.rng = rng
        self.interval = interval
        self.species: dict[int, Species] = {}
        self.next_id = 1
        self.events: list[tuple[int, str]] = []    # (time, message)

    # ------------------------------------------------------------------
    def living(self) -> list[Species]:
        return [s for s in self.species.values() if s.alive]

    def _pick_color(self, hue: float) -> tuple:
        used = [colorsys.rgb_to_hsv(*(c / 255 for c in s.color))[0] for s in self.living()]
        h = hue
        for _ in range(12):
            if all(min(abs(h - u), 1 - abs(h - u)) > 0.06 for u in used):
                break
            h = (h + 0.083) % 1.0
        return hue_color(h)

    def new_species(self, centroid: np.ndarray, t: int, parent: int | None = None,
                    reference: np.ndarray | None = None) -> Species:
        sid = self.next_id
        self.next_id += 1
        ref = reference if reference is not None else np.full(6, 0.5)
        if parent is not None and self.rng.random() < 0.5:
            genus = self.species[parent].genus
        else:
            genus = make_genus(self.rng)
        ep = epithet(centroid, ref)
        name = f"{genus} {ep}"
        taken = {s.name for s in self.species.values()}
        k = 2
        while name in taken:
            name = f"{genus} {ep} {_roman(k)}"
            k += 1
        sp = Species(sid, name, genus, self._pick_color(centroid[HUE]), parent, t,
                     centroid=centroid.copy())
        self.species[sid] = sp
        if parent is None:
            self.events.append((t, f"{name} founded"))
        else:
            self.events.append((t, f"{name} split from {self.species[parent].name}"))
        return sp

    def _extinct(self, sp: Species, t: int, fate: str) -> None:
        sp.extinct = t
        sp.fate = fate
        sp.count = 0
        self.events.append((t, f"{sp.name} {fate}"))

    # ------------------------------------------------------------------
    def update(self, genes: np.ndarray, labels: np.ndarray, t: int) -> np.ndarray:
        """Re-cluster. Returns the new label array (same length as genes)."""
        labels = labels.copy()
        if len(genes) == 0:
            for sp in self.living():
                self._extinct(sp, t, "went extinct")
            return labels
        X = features(genes)
        ref = genes.mean(0)

        # creatures without a known living species: give them one
        known = np.array([sid in self.species and self.species[sid].alive for sid in labels])
        if (~known).any():
            orphans = np.nonzero(~known)[0]
            if not self.living():
                sp = self.new_species(genes[orphans].mean(0), t, reference=ref)
            labels[orphans] = self.living()[0].id

        # 1. try to split each species in two
        for sp in list(self.living()):
            idx = np.nonzero(labels == sp.id)[0]
            if len(idx) < 2 * MIN_SPECIES_SIZE or len(self.living()) >= MAX_LIVING:
                continue
            sub, cent = kmeans(X[idx], 2, self.rng, iters=10)
            sizes = np.bincount(sub, minlength=2)
            if sizes.min() < MIN_SPECIES_SIZE:
                continue
            if not is_distinct(X[idx], sub, cent):
                continue
            small = int(np.argmin(sizes))
            child_idx = idx[sub == small]
            child = self.new_species(genes[child_idx].mean(0), t, parent=sp.id,
                                     reference=genes[idx[sub != small]].mean(0))
            labels[child_idx] = child.id

        # 2. one Lloyd step: everyone joins the nearest living centroid
        live = [sp for sp in self.living() if (labels == sp.id).any()]
        if live:
            cents = np.array([X[labels == sp.id].mean(0) for sp in live])
            d2 = ((X[:, None, :] - cents[None]) ** 2).sum(-1)
            nearest = np.argmin(d2, axis=1)
            own = np.array([sp.id for sp in live])
            own_d = np.full(len(X), np.inf)
            for j, sid in enumerate(own):
                m = labels == sid
                own_d[m] = d2[m, j]
            # hysteresis: only switch if clearly closer to another species
            switch = d2[np.arange(len(X)), nearest] < 0.6 * own_d
            labels = np.where(switch, own[nearest], labels)

        # 3. merge converged species (younger into older)
        merged = True
        while merged:
            merged = False
            live = [sp for sp in self.living() if (labels == sp.id).any()]
            if len(live) < 2:
                break
            cents = np.array([X[labels == sp.id].mean(0) for sp in live])
            d = np.sqrt(((cents[:, None] - cents[None]) ** 2).sum(-1))
            np.fill_diagonal(d, np.inf)
            i, j = np.unravel_index(np.argmin(d), d.shape)
            if d[i, j] < MERGE_DIST:
                old, young = sorted((live[i], live[j]), key=lambda s: s.born)
                labels[labels == young.id] = old.id
                self._extinct(young, t, f"merged into {old.name}")
                merged = True

        # 4. bookkeeping: counts, centroids, extinctions
        self.refresh(genes, labels, t)
        return labels

    def refresh(self, genes: np.ndarray, labels: np.ndarray, t: int) -> None:
        """Update counts (cheap; can run every step)."""
        for sp in self.living():
            m = labels == sp.id
            sp.count = int(m.sum())
            if sp.count == 0:
                self._extinct(sp, t, "went extinct")
            else:
                sp.peak = max(sp.peak, sp.count)
                if t % self.interval == 0:
                    sp.centroid = genes[m].mean(0)


def _roman(n: int) -> str:
    vals = [(10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
    out = ""
    for v, r in vals:
        while n >= v:
            out += r
            n -= v
    return out
