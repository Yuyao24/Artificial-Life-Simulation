"""pygame-ce front end. All drawing lives here; the simulation knows nothing about it."""
from __future__ import annotations

import colorsys
import math
import time

import numpy as np
import pygame

from .creatures import AGGR, GENE_NAMES, HUE, SIZE
from .simulation import Simulation

PANEL_W = 380
BG = (12, 14, 20)
PANEL_BG = (18, 21, 30)
TEXT = (225, 230, 240)
DIM = (130, 138, 155)
ACCENT = (120, 220, 170)
SPEEDS = [1, 2, 3, 5, 8, 12, 20]
GRAPH_POINTS = 300          # history samples shown (x HISTORY_EVERY steps)


def hsv_array(h, s, v):
    """Vectorised HSV -> RGB (uint8). h, s, v are arrays in [0, 1]."""
    h6 = (h % 1.0) * 6.0
    i = np.floor(h6).astype(int) % 6
    f = h6 - np.floor(h6)
    p = v * (1 - s)
    q = v * (1 - s * f)
    t = v * (1 - s * (1 - f))
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


class Effect:
    def __init__(self, kind, x, y, r, duration):
        self.kind, self.x, self.y, self.r = kind, x, y, r
        self.t0 = time.perf_counter()
        self.duration = duration

    @property
    def age(self):
        return (time.perf_counter() - self.t0) / self.duration


class Renderer:
    def __init__(self, sim: Simulation, cell: int = 8):
        pygame.display.init()
        pygame.font.init()
        pygame.display.set_caption("New Worlds")
        self.cell = cell
        self.sim = sim
        self.world_w = sim.world.w * cell
        self.world_h = sim.world.h * cell
        self.screen = pygame.display.set_mode((self.world_w + PANEL_W, max(self.world_h, 720)))
        self.font = pygame.font.SysFont("dejavusans,arial", 14)
        self.font_b = pygame.font.SysFont("dejavusans,arial", 15, bold=True)
        self.font_s = pygame.font.SysFont("dejavusans,arial", 12)
        self.font_title = pygame.font.SysFont("dejavusans,arial", 26, bold=True)
        self.font_banner = pygame.font.SysFont("dejavusans,arial", 64, bold=True)
        self.effects: list[Effect] = []
        self.banner = None          # (text, colour, t0)
        self.toasts: list[tuple[str, tuple, float]] = []
        self.seen_events = 0
        self.color_by_species = False
        self.graph_cache = None
        self.graph_key = None

    def reset(self, sim: Simulation):
        self.sim = sim
        self.effects.clear()
        self.toasts.clear()
        self.seen_events = 0
        self.graph_key = None
        self.show_banner(f"NEW WORLD  #{sim.seed}", ACCENT)

    # ------------------------------------------------------------------
    def show_banner(self, text, colour):
        self.banner = (text, colour, time.perf_counter())

    def add_effect(self, kind, x, y, r=0.0, duration=1.0):
        self.effects.append(Effect(kind, x, y, r, duration))

    def to_world(self, px, py):
        return px / self.cell, py / self.cell

    # --- terrain --------------------------------------------------------
    def _terrain(self) -> pygame.Surface:
        w = self.sim.world
        lt = np.clip(w.local_temperature(), -0.3, 1.3)
        # base ground: icy blue when cold, brown when mild, sand when hot
        cold = np.clip((0.35 - lt) / 0.4, 0, 1)[..., None]
        hot = np.clip((lt - 0.65) / 0.4, 0, 1)[..., None]
        ground = np.array([58, 46, 36]) * (1 - cold - hot) \
            + np.array([170, 195, 220]) * cold + np.array([175, 130, 70]) * hot
        food = np.clip(w.food / 0.8, 0, 1.5)[..., None]
        green = np.array([50, 200, 85])
        f1 = np.minimum(food, 1)
        rgb = ground * (1 - f1 * 0.9) + green * f1 * 0.9 + np.array([90, 60, 0]) * np.maximum(food - 1, 0)
        # subtle relief from fertility so the map has texture
        rgb *= (0.82 + 0.3 * w.fertility[..., None])
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        surf = pygame.surfarray.make_surface(rgb)
        return pygame.transform.smoothscale(surf, (self.world_w, self.world_h))

    # --- creatures ------------------------------------------------------
    def _creatures(self, surf):
        p = self.sim.pop
        n = len(p)
        if n == 0:
            return
        c = self.cell
        xs = (p.x * c).astype(int)
        ys = (p.y * c).astype(int)
        rad = (c * (0.3 + 0.6 * p.genes[:, SIZE])).astype(int) + 1
        if self.color_by_species:
            lut = {sp.id: sp.color for sp in self.sim.tracker.species.values()}
            cols = np.array([lut.get(int(s), (200, 200, 200)) for s in p.species], dtype=np.uint8)
        else:
            cols = hsv_array(p.genes[:, HUE], np.full(n, 0.7), np.full(n, 1.0))
        hunters = p.genes[:, AGGR] > 0.5
        draw = pygame.draw.circle
        dark = (10, 10, 14)
        for i in range(n):
            x, y, r = xs[i], ys[i], rad[i]
            col = cols[i]
            draw(surf, dark, (x, y), r + 1)
            draw(surf, col, (x, y), r)
            if hunters[i]:
                draw(surf, (255, 255, 255), (x, y), r + 2, 1)   # white ring = predator

    # --- effects --------------------------------------------------------
    def _effects(self, surf):
        alive = []
        overlay = None
        for e in self.effects:
            a = e.age
            if a >= 1:
                continue
            alive.append(e)
            c = self.cell
            if e.kind == "meteor":
                if overlay is None:
                    overlay = pygame.Surface((self.world_w, self.world_h), pygame.SRCALPHA)
                r = int(e.r * c * (0.3 + 0.9 * min(a * 3, 1)))
                pygame.draw.circle(overlay, (255, 170, 60, int(220 * (1 - a))),
                                   (int(e.x * c), int(e.y * c)), r)
                pygame.draw.circle(overlay, (255, 255, 200, int(255 * (1 - a))),
                                   (int(e.x * c), int(e.y * c)), r, 4)
            elif e.kind == "food":
                r = int((2 + a * 8) * c)
                pygame.draw.circle(surf, (120, 255, 140), (int(e.x * c), int(e.y * c)), r, 2)
            elif e.kind in ("ice", "heat"):
                if overlay is None:
                    overlay = pygame.Surface((self.world_w, self.world_h), pygame.SRCALPHA)
                col = (140, 190, 255) if e.kind == "ice" else (255, 120, 40)
                overlay.fill((*col, int(90 * (1 - a))), special_flags=0)
        self.effects = alive
        if overlay is not None:
            surf.blit(overlay, (0, 0))

    def _banner(self, surf):
        # new species / extinction toasts from the tracker's event stream
        events = self.sim.tracker.events
        for t, msg in events[self.seen_events:]:
            if t == 0:
                continue
            col = ACCENT if "split" in msg else (255, 140, 140) if "extinct" in msg else DIM
            self.toasts.append((msg, col, time.perf_counter()))
        self.seen_events = len(events)
        self.toasts = [x for x in self.toasts if time.perf_counter() - x[2] < 4.0][-5:]
        y = self.world_h - 28
        for msg, col, t0 in reversed(self.toasts):
            a = time.perf_counter() - t0
            alpha = int(255 * min(1, (4.0 - a)))
            txt = self.font_b.render(("* " if "split" in msg else "x ") + msg, True, col)
            txt.set_alpha(alpha)
            bg = pygame.Surface((txt.get_width() + 16, txt.get_height() + 6), pygame.SRCALPHA)
            bg.fill((0, 0, 0, int(alpha * 0.6)))
            surf.blit(bg, (10, y - 3))
            surf.blit(txt, (18, y))
            y -= 26
        if self.banner:
            text, col, t0 = self.banner
            a = (time.perf_counter() - t0) / 2.2
            if a >= 1:
                self.banner = None
            else:
                img = self.font_banner.render(text, True, col)
                img.set_alpha(int(255 * min(1, (1 - a) * 2.5)))
                shadow = self.font_banner.render(text, True, (0, 0, 0))
                shadow.set_alpha(int(160 * min(1, (1 - a) * 2.5)))
                x = (self.world_w - img.get_width()) // 2
                y = self.world_h // 2 - 60
                surf.blit(shadow, (x + 3, y + 3))
                surf.blit(img, (x, y))

    # --- side panel ----------------------------------------------------
    def _text(self, s, pos, font=None, col=TEXT):
        img = (font or self.font).render(s, True, col)
        self.screen.blit(img, pos)
        return img.get_height()

    def _section(self, title, y):
        self._text(title.upper(), (self.world_w + 16, y), self.font_s, DIM)
        pygame.draw.line(self.screen, (45, 50, 66), (self.world_w + 16, y + 17),
                         (self.world_w + PANEL_W - 16, y + 17))
        return y + 24

    def _graph(self, rect):
        sim = self.sim
        key = (len(sim.history_t), sim.seed, self.color_by_species, rect.size)
        if key != self.graph_key:
            self.graph_key = key
            surf = pygame.Surface(rect.size)
            surf.fill((24, 28, 40))
            hist = sim.history_species[-GRAPH_POINTS:]
            totals = np.array(sim.history_total[-GRAPH_POINTS:], float)
            m = len(hist)
            if m >= 2:
                ids = sorted({k for h in hist for k in h}, key=lambda k: sim.tracker.species[k].born
                             if k in sim.tracker.species else 0)
                mat = np.array([[h.get(k, 0) for k in ids] for h in hist], float)
                ymax = max(totals.max(), 200) * 1.05
                W, H = rect.size
                xs = np.linspace(0, W - 1, m)
                base = np.zeros(m)
                for j, k in enumerate(ids):
                    top = base + mat[:, j]
                    if mat[:, j].max() == 0:
                        continue
                    sp = sim.tracker.species.get(k)
                    col = sp.color if sp else (120, 120, 120)
                    pts = [(xs[i], H - 1 - top[i] / ymax * (H - 2)) for i in range(m)]
                    pts += [(xs[i], H - 1 - base[i] / ymax * (H - 2)) for i in range(m - 1, -1, -1)]
                    pygame.draw.polygon(surf, col, pts)
                    base = top
                # temperature line overlay
                temps = np.array(sim.history_temp[-GRAPH_POINTS:])
                tp = [(xs[i], H - 1 - temps[i] * (H - 2)) for i in range(m)]
                pygame.draw.lines(surf, (255, 255, 255), False, tp, 1)
                lbl = self.font_s.render(f"max {int(totals.max())}", True, DIM)
                surf.blit(lbl, (4, 2))
            self.graph_cache = surf
        self.screen.blit(self.graph_cache, rect.topleft)
        pygame.draw.rect(self.screen, (55, 60, 80), rect, 1)

    def _panel(self, fps, speed, paused):
        sim = self.sim
        x0 = self.world_w
        pygame.draw.rect(self.screen, PANEL_BG, (x0, 0, PANEL_W, self.screen.get_height()))
        x = x0 + 16
        y = 12
        self._text("NEW WORLDS", (x, y), self.font_title, ACCENT)
        status = "PAUSED" if paused else f"x{speed}"
        self._text(f"seed {sim.seed}   step {sim.t}   {status}   {fps:4.0f} fps",
                   (x, y + 34), self.font_s, DIM)
        y += 58

        # temperature gauge
        y = self._section("Climate", y)
        T = sim.world.temperature
        bar = pygame.Rect(x, y, PANEL_W - 32, 12)
        if getattr(self, "_gauge", None) is None:
            f = np.linspace(0, 1, bar.w)
            cols = hsv_array(0.62 - 0.62 * f, np.full(bar.w, 0.7), np.full(bar.w, 0.95))
            self._gauge = pygame.surfarray.make_surface(np.repeat(cols[:, None, :], bar.h, axis=1))
        self.screen.blit(self._gauge, bar.topleft)
        mx = bar.x + int(T * bar.w)
        pygame.draw.polygon(self.screen, (255, 255, 255), [(mx, bar.y - 2), (mx - 5, bar.y - 8), (mx + 5, bar.y - 8)])
        pygame.draw.line(self.screen, (255, 255, 255), (mx, bar.y - 2), (mx, bar.bottom + 2), 2)
        celsius = -15 + 60 * T
        mood = "ICE AGE" if T < 0.35 else "HEATWAVE" if T > 0.65 else "temperate"
        self._text(f"{celsius:+.1f} °C   {mood}   food {sim.world.food.sum():.0f}", (x, y + 16), self.font_s)
        y += 40

        # population graph
        n = len(sim.pop)
        hunters = int((sim.pop.genes[:, AGGR] > 0.5).sum()) if n else 0
        y = self._section(f"Population {n}  ({hunters} predators)", y)
        self._graph(pygame.Rect(x, y, PANEL_W - 32, 110))
        y += 118

        # species list
        living = sorted(sim.tracker.living(), key=lambda s: -s.count)
        y = self._section(f"Species  {len(living)} living / {len(sim.tracker.species)} ever", y)
        top = max([s.count for s in living] + [1])
        for sp in living[:11]:
            if sp.centroid[AGGR] > 0.5:      # predator species: white ring, as on the map
                pygame.draw.circle(self.screen, (255, 255, 255), (x + 6, y + 9), 8, 1)
            pygame.draw.circle(self.screen, sp.color, (x + 6, y + 9), 6)
            self._text(sp.name, (x + 18, y), self.font)
            cnt = self.font.render(str(sp.count), True, TEXT)
            self.screen.blit(cnt, (x0 + PANEL_W - 16 - cnt.get_width(), y))
            bw = int(70 * sp.count / top)
            pygame.draw.rect(self.screen, (*sp.color,), (x0 + PANEL_W - 100 - bw, y + 7, bw, 3))
            y += 19
        extinct = sorted([s for s in sim.tracker.species.values() if not s.alive],
                         key=lambda s: -s.extinct)[:2]
        for sp in extinct:
            self._text(f"† {sp.name}  (t={sp.extinct})", (x + 18, y), self.font_s, (110, 110, 125))
            y += 16
        y += 6

        # average genes
        y = self._section("Average genes", y)
        means = sim.mean_genes()
        for name in GENE_NAMES:
            v = means[name]
            self._text(name, (x, y), self.font_s, DIM)
            bx = x + 90
            bw = PANEL_W - 32 - 90 - 40
            pygame.draw.rect(self.screen, (40, 45, 60), (bx, y + 4, bw, 8), border_radius=4)
            col = hsv_array(np.array(v), np.array(0.7), np.array(1.0)).tolist() if name == "hue" else ACCENT
            pygame.draw.rect(self.screen, col, (bx, y + 4, max(2, int(bw * v)), 8), border_radius=4)
            self._text(f"{v:.2f}", (bx + bw + 6, y), self.font_s)
            y += 16
        y += 6

        # timeline of recent events
        bottom = self.screen.get_height() - 76
        if y < bottom - 60:
            y = self._section("Timeline", y)
            evs = sorted(sim.tracker.events + sim.event_log, key=lambda e: e[0])
            for t, msg in reversed(evs[-((bottom - y) // 16):]):
                if "split" in msg:
                    col = ACCENT
                elif "extinct" in msg or "merged" in msg:
                    col = (255, 150, 150)
                elif "founded" in msg:
                    col = DIM
                else:
                    col = (255, 200, 120)        # user events
                line = f"{t:>6}  {msg}"
                while self.font_s.size(line)[0] > PANEL_W - 32:
                    line = line[:-2]
                self._text(line, (x, y), self.font_s, col)
                y += 16

        # controls
        y = max(y, self.screen.get_height() - 70)
        for line in ("SPACE pause   UP/DOWN speed   V colour mode",
                     "C ice age   H heatwave   M meteor   click: food",
                     "R new world   P save report   ESC quit"):
            self._text(line, (x, y), self.font_s, DIM)
            y += 16

    # ------------------------------------------------------------------
    def draw(self, fps, speed, paused):
        self.screen.fill(BG)
        world = self._terrain()
        self._creatures(world)
        self._effects(world)
        self._banner(world)
        self.screen.blit(world, (0, 0))
        self._panel(fps, speed, paused)
        pygame.display.flip()
