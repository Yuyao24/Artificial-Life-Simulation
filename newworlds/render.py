"""pygame-ce front end. All drawing lives here; the simulation knows nothing about it.

Everything is drawn on a fixed 1920x1080 canvas. pygame's SCALED mode letterboxes it
(black bars) to whatever window or screen size is used, and maps mouse
coordinates back to canvas pixels for us.
"""
from __future__ import annotations

import colorsys
import time
from pathlib import Path

import numpy as np
import pygame

from .creatures import AGGR, GENE_NAMES, HUE, SIZE
from .simulation import Simulation

SCREEN_W, SCREEN_H = 1920, 1080
MARGIN = 28
CELL = 9                                   # 160 x 100 cells -> 1440 x 900 px
PANEL_GAP = 28
SPEEDS = [1, 2, 3, 5, 8, 12, 20]
GRAPH_POINTS = 300                          # history samples shown (x HISTORY_EVERY steps)

# palette ---------------------------------------------------------------
BG = (7, 9, 14)
CARD = (16, 20, 30)
CARD_BORDER = (32, 39, 56)
TEXT = (232, 236, 244)
MUTED = (138, 148, 168)
FAINT = (82, 92, 112)
ACCENT = (110, 231, 183)                   # mint
WARN = (251, 146, 120)                     # coral
GOLD = (252, 211, 120)
ICE = (147, 197, 253)

FONT_DIR = Path(__file__).parent / "assets" / "fonts"


def _font(name: str, size: int, fallback_bold: bool = False) -> pygame.font.Font:
    path = FONT_DIR / name
    if path.exists():
        return pygame.font.Font(str(path), size)
    return pygame.font.SysFont("inter,segoeui,helveticaneue,arial", size, bold=fallback_bold)


def hsv_array(h, s, v):
    """Vectorised HSV -> RGB (uint8). h, s, v are arrays in [0, 1]."""
    h6 = (np.asarray(h) % 1.0) * 6.0
    i = np.floor(h6).astype(int) % 6
    f = h6 - np.floor(h6)
    p = v * (1 - s)
    q = v * (1 - s * f)
    t = v * (1 - s * (1 - f))
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


def hue_rgb(h, s=0.62, v=1.0):
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, s, v)
    return int(r * 255), int(g * 255), int(b * 255)


def mix(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


class Effect:
    def __init__(self, kind, x, y, r, duration, data=None):
        self.kind, self.x, self.y, self.r, self.data = kind, x, y, r, data
        self.t0 = time.perf_counter()
        self.duration = duration

    @property
    def age(self):
        return (time.perf_counter() - self.t0) / self.duration


# --- creature sprites ----------------------------------------------------
HUE_BUCKETS = 64
MIN_R, MAX_R = 3, 11


def make_sprite(rgb, r: int, predator: bool) -> pygame.Surface:
    """Anti-aliased glowing disc, drawn 4x oversized and scaled down."""
    S = 4
    glow = r * 1.1 + 4
    size = int(2 * (r + glow + 2))
    big = pygame.Surface((size * S, size * S), pygame.SRCALPHA)
    c = size * S // 2
    steps = 14
    for i in range(steps):                         # soft outer glow
        f = i / steps
        rad = (r + glow) * S * (1 - f) + r * S * f
        a = int(46 * f ** 2.2)
        pygame.draw.circle(big, (*rgb, a), (c, c), int(rad))
    if predator:
        pygame.draw.circle(big, (255, 255, 255, 235), (c, c), int((r + 2.4) * S), int(1.3 * S))
    pygame.draw.circle(big, (*mix(rgb, (0, 0, 0), 0.45), 255), (c, c), int(r * S))
    pygame.draw.circle(big, (*rgb, 255), (c, c), int((r - 0.9) * S))
    hl = mix(rgb, (255, 255, 255), 0.55)
    pygame.draw.circle(big, (*hl, 255), (c - int(r * S * 0.28), c - int(r * S * 0.28)),
                       max(1, int(r * S * 0.38)))
    return pygame.transform.smoothscale(big, (size, size))


class Renderer:
    def __init__(self, sim: Simulation, fullscreen: bool = False):
        pygame.display.init()
        pygame.font.init()
        pygame.display.set_caption("New Worlds")
        flags = pygame.SCALED | pygame.RESIZABLE
        if fullscreen:
            flags |= pygame.FULLSCREEN
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H), flags, vsync=0)
        self.sim = sim
        self.cell = CELL
        self.world_w = sim.world.w * CELL
        self.world_h = sim.world.h * CELL
        self.wx, self.wy = MARGIN, MARGIN
        self.world_rect = pygame.Rect(self.wx, self.wy, self.world_w, self.world_h)
        self.panel_x = self.wx + self.world_w + PANEL_GAP
        self.panel_w = SCREEN_W - self.panel_x - MARGIN

        self.f_title = _font("InterDisplay-Bold.otf", 34, True)
        self.f_banner = _font("InterDisplay-Bold.otf", 76, True)
        self.f_big = _font("Inter-SemiBold.otf", 30, True)
        self.f_head = _font("Inter-SemiBold.otf", 12, True)
        self.f_body = _font("Inter-Medium.otf", 15)
        self.f_small = _font("Inter-Regular.otf", 13)
        self.f_toast = _font("Inter-SemiBold.otf", 15, True)
        self.f_mono = _font("JetBrainsMono-Medium.ttf", 14)
        self.f_mono_s = _font("JetBrainsMono-Regular.ttf", 12)
        self.f_key = _font("Inter-SemiBold.otf", 12, True)

        self.effects: list[Effect] = []
        self.banner = None
        self.toasts: list[tuple[str, tuple, float]] = []
        self.seen_events = 0
        self.color_by_species = False
        self.graph_cache = None
        self.graph_key = None
        self.terrain_cache = None
        self.terrain_frame = 0
        self.text_cache: dict = {}
        self.sprites: list = [None] * (HUE_BUCKETS * (MAX_R + 1) * 2)
        self.world_surf = pygame.Surface((self.world_w, self.world_h)).convert()
        self._build_overlay()
        self._gauge = None

    def reset(self, sim: Simulation):
        self.sim = sim
        self.effects.clear()
        self.toasts.clear()
        self.seen_events = 0
        self.graph_key = None
        self.terrain_cache = None
        self.show_banner(f"New world  ·  seed {sim.seed}", ACCENT)

    # ------------------------------------------------------------------
    def show_banner(self, text, colour):
        self.banner = (text, colour, time.perf_counter())

    def add_effect(self, kind, x, y, r=0.0, duration=1.0, data=None):
        self.effects.append(Effect(kind, x, y, r, duration, data))

    def in_world(self, px, py) -> bool:
        return self.world_rect.collidepoint(px, py)

    def to_world(self, px, py):
        return (px - self.wx) / CELL, (py - self.wy) / CELL

    def _txt(self, s, font, col):
        key = (s, id(font), col)
        img = self.text_cache.get(key)
        if img is None:
            if len(self.text_cache) > 2000:
                self.text_cache.clear()
            img = font.render(s, True, col)
            self.text_cache[key] = img
        return img

    def _blit_text(self, surf, s, font, col, pos, anchor="topleft"):
        img = self._txt(s, font, col)
        r = img.get_rect(**{anchor: pos})
        surf.blit(img, r)
        return r

    # --- static overlay: dot grid, vignette, rounded corners -------------
    def _build_overlay(self):
        w, h = self.world_w, self.world_h
        xs = np.arange(w)[:, None]
        ys = np.arange(h)[None, :]
        # vignette
        dx = (xs - w / 2) / (w / 2)
        dy = (ys - h / 2) / (h / 2)
        d = np.sqrt(dx ** 2 * 0.8 + dy ** 2 * 0.8)
        alpha = np.clip((d - 0.55) / 0.75, 0, 1) ** 1.6 * 150
        rgb = np.zeros((w, h, 3), np.uint8)
        # faint dot grid
        grid = ((xs % 36) == 18) & ((ys % 36) == 18)
        rgb[grid] = (255, 255, 255)
        alpha = np.where(grid, 22, alpha)
        # rounded corners filled with the page background
        R = 18
        cx = np.clip(xs, R, w - 1 - R)
        cy = np.clip(ys, R, h - 1 - R)
        outside = (xs - cx) ** 2 + (ys - cy) ** 2 > R * R
        rgb[outside] = BG
        alpha = np.where(outside, 255, alpha)
        surf = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.surfarray.pixels3d(surf)[:] = rgb
        pygame.surfarray.pixels_alpha(surf)[:] = alpha.astype(np.uint8)
        self.overlay = surf.convert_alpha()

    # --- terrain --------------------------------------------------------
    def _terrain(self) -> pygame.Surface:
        self.terrain_frame += 1
        if self.terrain_cache is not None and self.terrain_frame % 6:
            return self.terrain_cache
        w = self.sim.world
        lt = np.clip(w.local_temperature(), -0.3, 1.3)
        cold = np.clip((0.38 - lt) / 0.4, 0, 1)[..., None]
        hot = np.clip((lt - 0.62) / 0.4, 0, 1)[..., None]
        mild = 1 - cold - hot
        base = (np.array([17, 24, 30]) * mild + np.array([46, 66, 98]) * cold
                + np.array([72, 50, 30]) * hot)
        f = np.clip(w.food / 0.8, 0, 1.4)
        fm = np.minimum(f, 1.0)[..., None] ** 0.85
        rgb = base * (1 - fm * 0.55) + np.array([38, 168, 118]) * fm * 0.8
        rgb += np.array([90, 110, 40]) * np.maximum(f - 1, 0)[..., None]
        rgb *= (0.86 + 0.22 * w.fertility[..., None])
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        small = pygame.surfarray.make_surface(rgb)
        mid = pygame.transform.smoothscale(small, (w.w * 2, w.h * 2))
        mid = pygame.transform.gaussian_blur(mid, 2)
        self.terrain_cache = pygame.transform.smoothscale(mid, (self.world_w, self.world_h)).convert()
        return self.terrain_cache

    # --- creatures ------------------------------------------------------
    def _sprite(self, key, hb, r, pred):
        spr = self.sprites[key]
        if spr is None:
            col = hue_rgb((hb + 0.5) / HUE_BUCKETS)
            spr = make_sprite(col, r, pred)
            self.sprites[key] = (spr, spr.get_width() // 2)
            spr = self.sprites[key]
        return spr

    def _creature_arrays(self):
        p = self.sim.pop
        if self.color_by_species:
            lut = {sp.id: sp.hue for sp in self.sim.tracker.species.values()}
            hue = np.array([lut.get(int(s), 0.0) for s in p.species])
        else:
            hue = p.genes[:, HUE]
        hb = (hue * HUE_BUCKETS).astype(int) % HUE_BUCKETS
        r = np.clip(np.round(MIN_R + p.genes[:, SIZE] * (MAX_R - MIN_R)), MIN_R, MAX_R).astype(int)
        pred = (p.genes[:, AGGR] > 0.5).astype(int)
        return hb, r, pred

    def _creatures(self, surf, hb, r, pred):
        p = self.sim.pop
        if len(p) == 0:
            return
        xs = (p.x * CELL).astype(int).tolist()
        ys = (p.y * CELL).astype(int).tolist()
        keys = ((hb * (MAX_R + 1) + r) * 2 + pred).tolist()
        hbl, rl, pl = hb.tolist(), r.tolist(), pred.tolist()
        sprites = self.sprites
        seq = []
        for i, k in enumerate(keys):
            s = sprites[k] or self._sprite(k, hbl[i], rl[i], pl[i])
            seq.append((s[0], (xs[i] - s[1], ys[i] - s[1])))
        surf.fblits(seq)

    # --- effects --------------------------------------------------------
    def _effects(self, surf):
        alive = []
        for e in self.effects:
            a = e.age
            if a >= 1:
                continue
            alive.append(e)
            if e.kind == "meteor":
                cx, cy = int(e.x * CELL), int(e.y * CELL)
                R = e.r * CELL
                fl = pygame.Surface((self.world_w, self.world_h), pygame.SRCALPHA)
                if a < 0.45:
                    k = 1 - a / 0.45
                    for i in range(10):                     # radial fireball glow
                        f = i / 10
                        pygame.draw.circle(fl, (255, int(150 + 100 * f), int(80 + 150 * f),
                                                int(255 * k * (0.15 + 0.85 * f) ** 2)),
                                           (cx, cy), int(R * (1.15 - f)))
                for k, lag in enumerate((0.0, 0.12, 0.24)):
                    t = a - lag
                    if 0 < t < 1:
                        rad = int(R * (0.2 + 1.6 * t))
                        alpha = int(230 * (1 - t) ** 1.5)
                        pygame.draw.circle(fl, (255, 180, 110, alpha), (cx, cy), rad, max(1, 4 - k))
                surf.blit(fl, (0, 0))
            elif e.kind == "food":
                rad = int((1.5 + a * 6) * CELL)
                fl = pygame.Surface((rad * 2 + 4, rad * 2 + 4), pygame.SRCALPHA)
                pygame.draw.circle(fl, (*ACCENT, int(200 * (1 - a))), (rad + 2, rad + 2), rad, 2)
                surf.blit(fl, (int(e.x * CELL) - rad - 2, int(e.y * CELL) - rad - 2))
            elif e.kind in ("ice", "heat"):
                col = ICE if e.kind == "ice" else (255, 140, 60)
                fl = pygame.Surface((self.world_w, self.world_h), pygame.SRCALPHA)
                fl.fill((*col, int(70 * (1 - a) ** 1.3)))
                surf.blit(fl, (0, 0))
            elif e.kind == "species":
                # pulse rings around the members of a brand-new species
                sid, col = e.data
                m = self.sim.pop.species == sid
                if m.any():
                    xs = (self.sim.pop.x[m] * CELL).astype(int)[:150]
                    ys = (self.sim.pop.y[m] * CELL).astype(int)[:150]
                    rad = int(8 + 22 * a)
                    alpha = int(220 * (1 - a))
                    ring = pygame.Surface((rad * 2 + 4, rad * 2 + 4), pygame.SRCALPHA)
                    pygame.draw.circle(ring, (*col, alpha), (rad + 2, rad + 2), rad, 2)
                    surf.fblits([(ring, (x - rad - 2, y - rad - 2)) for x, y in zip(xs.tolist(), ys.tolist())])
        self.effects = alive

    def _pill(self, surf, rect, fill, alpha, border=None, radius=None):
        s = pygame.Surface(rect.size, pygame.SRCALPHA)
        rr = rect.h // 2 if radius is None else radius
        pygame.draw.rect(s, (*fill, alpha), s.get_rect(), border_radius=rr)
        if border:
            pygame.draw.rect(s, (*border, min(255, alpha + 40)), s.get_rect(), 1, border_radius=rr)
        surf.blit(s, rect.topleft)

    def _toasts(self, surf):
        events = self.sim.tracker.events
        now = time.perf_counter()
        for t, msg in events[self.seen_events:]:
            if t == 0:
                continue
            if "split" in msg:
                col = ACCENT
                name = msg.split(" split from ")[0]
                sp = next((s for s in self.sim.tracker.species.values() if s.name == name), None)
                if sp is not None:
                    self.add_effect("species", 0, 0, duration=1.8, data=(sp.id, sp.color))
                msg = "New species · " + msg
            elif "extinct" in msg or "merged" in msg:
                col = WARN
            else:
                col = MUTED
            self.toasts.append((msg, col, now))
        self.seen_events = len(events)
        self.toasts = [x for x in self.toasts if now - x[2] < 5.0][-5:]
        y = 18
        for msg, col, t0 in self.toasts:
            a = now - t0
            fade = min(1.0, a * 5, (5.0 - a) * 1.5)
            img = self._txt(msg, self.f_toast, TEXT)
            rect = pygame.Rect(18, y, img.get_width() + 44, 34)
            self._pill(surf, rect, (10, 13, 20), int(200 * fade), border=CARD_BORDER)
            pygame.draw.circle(surf, col, (rect.x + 17, rect.centery), 5)
            img.set_alpha(int(255 * fade))
            surf.blit(img, img.get_rect(midleft=(rect.x + 30, rect.centery)))
            y += 42

    def _banner(self, surf):
        if not self.banner:
            return
        text, col, t0 = self.banner
        a = (time.perf_counter() - t0) / 2.4
        if a >= 1:
            self.banner = None
            return
        fade = min(1.0, a * 8, (1 - a) * 3)
        img = self.f_banner.render(text, True, col)
        rect = pygame.Rect(0, 0, img.get_width() + 96, img.get_height() + 36)
        rect.center = (self.world_w // 2, self.world_h // 2 - 40 - int(10 * (1 - min(1, a * 6))))
        self._pill(surf, rect, (8, 10, 16), int(185 * fade), border=col, radius=28)
        img.set_alpha(int(255 * fade))
        surf.blit(img, img.get_rect(center=rect.center))

    # --- side panel ----------------------------------------------------
    def _card(self, y, h, title, right=None):
        rect = pygame.Rect(self.panel_x, y, self.panel_w, h)
        pygame.draw.rect(self.screen, CARD, rect, border_radius=16)
        pygame.draw.rect(self.screen, CARD_BORDER, rect, 1, border_radius=16)
        self._blit_text(self.screen, title.upper(), self.f_head, MUTED, (rect.x + 20, rect.y + 16))
        if right:
            self._blit_text(self.screen, right, self.f_head, FAINT, (rect.right - 20, rect.y + 16), "topright")
        return rect

    def _graph(self, rect):
        sim = self.sim
        key = (len(sim.history_t), sim.seed, self.color_by_species, rect.size)
        if key != self.graph_key:
            self.graph_key = key
            W, H = rect.size
            surf = pygame.Surface(rect.size, pygame.SRCALPHA)
            hist = sim.history_species[-GRAPH_POINTS:]
            totals = np.array(sim.history_total[-GRAPH_POINTS:], float)
            m = len(hist)
            ymax = max(totals.max() if m else 0, 200) * 1.08
            for frac in (0.25, 0.5, 0.75):
                yy = int(H - frac * H)
                pygame.draw.line(surf, (*CARD_BORDER, 255), (0, yy), (W, yy))
            if m >= 2:
                ids = sorted({k for h in hist for k in h},
                             key=lambda k: sim.tracker.species[k].born if k in sim.tracker.species else 0)
                mat = np.array([[h.get(k, 0) for k in ids] for h in hist], float)
                xs = np.linspace(0, W - 1, m)
                base = np.zeros(m)
                tops = []
                for j, k in enumerate(ids):
                    if mat[:, j].max() == 0:
                        continue
                    top = base + mat[:, j]
                    sp = sim.tracker.species.get(k)
                    col = sp.color if sp else (120, 120, 120)
                    yt = H - 1 - top / ymax * (H - 2)
                    yb = H - 1 - base / ymax * (H - 2)
                    pts = list(zip(xs, yt)) + list(zip(xs[::-1], yb[::-1]))
                    pygame.draw.polygon(surf, (*col, 120), pts)
                    tops.append((col, list(zip(xs, yt))))
                    base = top
                for col, pts in tops:
                    pygame.draw.aalines(surf, col, False, pts)
                temps = np.array(sim.history_temp[-GRAPH_POINTS:])
                tp = list(zip(xs, H - 1 - temps * (H - 2)))
                pygame.draw.aalines(surf, (255, 255, 255), False, tp)
            self.graph_cache = surf
        self.screen.blit(self.graph_cache, rect.topleft)

    def _panel(self, fps, speed, paused):
        sim = self.sim
        x = self.panel_x
        y = MARGIN

        # header
        self._blit_text(self.screen, "New Worlds", self.f_title, TEXT, (x + 2, y - 4))
        chips = [f"seed {sim.seed}", f"step {sim.t:,}", "paused" if paused else f"{speed}×",
                 f"{fps:.0f} fps"]
        cx = x + 2
        cy = y + 46
        for i, c in enumerate(chips):
            img = self._txt(c, self.f_mono_s, ACCENT if (i == 2 and not paused) else
                            GOLD if (i == 2 and paused) else MUTED)
            r = pygame.Rect(cx, cy, img.get_width() + 20, 24)
            pygame.draw.rect(self.screen, CARD, r, border_radius=12)
            pygame.draw.rect(self.screen, CARD_BORDER, r, 1, border_radius=12)
            self.screen.blit(img, img.get_rect(center=r.center))
            cx = r.right + 8
        y += 86

        # climate
        T = sim.world.temperature
        rect = self._card(y, 104, "Climate")
        celsius = -15 + 60 * T
        self._blit_text(self.screen, f"{celsius:+.1f}°C", self.f_big, TEXT, (rect.x + 20, rect.y + 36))
        mood, mcol = (("Ice age", ICE) if T < 0.35 else ("Heatwave", (255, 160, 90)) if T > 0.65
                      else ("Temperate", ACCENT))
        img = self._txt(mood, self.f_key, mcol)
        pr = pygame.Rect(0, 0, img.get_width() + 22, 24)
        pr.topright = (rect.right - 20, rect.y + 44)
        self._pill(self.screen, pr, mcol, 38)
        self.screen.blit(img, img.get_rect(center=pr.center))
        bar = pygame.Rect(rect.x + 20, rect.y + 82, rect.w - 40, 6)
        if self._gauge is None:
            f = np.linspace(0, 1, bar.w)
            cols = hsv_array(0.6 - 0.6 * f, np.full(bar.w, 0.6), np.full(bar.w, 0.95))
            g = pygame.Surface(bar.size, pygame.SRCALPHA)
            g.fill((0, 0, 0, 255))
            pygame.surfarray.pixels3d(g)[:] = np.repeat(cols[:, None, :], bar.h, axis=1)
            mask = pygame.Surface(bar.size, pygame.SRCALPHA)
            pygame.draw.rect(mask, (255, 255, 255, 255), mask.get_rect(), border_radius=3)
            g.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
            self._gauge = g
        self.screen.blit(self._gauge, bar.topleft)
        mx = bar.x + int(np.clip(T, 0, 1) * bar.w)
        pygame.draw.circle(self.screen, BG, (mx, bar.centery), 9)
        pygame.draw.circle(self.screen, TEXT, (mx, bar.centery), 7)
        y = rect.bottom + 14

        # population
        n = len(sim.pop)
        hunters = int((sim.pop.genes[:, AGGR] > 0.5).sum()) if n else 0
        rect = self._card(y, 196, "Population", f"{hunters} predators")
        self._blit_text(self.screen, f"{n:,}", self.f_big, TEXT, (rect.x + 20, rect.y + 34))
        self._graph(pygame.Rect(rect.x + 20, rect.y + 78, rect.w - 40, 102))
        y = rect.bottom + 14

        # species
        living = sorted(sim.tracker.living(), key=lambda s: -s.count)
        extinct = sorted([s for s in sim.tracker.species.values() if not s.alive],
                         key=lambda s: -s.extinct)[:2]
        rows = living[:9]
        h = 52 + 26 * len(rows) + (8 + 20 * len(extinct) if extinct else 0) + 6
        rect = self._card(y, h, "Species", f"{len(living)} living · {len(sim.tracker.species)} ever")
        top = max([s.count for s in living] + [1])
        ry = rect.y + 46
        for sp in rows:
            cyy = ry + 10
            if sp.centroid[AGGR] > 0.5:
                pygame.draw.circle(self.screen, (255, 255, 255), (rect.x + 26, cyy), 8, 1)
            pygame.draw.circle(self.screen, sp.color, (rect.x + 26, cyy), 5)
            self._blit_text(self.screen, sp.name, self.f_body, TEXT, (rect.x + 44, cyy), "midleft")
            self._blit_text(self.screen, f"{sp.count:,}", self.f_mono, MUTED, (rect.right - 20, cyy), "midright")
            bw = max(3, int(64 * sp.count / top))
            br = pygame.Rect(rect.right - 76 - 64, cyy - 2, 64, 4)
            pygame.draw.rect(self.screen, CARD_BORDER, br, border_radius=2)
            pygame.draw.rect(self.screen, sp.color, (br.x, br.y, bw, 4), border_radius=2)
            ry += 26
        if extinct:
            ry += 6
            for sp in extinct:
                self._blit_text(self.screen, f"† {sp.name}", self.f_small, FAINT, (rect.x + 20, ry))
                self._blit_text(self.screen, f"t {sp.extinct:,}", self.f_mono_s, FAINT,
                                (rect.right - 20, ry + 1), "topright")
                ry += 20
        y = rect.bottom + 14

        # average genes
        rect = self._card(y, 52 + 6 * 22, "Average genes")
        means = sim.mean_genes()
        ry = rect.y + 46
        for name in GENE_NAMES:
            v = means[name]
            self._blit_text(self.screen, name.capitalize(), self.f_small, MUTED, (rect.x + 20, ry + 8), "midleft")
            br = pygame.Rect(rect.x + 120, ry + 5, rect.w - 120 - 70, 6)
            pygame.draw.rect(self.screen, CARD_BORDER, br, border_radius=3)
            col = hue_rgb(v) if name == "hue" else ACCENT
            pygame.draw.rect(self.screen, col, (br.x, br.y, max(6, int(br.w * v)), 6), border_radius=3)
            self._blit_text(self.screen, f"{v:.2f}", self.f_mono_s, TEXT, (rect.right - 20, ry + 8), "midright")
            ry += 22
        y = rect.bottom + 14

        # timeline fills whatever room is left
        bottom = SCREEN_H - MARGIN
        if bottom - y > 80:
            rect = self._card(y, bottom - y, "Timeline")
            evs = sorted(sim.tracker.events + sim.event_log, key=lambda e: e[0])
            room = (rect.h - 50) // 21
            ry = rect.y + 44
            for t, msg in reversed(evs[-room:]):
                if "split" in msg:
                    col = ACCENT
                elif "extinct" in msg or "merged" in msg:
                    col = WARN
                elif "founded" in msg:
                    col = FAINT
                else:
                    col = GOLD
                pygame.draw.circle(self.screen, col, (rect.x + 24, ry + 9), 3)
                self._blit_text(self.screen, f"{t:>6,}", self.f_mono_s, FAINT, (rect.x + 34, ry + 2))
                line = msg
                maxw = rect.w - 110
                while self.f_small.size(line)[0] > maxw and len(line) > 4:
                    line = line[:-2]
                if line != msg:
                    line = line.rstrip() + "…"
                self._blit_text(self.screen, line, self.f_small, TEXT if col != FAINT else MUTED,
                                (rect.x + 90, ry + 1))
                ry += 21

    def _controls(self):
        """Key-cap hints in the strip under the map."""
        y = self.wy + self.world_h + 22
        x = self.wx + 4
        items = [("Space", "Pause"), ("↑ ↓", "Speed"), ("C", "Ice age"), ("H", "Heatwave"),
                 ("M", "Meteor"), ("Click", "Food"), ("V", "Colours"), ("R", "New world"),
                 ("P", "Report"), ("F11", "Fullscreen"), ("Esc", "Quit")]
        for key, label in items:
            img = self._txt(key, self.f_key, TEXT)
            kr = pygame.Rect(x, y, max(28, img.get_width() + 18), 28)
            pygame.draw.rect(self.screen, (24, 29, 42), kr, border_radius=7)
            pygame.draw.rect(self.screen, CARD_BORDER, kr, 1, border_radius=7)
            pygame.draw.line(self.screen, (40, 48, 66), (kr.x + 6, kr.bottom - 1), (kr.right - 7, kr.bottom - 1))
            self.screen.blit(img, img.get_rect(center=kr.center))
            lr = self._blit_text(self.screen, label, self.f_small, MUTED, (kr.right + 8, kr.centery), "midleft")
            x = lr.right + 22
        legend = "White ring = predator   ·   bright green = food"
        self._blit_text(self.screen, legend, self.f_small, FAINT,
                        (self.wx + self.world_w - 4, y + 52), "topright")

    # ------------------------------------------------------------------
    def draw(self, fps, speed, paused):
        self.screen.fill(BG)
        world = self.world_surf
        world.blit(self._terrain(), (0, 0))
        hb, r, pred = self._creature_arrays()
        self._creatures(world, hb, r, pred)
        self._effects(world)
        world.blit(self.overlay, (0, 0))
        self._toasts(world)
        self._banner(world)
        self.screen.blit(world, self.world_rect.topleft)
        pygame.draw.rect(self.screen, CARD_BORDER, self.world_rect.inflate(2, 2), 1, border_radius=19)
        self._panel(fps, speed, paused)
        self._controls()
        pygame.display.flip()
