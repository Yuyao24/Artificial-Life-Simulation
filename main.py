"""New Worlds - entry point.

    python main.py                 # interactive window
    python main.py --seed 42       # reproducible world
    python main.py --headless --steps 3000   # no window, just simulate + report
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from newworlds.simulation import Simulation


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="New Worlds artificial-life simulation")
    ap.add_argument("--seed", type=int, default=None, help="random seed for a reproducible run")
    ap.add_argument("--fullscreen", action="store_true", help="start fullscreen (F11 toggles)")
    ap.add_argument("--speed", type=int, default=2, help="initial speed level index (0-6)")
    ap.add_argument("--headless", action="store_true", help="run without a window")
    ap.add_argument("--steps", type=int, default=3000, help="steps for --headless")
    ap.add_argument("--frames", type=int, default=0,
                    help="quit after N frames and print the average FPS (benchmark)")
    ap.add_argument("--screenshot", default=None, help="save a screenshot on exit to this path")
    ap.add_argument("--no-report", action="store_true", help="don't write the PNG report on exit")
    return ap.parse_args(argv)


def run_headless(args) -> None:
    sim = Simulation(seed=args.seed)
    t0 = time.perf_counter()
    for i in range(args.steps):
        sim.step()
        if (i + 1) % 500 == 0:
            living = sorted(sim.tracker.living(), key=lambda s: -s.count)
            print(f"step {sim.t:6d}  pop {len(sim.pop):5d}  T {sim.world.temperature:.2f}  "
                  f"species {len(living):2d} alive / {len(sim.tracker.species):2d} ever  "
                  f"top: {', '.join(f'{s.name} ({s.count})' for s in living[:3])}")
    dt = time.perf_counter() - t0
    print(f"{args.steps} steps in {dt:.1f}s ({dt / max(args.steps, 1) * 1000:.2f} ms/step)")
    if not args.no_report:
        from newworlds.report import save_report
        print("report saved to", save_report(sim))


def run_interactive(args) -> None:
    import pygame

    from newworlds.render import SPEEDS, Renderer

    sim = Simulation(seed=args.seed)
    ren = Renderer(sim, fullscreen=args.fullscreen)
    ren.show_banner(f"New world  ·  seed {sim.seed}", (110, 231, 183))
    clock = pygame.time.Clock()
    speed_i = max(0, min(args.speed, len(SPEEDS) - 1))
    paused = False
    frames = 0
    fps_samples = []
    running = True
    mouse_down = False

    def save():
        from newworlds.report import save_report
        path = save_report(sim)
        print("report saved to", path)
        ren.toasts.append((f"Report saved: {path}", (200, 200, 255), time.perf_counter()))

    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                k = ev.key
                if k in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif k == pygame.K_SPACE:
                    paused = not paused
                elif k == pygame.K_UP:
                    speed_i = min(speed_i + 1, len(SPEEDS) - 1)
                elif k == pygame.K_DOWN:
                    speed_i = max(speed_i - 1, 0)
                elif k == pygame.K_c:
                    sim.ice_age()
                    ren.show_banner("Ice age", (147, 197, 253))
                    ren.add_effect("ice", 0, 0, duration=2.5)
                elif k == pygame.K_h:
                    sim.heatwave()
                    ren.show_banner("Heatwave", (255, 160, 90))
                    ren.add_effect("heat", 0, 0, duration=2.5)
                elif k == pygame.K_m:
                    mx, my = pygame.mouse.get_pos()
                    if ren.in_world(mx, my) and pygame.mouse.get_focused():
                        x, y, r, killed = sim.meteor(*ren.to_world(mx, my))
                    else:
                        x, y, r, killed = sim.meteor()
                    ren.add_effect("meteor", x, y, r, duration=1.6)
                    ren.show_banner(f"Meteor strike  ·  {killed} lost", (255, 190, 120))
                elif k == pygame.K_F11:
                    pygame.display.toggle_fullscreen()
                elif k == pygame.K_v:
                    ren.color_by_species = not ren.color_by_species
                elif k == pygame.K_r:
                    sim = Simulation(seed=int(np.random.SeedSequence().entropy % 1_000_000))
                    ren.reset(sim)
                elif k == pygame.K_p:
                    save()
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                mouse_down = True
            elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
                mouse_down = False

        if mouse_down:
            mx, my = pygame.mouse.get_pos()
            if ren.in_world(mx, my):
                wx, wy = ren.to_world(mx, my)
                sim.drop_food(wx, wy)
                if frames % 6 == 0:
                    ren.add_effect("food", wx, wy, duration=0.6)

        if not paused:
            for _ in range(SPEEDS[speed_i]):
                sim.step()

        fps = clock.get_fps()
        ren.draw(fps, SPEEDS[speed_i], paused)
        clock.tick(60)
        frames += 1
        if frames > 30:
            fps_samples.append(fps)
        if args.frames and frames >= args.frames:
            running = False

    if args.screenshot:
        pygame.image.save(ren.screen, args.screenshot)
    if args.frames:
        print(f"average FPS {np.mean(fps_samples):.1f} with {len(sim.pop)} creatures "
              f"at speed x{SPEEDS[speed_i]} (step {sim.t})")
    if not args.no_report:
        save()
    pygame.quit()


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.headless:
        run_headless(args)
    else:
        run_interactive(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
