import numpy as np

from newworlds.simulation import Simulation


def test_same_seed_gives_same_run():
    a, b = Simulation(seed=11, n_start=200), Simulation(seed=11, n_start=200)
    a.run(150)
    b.run(150)
    assert len(a.pop) == len(b.pop)
    np.testing.assert_array_equal(a.pop.genes, b.pop.genes)
    np.testing.assert_array_equal(a.world.food, b.world.food)


def test_events_change_the_world():
    sim = Simulation(seed=5, n_start=400)
    t0 = sim.world.temperature
    sim.ice_age()
    assert sim.world.temperature < t0
    sim.heatwave()
    sim.heatwave()
    assert sim.world.temperature > t0
    x, y, r, killed = sim.meteor(80, 50, 20)
    assert ((sim.pop.x - x) ** 2 + (sim.pop.y - y) ** 2 > r * r).all()
    assert sim.world.food[80, 50] == 0.0
    sim.drop_food(80, 50)
    assert sim.world.food[80, 50] > 1.0


def test_headless_run_records_history_and_report(tmp_path):
    sim = Simulation(seed=3, n_start=300)
    sim.run(300)
    assert len(sim.history_t) == 31
    assert sim.tracker.living()
    from newworlds.report import save_report
    out = save_report(sim, str(tmp_path / "r.png"))
    assert (tmp_path / "r.png").stat().st_size > 10_000
    assert out.endswith("r.png")
