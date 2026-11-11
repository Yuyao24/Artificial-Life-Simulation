import numpy as np

from newworlds import creatures as C
from newworlds.creatures import Population, repro_threshold, step_cost
from newworlds.world import World


def genes(speed=0.5, size=0.5, vision=0.5, metab=0.5, aggr=0.0, hue=0.3, n=1):
    return np.tile([speed, size, vision, metab, aggr, hue], (n, 1)).astype(float)


def make(n=0, seed=0):
    rng = np.random.default_rng(seed)
    world = World(40, 30, rng=rng)
    return world, Population(rng, world.w, world.h)


def test_moving_costs_more_for_faster_and_bigger():
    g = genes()
    assert step_cost(g, np.array([1.0]))[0] > step_cost(g, np.array([0.2]))[0]
    big = genes(size=1.0)
    small = genes(size=0.0)
    moved = np.array([0.8])
    assert step_cost(big, moved)[0] > step_cost(small, moved)[0]


def test_creature_dies_at_zero_energy():
    world, pop = make()
    world.food[:] = 0.0
    world.cap[:] = 0.0
    pop.add([10, 20], [10, 10], genes(n=2), energy=[0.001, 5.0])
    pop.step(world)
    assert len(pop) == 1
    assert pop.energy[0] < 5.0          # survivor paid its costs
    assert pop.deaths == 1


def test_starving_population_dies_out():
    world, pop = make()
    world.fertility[:] = 0.0
    world.update_cap()
    world.food[:] = 0.0
    pop.add(np.full(20, 20.0), np.full(20, 15.0), genes(n=20), energy=np.full(20, 1.0))
    for _ in range(300):
        world.step()
        pop.step(world)
    assert len(pop) == 0


def test_reproduction_splits_energy_and_inherits_species():
    world, pop = make()
    g = genes(size=0.5)
    thr = repro_threshold(g[:, C.SIZE])[0]
    pop.add([15.0], [15.0], g, energy=[thr + 1.0], species=[7])
    pop.reproduce()
    assert len(pop) == 2
    assert pop.births == 1
    assert np.allclose(pop.energy, (thr + 1.0) / 2)
    assert (pop.species == 7).all()
    assert not np.array_equal(pop.genes[0], pop.genes[1])   # child mutated


def test_population_cap_is_respected(monkeypatch):
    monkeypatch.setattr(C, "MAX_POP", 50)
    world, pop = make()
    g = genes(n=45)
    pop.add(np.full(45, 10.0), np.full(45, 10.0), g, energy=np.full(45, 100.0))
    pop.reproduce()
    assert len(pop) == 50


def test_predator_eats_smaller_prey():
    world, pop = make(seed=3)
    world.food[:] = 0.0
    hunter = genes(speed=0.8, size=1.0, vision=1.0, aggr=1.0)
    prey = genes(speed=0.0, size=0.0, vision=0.0, aggr=0.0)
    pop.add([20.0, 20.5], [15.0, 15.0], np.vstack([hunter, prey]), energy=[3.0, 3.0])
    for _ in range(20):
        pop.step(world)
        if pop.kills:
            break
    assert pop.kills == 1
    assert len(pop) == 1 and pop.genes[0, C.AGGR] == 1.0
