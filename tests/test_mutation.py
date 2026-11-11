import numpy as np

from newworlds.creatures import HUE, MUTATION_SIGMA, N_GENES, mutate


def test_mutation_keeps_genes_in_range():
    rng = np.random.default_rng(0)
    genes = rng.random((5000, N_GENES))
    genes[:10] = 0.0
    genes[10:20] = 1.0
    child = mutate(genes, rng)
    assert child.shape == genes.shape
    assert child.min() >= 0.0 and child.max() <= 1.0
    assert (child[:, HUE] < 1.0).all()          # hue wraps instead of clipping


def test_mutations_are_small_on_average():
    rng = np.random.default_rng(1)
    genes = np.full((20000, N_GENES), 0.5)
    child = mutate(genes, rng)
    diff = child[:, :HUE] - genes[:, :HUE]
    assert abs(diff.mean()) < 0.005                       # unbiased
    assert 0.5 * MUTATION_SIGMA < diff.std() < 2.5 * MUTATION_SIGMA


def test_no_mutation_when_sigma_zero():
    rng = np.random.default_rng(2)
    genes = rng.random((100, N_GENES))
    child = mutate(genes, rng, sigma=0.0, macro_rate=0.0)
    np.testing.assert_allclose(child[:, :HUE], genes[:, :HUE])


def test_macro_mutations_happen_at_expected_rate():
    rng = np.random.default_rng(3)
    genes = np.full((20000, N_GENES), 0.5)
    child = mutate(genes, rng, sigma=0.0, macro_rate=0.05)
    changed = (np.abs(child[:, :HUE] - 0.5) > 1e-9).any(1).mean()
    assert 0.035 < changed < 0.065


def test_mutation_is_reproducible_with_seed():
    genes = np.random.default_rng(4).random((50, N_GENES))
    a = mutate(genes, np.random.default_rng(42))
    b = mutate(genes, np.random.default_rng(42))
    np.testing.assert_array_equal(a, b)
