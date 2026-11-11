import numpy as np

from newworlds.creatures import N_GENES
from newworlds.species import SpeciesTracker, features, gene_mean, kmeans


def blob(center, n, rng, sd=0.02):
    g = np.clip(np.array(center) + rng.normal(0, sd, (n, N_GENES)), 0, 1)
    g[:, 5] %= 1.0
    return g


def test_kmeans_separates_two_blobs():
    rng = np.random.default_rng(0)
    a = blob([0.2, 0.2, 0.2, 0.2, 0.1, 0.3], 100, rng)
    b = blob([0.8, 0.8, 0.8, 0.8, 0.9, 0.3], 60, rng)
    labels, cent = kmeans(features(np.vstack([a, b])), 2, rng)
    assert len(set(labels[:100])) == 1 and len(set(labels[100:])) == 1
    assert labels[0] != labels[100]


def test_tracker_detects_new_species_with_parent():
    rng = np.random.default_rng(1)
    tr = SpeciesTracker(rng)
    a = blob([0.3, 0.3, 0.5, 0.5, 0.1, 0.3], 150, rng)
    root = tr.new_species(gene_mean(a), 0)
    b = blob([0.3, 0.8, 0.5, 0.5, 0.8, 0.3], 80, rng)   # big, aggressive offshoot
    genes = np.vstack([a, b])
    labels = np.full(len(genes), root.id)
    labels = tr.update(genes, labels, 100)
    assert len(tr.living()) == 2
    child = [s for s in tr.living() if s.id != root.id][0]
    assert child.parent == root.id
    assert (labels[150:] == child.id).all()
    assert (labels[:150] == root.id).all()
    assert child.name != root.name


def test_single_cluster_is_not_split():
    rng = np.random.default_rng(2)
    tr = SpeciesTracker(rng)
    g = blob([0.5] * 6, 400, rng, sd=0.05)
    sp = tr.new_species(gene_mean(g), 0)
    labels = tr.update(g, np.full(len(g), sp.id), 100)
    assert len(tr.species) == 1 and (labels == sp.id).all()


def test_species_goes_extinct_when_members_vanish():
    rng = np.random.default_rng(3)
    tr = SpeciesTracker(rng)
    a = blob([0.2] * 5 + [0.1], 50, rng)
    b = blob([0.8] * 5 + [0.6], 50, rng)
    sa = tr.new_species(gene_mean(a), 0)
    sb = tr.new_species(gene_mean(b), 0)
    tr.update(a, np.full(50, sa.id), 100)          # all of b's members died
    assert sa.alive and not sb.alive
    assert sb.extinct == 100
    assert any("extinct" in msg for _, msg in tr.events)


def test_converged_species_merge():
    rng = np.random.default_rng(4)
    tr = SpeciesTracker(rng)
    g = blob([0.5] * 6, 100, rng)
    s1 = tr.new_species(gene_mean(g), 0)
    s2 = tr.new_species(gene_mean(g), 50)
    labels = np.where(np.arange(100) < 50, s1.id, s2.id)
    labels = tr.update(g, labels, 100)
    assert (labels == s1.id).all()      # younger merged into older
    assert not s2.alive


def test_circular_hue_mean():
    g = np.zeros((2, N_GENES))
    g[:, 5] = [0.98, 0.04]
    assert abs(gene_mean(g)[5] - 0.01) < 1e-6
