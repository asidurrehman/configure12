"""Tests for the NSGA-II building blocks in configure12_functions.py."""
import numpy as np
import pytest

# Three ponds with 7, 6 and 5 size steps -> size chromosome of 18 genes
GENE_START = np.array([0, 7, 13])
GENE_END = np.array([7, 13, 18])
N_PONDS = 3
SIZE_LEN = 18

# Two objectives (cost, damage) with three clear fronts:
#   front 0: (1, 5) (2, 3) (3, 1)   front 1: (2, 4) (3, 3)   front 2: (4, 4)
OBJS = np.array([[1, 5], [2, 3], [3, 1], [2, 4], [3, 3], [4, 4]], dtype=float)


def block_sums(size_chrom):
    return np.array([size_chrom[s:e].sum() for s, e in zip(GENE_START, GENE_END)])


# --- initial population ------------------------------------------------------

def test_initial_pop_shapes_and_values(cf):
    np.random.seed(0)
    locat, size, size_val = cf.initial_pop(10, N_PONDS, SIZE_LEN, GENE_START, GENE_END)

    assert locat.shape == (10, N_PONDS)
    assert size.shape == (10, SIZE_LEN)
    assert size_val.shape == (10, N_PONDS)
    assert set(np.unique(locat)) <= {0, 1}
    assert set(np.unique(size)) <= {0, 1}

    # Built-in baselines: no ponds at all, and every pond at full size
    assert not locat[-2].any()
    assert locat[-1].all() and size[-1].all()
    assert np.array_equal(size_val[-1], GENE_END - GENE_START)

    for i in range(10):
        # size values are the number of size steps switched on per pond
        assert np.array_equal(size_val[i], block_sums(size[i]))
        # ponds that are switched off have no size steps
        for j in range(N_PONDS):
            if locat[i, j] == 0:
                assert not size[i, GENE_START[j]:GENE_END[j]].any()


def test_initial_pop_is_reproducible(cf):
    np.random.seed(42)
    a = cf.initial_pop(6, N_PONDS, SIZE_LEN, GENE_START, GENE_END)
    np.random.seed(42)
    b = cf.initial_pop(6, N_PONDS, SIZE_LEN, GENE_START, GENE_END)
    for x, y in zip(a, b):
        assert np.array_equal(x, y)


# --- duplicate handling ------------------------------------------------------

def test_remove_duplicate_same_population(cf):
    pop = np.array([[0, 1, 1], [1, 0, 0], [0, 1, 1], [1, 0, 0], [1, 1, 1]])
    uniq, removed = cf.remove_duplicate_same_population(pop)
    assert np.array_equal(uniq, [[0, 1, 1], [1, 0, 0], [1, 1, 1]])
    assert sorted(removed) == [2, 3]
    # the input is not modified
    assert len(pop) == 5


def test_remove_duplicate_list(cf):
    records = np.array([[0.1, 5.0], [0.2, 4.0], [0.1, 5.0], [0.3, 3.0]])
    uniq, removed = cf.remove_duplicate_list(records)
    assert np.array_equal(uniq, [[0.1, 5.0], [0.2, 4.0], [0.3, 3.0]])
    assert removed == [2]


def test_remove_duplicate_list_without_duplicates(cf):
    records = np.array([[0.1, 5.0], [0.2, 4.0]])
    uniq, removed = cf.remove_duplicate_list(records)
    assert np.array_equal(uniq, records)
    assert removed == []


def test_remove_duplicate_ponds_settigs(cf):
    off_val = np.array([[1, 0, 2], [3, 3, 3], [0, 0, 1]])
    off_chrom = np.arange(9).reshape(3, 3)
    simulated_val = np.array([[3, 3, 3], [5, 5, 5]])
    kept_val, kept_chrom = cf.remove_duplicate_ponds_settigs(
        off_val, off_chrom, simulated_val, np.zeros((2, 3)))
    assert np.array_equal(kept_val, [[1, 0, 2], [0, 0, 1]])
    assert np.array_equal(kept_chrom, [[0, 1, 2], [6, 7, 8]])


def test_remove_same_objectives_population(cf):
    pop = np.arange(12).reshape(4, 3)
    assert np.array_equal(cf.remove_same_objectives_population(pop, [1, 3]),
                          [[0, 1, 2], [6, 7, 8]])
    assert np.array_equal(cf.remove_same_objectives_population(pop, []), pop)


# --- ranking -----------------------------------------------------------------

def test_non_dominated_sorting(cf):
    fronts = cf.non_dominated_sorting(len(OBJS), OBJS)
    assert fronts == {0: [0, 1, 2], 1: [3, 4], 2: [5]}


def test_non_dominated_sorting_single_front(cf):
    objs = np.array([[1, 4], [2, 3], [3, 2], [4, 1]], dtype=float)
    assert cf.non_dominated_sorting(4, objs) == {0: [0, 1, 2, 3]}


def test_calculate_crowding_distance(cf):
    fronts = cf.non_dominated_sorting(len(OBJS), OBJS)
    dist = cf.calculate_crowding_distance(fronts, OBJS)
    boundary = 666666666 + 333333333
    # middle of front 0: normalised neighbour gap of 1 for each objective
    assert dist[0] == {0: boundary, 1: 2.0, 2: boundary}
    # fronts with one or two members are all boundary solutions
    assert dist[1] == {3: boundary, 4: boundary}
    assert dist[2] == {5: boundary}


def test_fitness_sort(cf):
    fronts = cf.non_dominated_sorting(len(OBJS), OBJS)
    dist = cf.calculate_crowding_distance(fronts, OBJS)
    # within a front: larger distance first; ties keep the order in which
    # calculate_crowding_distance stored them (sorted by damage)
    assert cf.fitness_sort(dist, len(OBJS)) == [2, 0, 1, 4, 3, 5]


def test_fitter_parent_never_picks_the_worst(cf):
    sorted_fitness = np.array([3, 1, 0, 2])   # 3 is fittest, 2 is least fit
    np.random.seed(1)
    picks = {int(cf.fitter_parent(sorted_fitness, 4)) for _ in range(300)}
    assert 2 not in picks
    assert picks <= {0, 1, 3}
    assert 3 in picks


def test_fitter_parent_with_two_solutions_picks_the_best(cf):
    np.random.seed(3)
    for _ in range(20):
        assert cf.fitter_parent(np.array([1, 0]), 2) == 1


# --- crossover and mutation --------------------------------------------------

def _parents():
    p1_loc = np.array([1, 1, 0])
    p2_loc = np.array([0, 1, 1])
    p1 = np.zeros(SIZE_LEN, dtype=int)
    p1[0:4] = 1      # pond 0: 4 steps
    p1[7:9] = 1      # pond 1: 2 steps
    p2 = np.zeros(SIZE_LEN, dtype=int)
    p2[7:13] = 1     # pond 1: 6 steps
    p2[13:15] = 1    # pond 2: 2 steps
    return p1, p2, p1_loc, p2_loc


def test_crossover_without_crossover_or_mutation_copies_parents(cf):
    p1, p2, l1, l2 = _parents()
    np.random.seed(0)
    c1, c2, v1, v2 = cf.crossover_uniform_mutation_bitflip(
        p1, p2, l1, l2, 0.0, np.array([0.0, 0.0]), N_PONDS, SIZE_LEN,
        GENE_START, GENE_END)
    assert np.array_equal(c1, p1) and np.array_equal(c2, p2)
    assert np.array_equal(v1, [4, 2, 0]) and np.array_equal(v2, [0, 6, 2])


def test_crossover_with_full_crossover_swaps_parents(cf):
    p1, p2, l1, l2 = _parents()
    np.random.seed(0)
    c1, c2, v1, v2 = cf.crossover_uniform_mutation_bitflip(
        p1, p2, l1, l2, 1.0, np.array([0.0, 0.0]), N_PONDS, SIZE_LEN,
        GENE_START, GENE_END)
    assert np.array_equal(c1, p2) and np.array_equal(c2, p1)
    assert np.array_equal(v1, [0, 6, 2]) and np.array_equal(v2, [4, 2, 0])


def test_crossover_children_are_valid(cf):
    p1, p2, l1, l2 = _parents()
    np.random.seed(7)
    for _ in range(50):
        c1, c2, v1, v2 = cf.crossover_uniform_mutation_bitflip(
            p1, p2, l1, l2, 0.6, np.array([0.02, 0.02]), N_PONDS, SIZE_LEN,
            GENE_START, GENE_END)
        for c, v in ((c1, v1), (c2, v2)):
            assert c.shape == (SIZE_LEN,)
            assert set(np.unique(c)) <= {0, 1}
            assert np.array_equal(v, block_sums(c))
        # parents are not modified
        assert np.array_equal(p1, _parents()[0])


def test_crossover_is_reproducible(cf):
    p1, p2, l1, l2 = _parents()
    args = (p1, p2, l1, l2, 0.6, np.array([0.02, 0.02]), N_PONDS, SIZE_LEN,
            GENE_START, GENE_END)
    np.random.seed(11)
    a = cf.crossover_uniform_mutation_bitflip(*args)
    np.random.seed(11)
    b = cf.crossover_uniform_mutation_bitflip(*args)
    for x, y in zip(a, b):
        assert np.array_equal(x, y)


# --- population bookkeeping --------------------------------------------------

def test_separate_new_old(cf):
    new = np.array([[1, 0], [2, 2], [0, 3]])
    old = np.array([[2, 2], [5, 5]])
    new_chroms, old_chroms, old_idx = cf.separate_new_old(new, old)
    assert np.array_equal(new_chroms, [[1, 0], [0, 3]])
    assert np.array_equal(old_chroms, [[2, 2]])
    assert old_idx == [1]


@pytest.mark.xfail(raises=UnboundLocalError, strict=True,
                   reason="known issue: separate_new_old fails when no solution "
                          "of the old population is kept")
def test_separate_new_old_when_nothing_is_kept(cf):
    new = np.array([[1, 0], [0, 3]])
    old = np.array([[2, 2]])
    new_chroms, old_chroms, old_idx = cf.separate_new_old(new, old)
    assert np.array_equal(new_chroms, new)
    assert old_idx == []


# --- termination criterion ---------------------------------------------------

def test_auc_func(cf):
    cost = np.array([[1.0], [2.0], [3.0]])
    damage = np.array([[3.0], [2.0], [2.5]])
    # damage minus its minimum: 1, 0, 0.5 -> trapezoids 0.5 + 0.25
    assert cf.auc_func(cost, damage) == pytest.approx(0.75)
    # order of the solutions does not matter
    order = [2, 0, 1]
    assert cf.auc_func(cost[order], damage[order]) == pytest.approx(0.75)
