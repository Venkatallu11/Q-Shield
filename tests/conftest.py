import random

import pytest

from qshield.experiments.generator import GeneratorConfig, make_instance


@pytest.fixture
def rng():
    return random.Random(20260907)


@pytest.fixture
def small_problem():
    """A deterministic instance small enough to brute-force in a test."""
    return make_instance(random.Random(3), GeneratorConfig(nodes=5, cross_links=3))


def random_problems(count, seed=1, **kwargs):
    r = random.Random(seed)
    return [make_instance(r, GeneratorConfig(**kwargs)) for _ in range(count)]
