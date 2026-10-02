"""Stateless count sampling for the uniform four-state pilot.

Each call constructs a fresh SeedSequence and PCG64 Generator from the five
explicit seed components. It makes exactly one scalar-n multinomial call,
with size=None and probabilities (1/4,1/4,1/4,1/4). It does not use global
random state, a shared generator, Python hash(), or execution-order seeds.

Sampling is separate from certificate evaluation so a campaign runner can
persist the returned counts before attempting subsequent calculations.
This module performs no file I/O, campaign execution, or provenance checks.
Importing it does not draw samples. The runner must enforce the committed
configuration, unique replicate identifiers, and clean-source policy.

All seed components are restricted to unsigned 32-bit integers, with a
positive n_pairs, to keep a fixed five-word seed layout. Values are never
truncated or reduced modulo a bound. Extending this range requires a new
validated seed scheme. Different seed tuples do not guarantee different
sampled counts or constitute an empirical independence test.

Exact random-stream reproduction requires the reference NumPy environment;
no cross-version or cross-platform bitwise guarantee is made. Save counts
as well as seed metadata so their statistics remain independently checkable.
"""

from dataclasses import dataclass
from numbers import Integral

import numpy as np

COUNT_SAMPLER_ID = "four_state_multinomial_pcg64_v1"
MAX_SEED_COMPONENT = 2**32 - 1
SEED_MATERIAL_ORDER = (
    "campaign_key", "case_key", "n_pairs", "replicate_index", "root_seed",
)
STATE_ORDER = ("a", "b", "d", "e")
STATE_PROBABILITIES = (0.25, 0.25, 0.25, 0.25)


@dataclass(frozen=True)
class SeededFourStateCounts:
    """Immutable sampling result; serialize together with campaign provenance."""

    sampler_id: str
    numpy_version: str
    campaign_key: int
    case_key: int
    n_pairs: int
    replicate_index: int
    root_seed: int
    seed_material: tuple[int, ...]
    seed_material_order: tuple[str, ...]
    bit_generator: str
    seed_sequence: str
    seed_sequence_pool_size: int
    draws_per_replicate: int
    state_order: tuple[str, ...]
    probabilities: tuple[float, ...]
    state_counts: tuple[int, ...]


def _seed_component(name: str, value: Integral) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, not a boolean.")
    number = int(value)
    lower = 1 if name == "n_pairs" else 0
    if not lower <= number <= MAX_SEED_COMPONENT:
        raise ValueError(f"{name} must lie in [{lower}, {MAX_SEED_COMPONENT}].")
    return number


def sample_four_state_counts(
    *, campaign_key: Integral, case_key: Integral, n_pairs: Integral,
    replicate_index: Integral, root_seed: Integral,
) -> SeededFourStateCounts:
    """Draw one count vector using the explicitly specified replicate seed.

    Repeating these inputs recreates the same draw in the same reference
    environment; reordering other calls does not affect this call. Repeating
    a replicate is a reproduction check, not a new independent replicate.
    Exceptions propagate: no fallback seed, resampling, or hidden retry.
    """
    raw = dict(
        campaign_key=campaign_key, case_key=case_key, n_pairs=n_pairs,
        replicate_index=replicate_index, root_seed=root_seed,
    )
    values = {name: _seed_component(name, raw[name]) for name in SEED_MATERIAL_ORDER}
    material = tuple(values[name] for name in SEED_MATERIAL_ORDER)

    sequence = np.random.SeedSequence(list(material), pool_size=4)
    generator = np.random.Generator(np.random.PCG64(sequence))
    drawn = generator.multinomial(values["n_pairs"], STATE_PROBABILITIES, size=None)
    if drawn.shape != (4,) or drawn.dtype.kind not in "iu":
        raise RuntimeError("Multinomial sampler returned invalid shape or dtype.")
    counts = tuple(int(n) for n in drawn)
    if any(n < 0 for n in counts) or sum(counts) != values["n_pairs"]:
        raise RuntimeError("Multinomial counts are invalid or have the wrong total.")

    return SeededFourStateCounts(
        sampler_id=COUNT_SAMPLER_ID,
        numpy_version=np.__version__,
        **values,
        seed_material=material,
        seed_material_order=SEED_MATERIAL_ORDER,
        bit_generator="PCG64",
        seed_sequence="SeedSequence",
        seed_sequence_pool_size=4,
        draws_per_replicate=1,
        state_order=STATE_ORDER,
        probabilities=STATE_PROBABILITIES,
        state_counts=counts,
    )
