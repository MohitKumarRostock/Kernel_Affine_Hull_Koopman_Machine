# Frozen block-witness comparison

This protocol is a new, prospective numerical check of the centered
Lipschitz block hierarchy. The eight distributions and their known behavior
were examined in the earlier witness comparison; this is a theoretical
follow-up on those same declared laws, not a new application population.

## Fixed design

Use all eight laws returned by `experiments/witness_comparison/core.py`, with
the same scalar 1-Lipschitz class and target interval [-1,1]. Generate 100
independent random streams per law, seed 202610291901 and NumPy SeedSequence
children indexed by law and repetition. Each stream contains 4096 iid draws.
Use nested prefixes n=256,1024,4096. No law, seed, or result is omitted.

For each prefix, split its observations in their original order into disjoint
blocks of size b=2,4,8,16,32. The score is the minimum Lipschitz residual SSE
divided by b-1. At b=2 evaluate the exact pair formula. At larger b use a
feasible dual lower value from the existing finite-support quadratic program,
multiplied by b/(b-1). Cache repeated count vectors only to avoid repeated
optimization of identical programs. This caching prevents interpreting the
recorded runtime as a general continuous-support timing benchmark.

Use the sharp scalar score range C_b=4*floor(b*b/4)/(b*(b-1)). Each lower
endpoint is bounded-mean binary-KL inversion applied to the independent block
scores. The full-sample dual-certified empirical minimum, with loss bound4,
is an explicit comparator. Each of these six methods is evaluated at each
of three sample sizes. Allocate .05/18 to each lower endpoint, so all18
statements have joint confidence at least95% for a given law and repetition.
This is not a single95% confidence statement across800 independent streams.

Keep every score summary, all raw categorical draws, every nontrivial cached
optimization's feasible primal and dual values, and all lower endpoints.
Record the number of stream-wise simultaneous coverage violations against
the known finite-support optimum, and the largest primal-dual gap.
Report every method even if the full-sample comparator is stronger. Increasing
b is not assumed to improve a finite-sample confidence endpoint.

## Deterministic examples

Also tabulate the exact population hierarchy for the rare-conflict law at
epsilon=.01,.05,.10, and for independent binary target noise on M=4,16,64
mutually separated representation points. Use every integer b=2,...,512.
The formulas are derived in the manuscript and independently checked by
count enumeration for small block sizes. These curves use no simulated data.

## Freeze and verification

The runner refuses an existing output directory. Before drawing any sample,
it writes the protocol and source SHA-256 hashes, exact settings, timestamp,
and source snapshot. Source hashes are checked again on completion. A separate
verifier reconstructs all endpoints from the retained samples and dual
records, checks constraints and weak duality numerically, and independently
checks the exact formulas. Numerical guards are not interval arithmetic.
