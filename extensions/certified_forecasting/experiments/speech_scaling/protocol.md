# Paired input scaling for speech-energy forecasting

## Scope

This descriptive follow-up compares four frozen input transformations on the
speech-energy task. The original speech study and its held-out results were
already available when this question was specified. The same recording groups
are reused. Consequently, this experiment is not an independent confirmation,
and it makes no new confidence or certification claim. The original study's
upper bounds and simultaneous allocation are not extended to these models.

This specification, the runner, its imported implementation, and the reference
records are hashed before fitting. All prescribed arms and fitting seeds are
reported, including unfavorable changes. Calibration groups are not scored.

## Data, outputs, and pairing

Use the official Mini Speech Commands archive with SHA-256
`49650f2341b26d886b46b3f4fb8fed59e30300b17550f1ee4a768b3106cf93a0`.
The source is
<https://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip>.
Attribution: Speech Commands, Pete Warden (2018), and the TensorFlow Mini Speech
Commands subset; CC BY 4.0. The source archive remains outside the repository.

Reuse exactly the original eligibility rule, group split, causal frame-energy
observable, histories {4,12}, horizons {1,4,12}, origins 11,15,...,87, and
seeds {8601,8602,8603}. The same deterministic 4,096 training rows are supplied
to every geometric family and scaling arm within a fitting seed. They are
sampled by equal training-group probability. Targets remain in [0,1], and
reported RMSE is multiplied by 60 to obtain dB for the floored energy observable.
Validation and test losses average over clips and origins within an identifier,
then give equal weight to the identifier. The existing 245 test groups are reused.

## Scaling intervention and candidate grid

For each of KAHM, RBF, and affine-hull features, compare these input maps:

1. Training mean and standard deviation (`standard`).
2. Untransformed energy histories (`raw`).
3. Per-coordinate training min--max mapping to [-1,+1] (`minmax`).
4. The same map, with query values clipped to [-1,+1] (`minmax_clip`).

Fit every transformation on all 4,096 sampled training histories, before the
1,600-row geometry subsample. Freeze its parameters thereafter. A coordinate
constant in these training rows maps to zero. Unclipped min--max values may
leave [-1,+1] on validation or test observations; neither these observations
nor the unused calibration groups determine an extremum. The clipped arm is a
separate model choice rather than a correction to the primary min--max map.

Keep the original dictionaries {2,4,8,16}, affinity exponent 16, local reference
limit 24, subspace dimension 2, coefficient budgets {1,4}, and ridge parameters
{0.0001,0.01}. RBF and affine-hull sharpness values are {0.25,1,4}; KAHM uses
one inherited reconstruction configuration. For each seed and scaling mode
there are 32 KAHM, 96 RBF, and 96 affine candidates. Across four modes and
three seeds the total is 2,688 fitted candidates.

Select one candidate independently within each (seed, family, scaling) arm by
mean validation MSE over all three horizons. Resolve ties by feature count,
history length, then candidate name, as in the original study. This gives
36 selected models. No scaling arm is chosen using test error. Freeze all
selected models and all development records before computing any test losses.
The standard/raw candidate values and the corresponding original combined-arm
selections are replayed against archived records to check comparability.

## Retained evidence and summaries

Retain all candidate validation scores, fitted scaling parameters, geometry
subsample indices, training rows, source/environment hashes, selected models,
and one test loss per selected model, identifier, and horizon. Retain model and
data hashes even when large, regenerable binary files are excluded from Git.

Report every family's mean test MSE and RMSE across the three fits, sample
standard deviation across fits, and per-seed MSE changes relative to its paired
standardized arm. A paired identifier-level mean difference and conditional
standard error describe cohort variation for a fixed fit; they are descriptive
statistics, not adjusted confidence intervals or additional independent fits.
Report min--max clipping diagnostics and all three horizons. Do not use large
numbers of windows to substitute for recording-group replication.

## Reproduction

From the extension root, with its documented dependencies installed:

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/speech_scaling/run_study.py --archive /path/to/mini_speech_commands.zip --output results/speech_scaling_reproduction
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/speech_scaling/verify_study.py --results results/speech_scaling_reproduction
```

Use a new output directory for every run. The verifier reads only model files
created by this experiment; Python pickle files from unknown sources must not
be loaded. A portable numerical audit can reconstruct all summaries directly
from the retained compressed group losses without loading models.
