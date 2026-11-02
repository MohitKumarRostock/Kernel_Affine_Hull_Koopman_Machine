# Speaker-separated forecasting of speech energy

## Research question and scope

This prospective analysis evaluates bounded-loss risk certification in a second
acquisition domain with documented recording identifiers and a substantially
larger number of held-out groups than the pen-trajectory study. It forecasts a
causally computed acoustic energy observable within released speech clips. It
does not evaluate keyword recognition, continuous-stream detection, or a
deployment requirement supplied by an audio application.

This protocol is fixed before downloading or inspecting the audio observations,
fitting models, or calculating calibration and test losses. It is an archived
analysis specification, not an external preregistration. Its hash is recorded
in the results. Every scheduled fitting repetition and every selected family's
result will be retained, including unsuccessful certification and unfavorable
comparisons. No choices below may be adapted to calibration or test outcomes.

## Data and independent units

Use the official TensorFlow Mini Speech Commands archive:
https://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip

The provider's tutorial describes 8,000 WAV clips of eight words. Warden (2018),
*Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition*,
https://arxiv.org/abs/1804.03209, documents the parent dataset's recording and
speaker identifiers, 16 kHz mono 16-bit PCM encoding, quality control, and
whole-recording selection of the loudest one-second segment. The dataset is
licensed under CC BY 4.0. Record the downloaded archive's SHA-256, all relative
clip paths, and source speaker identifiers (the filename prefix before
`_nohash_`). These are public pseudonymous recording-group identifiers.

Retain mono, 16 kHz, signed 16-bit clips containing exactly 16,000 samples.
List every excluded file and its reason. This duration restriction and the
provider's trimming define a retrospective population of released clips;
claims do not extend to arbitrary newly started recordings. Do not normalize
an utterance by its own peak, mean, variance, or future extrema.

Sort eligible speaker identifiers, permute them once with NumPy PCG64 seed
8602026, and allocate the first floor(0.15 G) to training, the next floor(0.10 G)
to validation, the next floor(0.60 G) to calibration, and the remainder to test,
where G is the eligible speaker count. All clips of an identifier stay together.
No official keyword-recognition benchmark score is claimed for this new split.

A statistical episode is the complete retained bundle of clips from one
identifier. Losses average over the predefined origins and then over clips;
speakers receive equal weight. Confidence statements assume independently
sampled, identically distributed speaker-record bundles and independence of
development and calibration bundles. The recording identifiers and separation
are documented; the common-population sampling assumption remains a modeling
assumption. The source uses session identifiers for web volunteers, so distinct
identifiers are not a biometric proof of distinct people. Fitting repetitions
on the same groups do not create new independent evaluation units.

## Forecasting observable

Divide a clip into 100 consecutive, non-overlapping frames of 160 samples
(10 ms). Convert PCM values to x = integer / 32768. For each frame define

    e_t = clip((10 log10(max(mean(x^2), 10^-6)) + 60) / 60, 0, 1).

Thus e_t is the frame power level in dB relative to full scale, floored at
-60 dBFS and linearly represented in [0,1]. The frame is observed at its end;
its calculation uses no later frame. Targets and predictions are in [0,1],
so every scalar squared loss is bounded by one from the task definition.
This is a fixed bounded energy observable, not an error guarantee for the raw
audio waveform or for unclipped acoustic power. Multiplying RMSE by 60 gives
RMSE in dB for the floored observable.

History lengths are 4 and 12 frames. Horizons are 1, 4, and 12 frames
(10, 40, and 120 ms). All candidates use origins 11,15,...,87 (zero based).
Prespecified reporting tolerances are 6, 12, and 18 dB, corresponding to
normalized RMSE 0.10, 0.20, and 0.30. They are transparent resolution levels
for the observable, not asserted requirements of a deployed system.

## Development and comparators

Perform three complete fits, seeds 8601, 8602, 8603, on the same group split.
For each fit sample at most 4,096 training histories with equal speaker
probability, then uniform clip and origin probability. Use all validation
groups with equal group weights. Select one candidate per family by mean
validation MSE over the three horizons. Resolve ties by feature count, history
length, and a deterministic candidate name. Freeze all selections before
opening any calibration or test loss record.

Eight reported families are retained KAHM, retained normalized RBF features,
retained affine-hull features, linear autoregression, quadratic regression,
Gaussian-kernel regression with a Nystrom approximation, persistence, and a
training-mean constant predictor. All predict the same observable and origins.

The three geometric families share histories {4,12}, preprocessing
{standardization, raw}, prefix block sizes {2,4,8,16}, coefficient budgets
{1,4}, and ridge coefficients {10^-4,10^-2}. Geometry uses at most 1,600
training inputs and 24 local KAHM reference points, affinity exponent 16,
and local subspace dimension 2. RBF and affine score multipliers are
{0.25,1,4}; affine locality and subspace conventions remain those of the
published implementation. KAHM uses its inherited reconstruction mechanism.
These are specified implementation comparisons, not equal statistical
capacity assertions.

Linear and quadratic regressions use training standardization and ridge
coefficients {10^-6,10^-4,10^-2,10^-1}. Gaussian-kernel regression uses
training standardization, 128 Nystrom landmarks, ridge coefficients
{10^-4,10^-2,10^-1}, and inverse bandwidths {0.25,1,4} times the inverse
median nonzero squared distance in a fixed 512-row training subsample.
All families have both history lengths. Predictions are clipped to [0,1].

## Calibration, test reporting, and timing

For each selected family, seed, and horizon, calculate one loss per calibration
speaker and a bounded-mean KL upper endpoint. Use one simultaneous failure
budget 0.05 over all 3 x 8 x 3 = 72 upper statements; each tail is 0.05/72.
The endpoint inverts n kl(mean_loss || population_loss) = log(72/0.05) on
the upper side. No data-dependent choice between concentration bounds is made.
All displayed tolerance decisions are consequences of these 72 simultaneous
statements. Test losses are scored only after development and calibration
calculations are frozen; no retuning follows either set of losses.

Report all selected hyperparameters, per-speaker calibration and test losses,
mean and sample SD of test RMSE across the three fits, upper endpoints,
accept/unresolved decisions, and training feasibility/optimization gaps.
The SD measures fitting variability on the same test cohort. Provide paired
speaker-loss differences descriptively; no additional population superiority
claim is made without a separately allocated test.

Measure geometry fitting and transforms, decoder fitting, and scoring with
one BLAS thread. Charge shared geometry once per bank; distinguish a whole
validation search from selected-model inference. Timing is environment-specific
and includes no deployment-speed or universal efficiency claim.

## Reproduction and failures

The runner checks input ranges, disjoint identifiers, causal frame indexing,
candidate count, all selected-model freezes, and complete loss arrays. Record
source hashes and software versions. An interrupted run may resume only from
completed immutable development records with the same protocol and source
hashes; otherwise use a new output directory and retain the failed record.
The audio archive is downloaded from the provider rather than committed to
the repository. Release the preparation, fitting, analysis, and verification
code and the compact derived evidence needed to reproduce the manuscript.
