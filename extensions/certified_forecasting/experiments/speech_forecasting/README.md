# Forecasting bounded speech energy

This study evaluates the forecasting and upper-risk calibration procedure on
the official TensorFlow Mini Speech Commands subset. It uses documented
recording identifiers, a causal frame-energy observable, eight model families,
and a protocol fixed before downloading or inspecting the audio. The protocol
is an archived specification, not an external preregistration.

The evaluated cohort contains 7,178 eligible one-second clips from 1,630
identifiers. The fixed split has 244 training, 163 validation, 978 calibration,
and 245 test identifiers. All clips from one identifier remain together.
Source identifiers can denote recording sessions and do not prove distinct
human identity. Population confidence statements assume iid group-record
bundles; source curation and duration eligibility define a retrospective
released-clip population.

The observable is frame power in decibels (dB) relative to full scale,
floored at -60 dB and linearly mapped to [0,1]. Full scale refers to unit
amplitude after dividing the signed 16-bit audio samples by 32,768. Frames
are nonoverlapping 10 ms intervals and become available at their end. Predictions concern 10, 40, and 120 ms horizons, without
normalization using future utterance statistics. A normalized RMSE multiplied
by 60 is RMSE in dB for this floored observable.

## Reproduction

From the extension root, download the provider's CC BY 4.0 archive to a local
data directory outside the retained result directories:

```bash
mkdir -p data_cache/speech
curl --fail --location \
  https://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip \
  --output data_cache/speech/mini_speech_commands.zip
sha256sum data_cache/speech/mini_speech_commands.zip
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
  python experiments/speech_forecasting/run_study.py \
  --archive data_cache/speech/mini_speech_commands.zip \
  --output results/reproduced_speech
python experiments/speech_forecasting/analyze.py \
  --results results/reproduced_speech \
  --output results/reproduced_speech/analysis --paper paper
```

The expected archive SHA-256 is
`49650f2341b26d886b46b3f4fb8fed59e30300b17550f1ee4a768b3106cf93a0`.
The runner refuses an existing output directory. It freezes all 24
validation-selected models before evaluation, writes a calibration freeze
before test scoring, and records every selected outcome. Its 72 upper
risk bounds cover 3 fitting seeds, 8 model families, and 3 horizons, with a
total failure probability of 0.05. Each endpoint receives probability 0.05/72. Tolerances of 6, 12, and 18 dB
are stated resolution levels, not application requirements.

The independent verifier accepts the provider archive, initial result
directory, and strict-order replay directory; use
`python experiments/speech_forecasting/verify_study.py --help` for its options.
It independently reconstructs the observable, model losses, selection minima,
Kullback--Leibler (KL) upper endpoints, and replay equivalence. The focused tests run with:

```bash
PYTHONPATH=src python -m unittest discover -s tests -p test_speech_forecasting.py -v
```

## Evaluation order and source provenance

`results/speech_forecasting_v1/` retains the first execution. All models were
already frozen before held-out losses were evaluated, but calibration and test
scoring were interleaved by model. That ordering differed from the protocol's
requirement to finish all calibration calculations before test scoring.

`results/speech_forecasting_v1_replay/` is an unchanged-seed verification replay
with the required order and a separate calibration freeze. Every selected
predictor value, group loss, and non-timing result matches the initial execution
exactly. No model, hyperparameter, observable, or tolerance was changed after
seeing either outcome. The two executions are one empirical study, not
independent replications. Both source versions and the independent audit are
retained. Pickled predictors are trusted generated artifacts; regenerate them
from the source before loading if their provenance is uncertain.

## Results and interpretation

All six learned families meet the all-horizon 18 dB upper criterion in all
three fits. None meets the all-horizon 6 or 12 dB criterion. Persistence and
the constant model do not meet 18 dB across all horizons. Ridge AR, quadratic,
and Gaussian-kernel regression have smaller test errors than KAHM at each
reported horizon. These accuracy differences are descriptive; no separately
budgeted population-superiority claim is made.

Three fitting seeds share the same evaluation cohort. Their standard deviations
measure fitting variation, not uncertainty over new cohorts. The analysis
exports all paired group differences and separates shared feature-bank costs
from selected decoder fitting and prediction times. Timing is specific to the
recorded environment.

Data attribution: Pete Warden (2018), *Speech Commands: A Dataset for
Limited-Vocabulary Speech Recognition*, https://arxiv.org/abs/1804.03209;
TensorFlow Mini Speech Commands, https://www.tensorflow.org/tutorials/audio/simple_audio.
Data are CC BY 4.0; the provider audio archive is downloaded rather than
committed to this repository.
