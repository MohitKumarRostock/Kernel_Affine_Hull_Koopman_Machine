Extract source_snapshot.tar.gz into an empty directory. It preserves the parent
KAHM Python files and extensions/certified_forecasting/ ancestry required by imports.
The three matched_geometry_*.json reference configs are included.

cd extensions/certified_forecasting
python -m pip install -r requirements-reference.txt
python experiments/prepare_pendigits.py --help
Prepare the pinned original PenDigits arrays using that script, then run:
python experiments/run_matched_geometry.py --output results/reproduced_matched_geometry --workers 6 --prepared data_cache/pendigits/pendigits_original.npz
python experiments/analyze_matched_geometry.py --study results/reproduced_matched_geometry --output results/reproduced_matched_geometry_analysis

This snapshot recreates the experiment sources and protocol, not Git history.
The frozen source/config hashes identify exact scientific inputs. Dependency pins
are supplied as reproduction support. See the full repository for the manuscript,
historical studies, tests, publication generators, and complete documentation.
