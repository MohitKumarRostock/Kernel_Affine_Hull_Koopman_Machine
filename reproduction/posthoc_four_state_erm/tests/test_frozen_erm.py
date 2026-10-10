"""Deterministic tests for the frozen four-state reanalysis; no simulations."""
import csv
import hashlib
import math
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from certificate_reanalysis import four_state_row
OUT=ROOT/"outputs"/"historical_117k"
EXPECTED={
 "cell_summary_joint.csv":"6ffea4bfe2e8e1e545ddb8aecf241fc1341a2c3b7a934767907c96d7e2bfac37",
 "cell_summary_pointwise.csv":"eefda7da73d7dc9024f36e33664b121e92cedbb6b305cb19c961ad8c298b35a6",
 "replicates_joint.csv":"380476b3ce1f01129580ebfa0932828be7e2eae1c495d90bd980335278b859b9",
 "replicates_pointwise.csv":"927d48bfd80707abf17ff319f712282c948b0a8162c77a86ab161d5aa4519373"
}

class FrozenFourStateTests(unittest.TestCase):
    def test_balanced_exact_erm(self):
        z={"dataset_id":"balanced","dynamics":"conflict","p":"31/32",
           "n_a":"1024","n_b":"1024","n_d":"1024","n_e":"1024"}
        out=four_state_row(z,0.05)
        self.assertAlmostEqual(out["s_hat"],225/1024,places=14)
        self.assertAlmostEqual(out["exact_empirical_min_RMSE"],15/32,places=14)
        self.assertAlmostEqual(out["L_label"],.3030649002592908,places=12)
        self.assertAlmostEqual(out["L_prototype"],.33617196558820084,places=12)
        self.assertAlmostEqual(out["L_ERM_generic"],.3290431912372553,places=12)

    def test_all_available_plain_csv_hashes(self):
        for name,checksum in EXPECTED.items():
            p=OUT/name
            if not p.is_file():continue # Compressed full replicates may be checked in instead.
            self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),checksum)

    def test_manifest_and_summary_if_present(self):
        import json
        m=OUT/"manifest.json"
        if not m.is_file():return
        meta=json.loads(m.read_text())
        self.assertEqual(meta["source_archive_records"],117000)
        self.assertEqual(meta["expected_and_observed_cells"],117)
        for name in ("cell_summary_joint.csv","cell_summary_pointwise.csv"):
            with (OUT/name).open(newline="") as f:rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),117)
        with (OUT/"cell_summary_joint.csv").open(newline="") as f:
            rows=list(csv.DictReader(f))
        by_size={int(z["M"]):z for z in rows if z["case_id"]=="conflicting_successors_p_31_over_32"}
        for M,expected in [(1024,.1468),(4096,.3126),(8192,.3593),(32768,.4144)]:
            self.assertEqual(round(float(by_size[M]["prototype_mean"]),4),expected)

if __name__=="__main__":unittest.main()
