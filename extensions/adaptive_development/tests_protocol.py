"""Causality checks on excluded mechanics streams, before the study freeze."""
from __future__ import annotations

import copy
import unittest

import numpy as np
from threadpoolctl import threadpool_limits

from .data import make_synthetic
from .run import config_read, execute


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg=config_read()
        cls.cfg["budgets"]={"small":{"reference_budget":16,"replay_size":32}}

    def test_future_changes_do_not_change_earlier_forecasts_or_selection(self):
        task=make_synthetic(20261098,"recurring",n=256)
        changed=copy.deepcopy(task)
        cut=task["eval_start"]
        changed["values"][cut:]+=3.0
        with threadpool_limits(limits=1):
            for method in ("ridge","adaptive_kahm","adaptive_affine","adaptive_rbf","kernel_rr"):
                a,pa,_=execute(task,method,"small",self.cfg)
                b,pb,_=execute(changed,method,"small",self.cfg)
                before=pa["origin"]<cut
                for key in ("baseline","correction","prediction"):
                    np.testing.assert_array_equal(pa[key][before],pb[key][before])
                self.assertEqual(a["calibration"],b["calibration"])

    def test_maturity_immutable_scoring_and_gamma(self):
        task=make_synthetic(20261097,"stationary",n=256)
        with threadpool_limits(limits=1):
            s,p,trace=execute(task,"adaptive_kahm","small",self.cfg)
        for row in trace:
            self.assertLessEqual(row["latest_supervised_target"],row["time"])
        scored=np.isfinite(p["truth"])
        np.testing.assert_array_equal(p["truth"][scored],task["values"][p["target"][scored]])
        self.assertTrue(np.all(p["target"]==p["origin"]+p["horizon"]))
        for selection in s["calibration"]:
            h=selection["horizon"]
            use=(p["horizon"]==h)&(p["origin"]<task["eval_start"])&(p["target"]<task["eval_start"])
            losses=[np.mean((p["truth"][use]-(p["baseline"][use]+g*p["correction"][use]))**2)
                    for g in self.cfg["gamma_grid"]]
            self.assertEqual(selection["gamma"],self.cfg["gamma_grid"][int(np.argmin(losses))])
            self.assertLess(selection["latest_target"],task["eval_start"])
        np.testing.assert_array_equal(p["prediction"],p["baseline"]+p["gamma"]*p["correction"])
        zero=p["gamma"]==0
        np.testing.assert_array_equal(p["prediction"][zero],p["baseline"][zero])
        expected=sum(len(task["values"])-h-task["eval_start"] for h in task["horizons"])
        self.assertEqual(sum(h["count"] for h in s["per_horizon"]),expected)
        self.assertEqual(s["framework_errors"],[])


if __name__=="__main__":
    unittest.main()
