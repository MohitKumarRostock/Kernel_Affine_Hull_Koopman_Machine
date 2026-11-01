import csv
import json
import math
from pathlib import Path
import numpy as np

TABLE_KEYS = {}

def table_comparison(directory, expected_tables, audit):
    """Compare every parsed CSV field; no sampled rows and no rounded comparison."""
    if set(expected_tables) != set(TABLE_KEYS):
        raise AssertionError("Independent complete table roster differs")
    observed_files = {path.name for path in Path(directory).glob("*.csv")}
    audit.require(observed_files == set(TABLE_KEYS), "Exported complete table roster")
    result = {}
    for name, expected in expected_tables.items():
        with (Path(directory)/name).open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            header = reader.fieldnames
            actual = list(reader)
        fields = set(expected[0])
        audit.require(set(header) == fields and len(header) == len(fields), "Table field roster", name)
        audit.require(len(actual) == len(expected), "Table row count", (name, len(actual), len(expected)))
        kinds = {}
        for field in fields:
            samples = [row[field] for row in expected if row[field] is not None]
            if not samples:
                kinds[field] = "none"
            elif all(isinstance(value, (bool, np.bool_)) for value in samples):
                kinds[field] = "bool"
            elif all(isinstance(value, (int, float, np.number)) and not isinstance(value, (bool, np.bool_)) for value in samples):
                kinds[field] = "number"
            elif all(isinstance(value, (list, dict, tuple)) for value in samples):
                kinds[field] = "json"
            else:
                kinds[field] = "string"

        def parse(field, value):
            if value == "":
                return "" if kinds[field] == "string" else None
            kind = kinds[field]
            if kind == "number":
                return float(value)
            if kind == "bool":
                if value not in ("True", "False"):
                    raise AssertionError("Invalid CSV boolean")
                return value == "True"
            if kind == "json":
                return json.loads(value)
            return value

        actual_index = {}
        for row in actual:
            audit.require(None not in row and all(value is not None for value in row.values()), "Complete CSV row", name)
            normalized = {field: parse(field, row[field]) for field in fields}
            key = tuple(normalized[field] for field in TABLE_KEYS[name])
            audit.require(key not in actual_index, "Duplicate table key", (name, key))
            actual_index[key] = normalized
        expected_keys = {tuple(row[field] for field in TABLE_KEYS[name]) for row in expected}
        audit.require(len(expected_keys) == len(expected) and set(actual_index) == expected_keys, "Exact table key roster", name)
        absolute = 0.
        numeric_cells = 0
        exact_cells = 0
        for row in expected:
            key = tuple(row[field] for field in TABLE_KEYS[name])
            found = actual_index[key]
            for field, wanted in row.items():
                observed = found[field]
                context = (name, key, field, observed, wanted)
                if wanted is None:
                    audit.require(observed is None, "Expected null table cell", context)
                    exact_cells += 1
                elif kinds[field] == "number":
                    audit.require(observed is not None and math.isfinite(observed) and math.isfinite(float(wanted))
                                  and math.isclose(observed, float(wanted), rel_tol=2e-11, abs_tol=2e-13),
                                  "Independent numeric table cell", context)
                    absolute = max(absolute, abs(observed-float(wanted)))
                    numeric_cells += 1
                else:
                    audit.require(observed == wanted, "Independent exact table cell", context)
                    exact_cells += 1
        audit.counts["export_numeric_cells_verified"] += numeric_cells
        audit.counts["export_exact_cells_verified"] += exact_cells
        audit.counts["export_rows_verified"] += len(actual)
        audit.counts["export_tables_verified"] += 1
        result[name] = {"rows": len(actual), "columns": len(fields), "numeric_cells": numeric_cells,
                        "exact_cells": exact_cells, "maximum_absolute_numeric_difference": absolute}
    return result
