#!/usr/bin/env python3
"""Display-only figures from the completed, replay-audited frozen analysis.

This file is deliberately outside the scientific source freeze. It neither
loads task predictions nor recomputes inference. Its inputs are the completed
execution, protocol, audit report and aggregate summary. Rendering is refused
until the publication audit has passed with selected-model replay enabled.

Example, after the run and publication audit finish::

    python extensions/direct_reconstruction/build_figures.py RESULTS_DIRECTORY

The required output is a 6.8 x 3.6 inch vector PDF of all eighteen frozen
paired contrasts. ``--include-natural`` additionally requests an explicitly
available-case accuracy/cost plot; that optional display is omitted if its
seventeen labels cannot be placed without overlap. Both output PDFs and all
read inputs are SHA256-recorded in FIGURE_MANIFEST.json.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


LAWS = ("nonlinear_ar", "vanderpol", "duffing")
PRIMARY_CONTROLS = (
    "raw_rbf", "gate_kahm", "reconstruction_affine",
    "reconstruction_locality", "reconstruction_rbf_local", "nonkahm_policy",
)
CANDIDATE = "reconstruction_kahm"
LAW_LABELS = {"nonlinear_ar": "Nonlinear AR", "vanderpol": "Van der Pol", "duffing": "Duffing"}
CONTROL_LABELS = {
    "raw_rbf": "Raw RBF",
    "gate_kahm": "Gate-trained KAHM",
    "reconstruction_affine": "Shape-trained affine",
    "reconstruction_locality": "Shape-trained locality",
    "reconstruction_rbf_local": "Shape-trained RBF-local",
    "nonkahm_policy": "Selected non-KAHM",
}
METHOD_LABELS = {
    "ridge": "Ridge", "raw_rbf": "Raw RBF", "raw_hgb": "Boosted trees",
    "seasonal_naive": "Seasonal naive", "nonkahm_policy": "Selected non-KAHM",
    "fixed_kahm": "Fixed KAHM", "gate_kahm": "Gate KAHM",
    "reconstruction_kahm": "Reconstruction KAHM",
    "fixed_affine": "Fixed affine", "gate_affine": "Gate affine",
    "reconstruction_affine": "Shape affine",
    "fixed_locality": "Fixed locality", "gate_locality": "Gate locality",
    "reconstruction_locality": "Shape locality",
    "fixed_rbf_local": "Fixed RBF-local", "gate_rbf_local": "Gate RBF-local",
    "reconstruction_rbf_local": "Shape RBF-local",
}
OUTCOME_COLORS = {"positive": "#24658A", "negative": "#AD734B", "unresolved": "#7C858D"}
SCALE = 1000.0


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path, relative_to=None):
    path = Path(path).resolve()
    name = str(path.relative_to(relative_to)) if relative_to is not None else str(path)
    return {"path": name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def load_completed_inputs(results):
    """Read aggregate evidence only after completion and replay audit gates."""
    execution_path = results / "execution.json"
    if not execution_path.is_file():
        raise ValueError("No completed execution.json; figures cannot read an active run")
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    if execution.get("status") not in {"complete", "complete_with_failures"} or not execution.get("completed_utc"):
        raise ValueError("Execution is still active; no figures were generated")
    audit_path = results / "audit_report.json"
    if not audit_path.is_file():
        raise ValueError("The completed publication audit is required before figures")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("passed") is not True or audit.get("selected_models_replayed") is not True:
        raise ValueError("A passing audit with selected-model replay is required")
    summary_path, protocol_path = results / "summary.json", results / "protocol.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if summary.get("audit_passed") is not True:
        raise ValueError("The aggregate summary does not record a passing audit")
    if summary.get("available_method_outputs") != audit.get("available_method_outputs"):
        raise ValueError("Summary and audit disagree about the available output roster")
    if audit.get("public_freeze_commit") != execution.get("public_freeze_commit"):
        raise ValueError("Audit and execution identify different public freezes")
    if tuple(protocol.get("laws", ())) != LAWS or tuple(protocol.get("primary_controls", ())) != PRIMARY_CONTROLS:
        raise ValueError("The declared law/contrast order differs from this display specification")
    if protocol.get("primary_candidate") != CANDIDATE:
        raise ValueError("Unexpected primary candidate")
    if len(protocol.get("methods", ())) != 17 or set(protocol["methods"]) != set(METHOD_LABELS):
        raise ValueError("The complete seventeen-method display roster is required")
    paths = [execution_path, audit_path, summary_path, protocol_path]
    freeze_path = results / "freeze.json"
    if freeze_path.is_file():
        paths.append(freeze_path)
    return summary, protocol, audit, [file_record(path, results) for path in paths]


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def primary_rows(summary):
    rows = summary.get("primary_contrasts", [])
    if len(rows) != 18:
        raise ValueError("The figure must contain all eighteen primary contrasts")
    mapping = {(row["law"], row["control"]): row for row in rows}
    if len(mapping) != 18 or set(mapping) != {(law, control) for law in LAWS for control in PRIMARY_CONTROLS}:
        raise ValueError("Missing, duplicate, or unexpected primary contrast")
    for row in rows:
        if row.get("complete") is not True or row.get("candidate") != CANDIDATE:
            raise ValueError("Every displayed primary contrast must be complete")
        if (row.get("n_repetitions"), row.get("degrees_of_freedom"), row.get("family_size")) != (12, 11, 18):
            raise ValueError("Unexpected repetition count or simultaneous-interval family")
        if row.get("family_alpha") != 0.05:
            raise ValueError("Unexpected family alpha")
        values = [row.get(key) for key in ("mean_difference", "interval_low", "interval_high")]
        if not all(finite_number(value) for value in values):
            raise ValueError("A primary difference or interval is not finite")
        mean, low, high = values
        if not low <= mean <= high:
            raise ValueError("A primary interval does not contain its reported estimate")
        outcome = "positive" if low > 0 else "negative" if high < 0 else "unresolved"
        if outcome != row.get("conclusion"):
            raise ValueError("Primary interval and its audited conclusion disagree")
    return mapping


def configure_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8,
        "axes.labelsize": 8, "axes.titlesize": 9,
        "xtick.labelsize": 7.3, "ytick.labelsize": 7.6,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "mathtext.fontset": "dejavusans", "axes.unicode_minus": True,
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })
    return matplotlib, plt


def save_vector_pdf(figure, path, title, subject):
    figure.savefig(path, format="pdf", dpi=300, metadata={
        "Title": title, "Subject": subject,
        "Creator": "KAHM direct reconstruction: display-only audited figure builder",
        "CreationDate": None, "ModDate": None,
    })


def draw_primary(summary, output, plt):
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    mapping = primary_rows(summary)
    figure, axes = plt.subplots(1, 3, figsize=(6.8, 3.6), sharey=True)
    figure.subplots_adjust(left=0.245, right=0.985, bottom=0.265, top=0.86, wspace=0.24)
    plotted = []
    for panel, (law, axis) in enumerate(zip(LAWS, axes)):
        selected = [mapping[(law, control)] for control in PRIMARY_CONTROLS]
        low = min(0.0, min(row["interval_low"] for row in selected) * SCALE)
        high = max(0.0, max(row["interval_high"] for row in selected) * SCALE)
        span = high - low
        padding = 0.09 * span if span > 0 else 1.0
        axis.set_xlim(low - padding, high + padding)
        axis.set_ylim(5.65, -0.65)
        for position, row in enumerate(selected):
            if position % 2 == 0:
                axis.axhspan(position - 0.45, position + 0.45, color="#F4F6F7", zorder=0)
            mean, lo, hi = [row[key] * SCALE for key in ("mean_difference", "interval_low", "interval_high")]
            color = OUTCOME_COLORS[row["conclusion"]]
            axis.errorbar(mean, position, xerr=[[mean - lo], [hi - mean]],
                          fmt="o", color=color, ecolor=color, markersize=4.3,
                          markeredgecolor="white", markeredgewidth=0.4,
                          capsize=2.2, elinewidth=1.1, capthick=1.0, zorder=4)
            plotted.append({"law": law, "control": row["control"],
                            "candidate": CANDIDATE, "mean_difference_times_1000": mean,
                            "interval_low_times_1000": lo, "interval_high_times_1000": hi,
                            "conclusion": row["conclusion"]})
        axis.axvline(0, color="#424950", linewidth=0.8, linestyle=(0, (3, 2)), zorder=2)
        axis.set_title(f"({chr(97 + panel)}) {LAW_LABELS[law]}", pad=12, fontsize=9)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=3, steps=[1, 2, 2.5, 5, 10]))
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        axis.grid(axis="x", color="#DBE0E4", linewidth=0.45, zorder=1)
        axis.tick_params(axis="y", length=0, pad=7)
        axis.tick_params(axis="x", length=3, color="#707A83")
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.spines["bottom"].set_color("#9BA3AA")
        axis.spines["bottom"].set_linewidth(0.65)
    axes[0].set_yticks(range(6), [CONTROL_LABELS[control] for control in PRIMARY_CONTROLS])
    figure.text(0.615, 0.15,
                r"Paired MSE difference: comparator $-$ reconstruction KAHM ($\times 10^3$)",
                ha="center", va="center", fontsize=8)
    handles = [Line2D([0], [0], marker="o", color=OUTCOME_COLORS[outcome],
                      linewidth=1.1, markersize=4, label=label)
               for outcome, label in (("positive", "CI favors KAHM"),
                                      ("negative", "CI favors comparator"),
                                      ("unresolved", "CI includes zero"))]
    figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.59, 0.035),
                  ncol=3, frameon=False, fontsize=7.0, handlelength=1.5,
                  columnspacing=1.25, handletextpad=0.5)
    path = output / "primary_contrasts.pdf"
    caption = (
        "Reserved synthetic evidence for forecast-directed KAHM reconstruction. "
        "Each point is the mean paired comparator-minus-reconstruction-KAHM MSE difference "
        "across twelve independent repetitions of the indicated law; bars are the frozen "
        "two-sided Bonferroni t intervals (11 degrees of freedom, family alpha 0.05 over "
        "all eighteen contrasts). Differences and interval endpoints are multiplied by 1,000; "
        "these are absolute MSE differences, not ratio or percentage intervals. "
        "Positive values favor reconstruction-trained KAHM. Blue intervals lie above zero, "
        "muted orange intervals lie below zero, and gray intervals include zero. "
        "Horizontal scales differ across panels. Coverage concerns this new contrast family "
        "under the paired-difference normal approximation, not the full manuscript."
    )
    save_vector_pdf(figure, path, "Forecast-directed KAHM: eighteen reserved paired contrasts", caption)
    plt.close(figure)
    return {"file": file_record(path, output), "size_inches": [6.8, 3.6],
            "vector_pdf": True, "statistics_source": "summary.json: primary_contrasts",
            "display_scale": SCALE, "law_order": list(LAWS),
            "control_order": list(PRIMARY_CONTROLS), "plotted_rows": plotted, "caption": caption}


def natural_rows(summary, protocol):
    """Require the declared seven available tasks and three complete sources."""
    methods = protocol["methods"]
    accuracy = {row["method"]: row for row in summary["method_aggregates"]
                if row["suite"] == "natural" and row["group"] == "available_collection"}
    costs = {row["method"]: row for row in summary["cost_aggregates"] if row["suite"] == "natural"}
    if set(accuracy) != set(methods) or set(costs) != set(methods):
        raise ValueError("Optional natural plot needs all seventeen available-case procedures")
    source_sets, rows = [], []
    for method in methods:
        a, c = accuracy[method], costs[method]
        if a.get("task_count") != 7 or c.get("tasks") != 7:
            raise ValueError("Optional natural plot requires exactly seven available tasks per procedure")
        sources = {row["source_key"] for row in summary["natural_collection_metrics"]
                   if row["method"] == method and row.get("complete") is True
                   and finite_number(row.get("mean_scaled_mse"))}
        source_sets.append(sources)
        x, y = c.get("mean_charged_wall_seconds"), a.get("mean_loss")
        if not finite_number(x) or x <= 0 or not finite_number(y) or y < 0:
            raise ValueError("Optional natural plot needs finite nonnegative losses and positive times")
        rows.append({"method": method, "mean_warm_seconds": x, "available_scaled_mse": y})
    if len(source_sets[0]) != 3 or any(sources != source_sets[0] for sources in source_sets):
        raise ValueError("Optional natural plot needs the same three available collections for every method")
    return rows, sorted(source_sets[0])


def method_style(name):
    raw = {"ridge": ("#454C52", "v"), "raw_rbf": ("#454C52", "D"),
           "raw_hgb": ("#454C52", "P"), "seasonal_naive": ("#454C52", "X"),
           "nonkahm_policy": ("#171C20", "*")}
    if name in raw:
        return raw[name]
    arm, gate = name.split("_", 1)
    colors = {"kahm": "#24658A", "affine": "#64865C", "locality": "#AD734B", "rbf_local": "#8173A4"}
    return colors[gate], {"fixed": "o", "gate": "^", "reconstruction": "s"}[arm]


def place_number_labels(axis, figure, rows):
    """Place all labels inside the axes without text/point overlap, or omit plot."""
    from matplotlib.transforms import Bbox
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    bounds = axis.get_window_extent(renderer)
    point_xy = [axis.transData.transform((row["mean_warm_seconds"], row["available_scaled_mse"])) for row in rows]
    point_boxes = [Bbox.from_bounds(x - 3, y - 3, 6, 6) for x, y in point_xy]
    label_boxes = []
    directions = ((1, 1), (1, -1), (-1, 1), (-1, -1), (1, 0), (-1, 0), (0, 1), (0, -1))
    for index, (row, point) in enumerate(zip(rows, point_xy), start=1):
        text = axis.annotate(f"{index:02d}",
                             (row["mean_warm_seconds"], row["available_scaled_mse"]),
                             xytext=(0, 0), textcoords="offset points", fontsize=6.8,
                             ha="center", va="center", color="#30383E", zorder=6,
                             bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6})
        chosen = None
        for radius in (9, 15, 23, 33, 45, 60, 76):
            for direction in directions:
                offset = (radius * direction[0], radius * direction[1])
                text.set_position(offset)
                box = text.get_window_extent(renderer).expanded(1.12, 1.18)
                inside = (box.x0 >= bounds.x0 + 2 and box.x1 <= bounds.x1 - 2
                          and box.y0 >= bounds.y0 + 2 and box.y1 <= bounds.y1 - 2)
                if inside and not any(box.overlaps(other) for other in label_boxes + point_boxes):
                    chosen = (offset, box)
                    break
            if chosen is not None:
                break
        if chosen is None:
            return False
        offset, box = chosen
        label_boxes.append(box)
        if math.hypot(*offset) > 13:
            axis.annotate("", (row["mean_warm_seconds"], row["available_scaled_mse"]),
                          xytext=offset, textcoords="offset points",
                          arrowprops={"arrowstyle": "-", "color": "#B2B9BF", "linewidth": 0.45}, zorder=2)
    return True


def draw_natural(summary, protocol, output, plt):
    from matplotlib.lines import Line2D
    rows, sources = natural_rows(summary, protocol)
    figure = plt.figure(figsize=(6.8, 4.2))
    axis = figure.add_axes([0.105, 0.205, 0.545, 0.69])
    x_values, y_values = [row["mean_warm_seconds"] for row in rows], [row["available_scaled_mse"] for row in rows]
    axis.set_xscale("log")
    log_low, log_high = math.log10(min(x_values)), math.log10(max(x_values))
    x_padding = max(0.16, 0.06 * (log_high - log_low))
    axis.set_xlim(10 ** (log_low - x_padding), 10 ** (log_high + x_padding))
    y_span = max(y_values) - min(y_values)
    y_padding = max(0.14 * y_span, 0.02 * max(y_values), 1e-8)
    axis.set_ylim(max(0.0, min(y_values) - y_padding), max(y_values) + y_padding)
    handles = []
    for index, row in enumerate(rows, start=1):
        color, marker = method_style(row["method"])
        size = 45 if row["method"] == "nonkahm_policy" else 27
        axis.scatter(row["mean_warm_seconds"], row["available_scaled_mse"],
                     s=size, marker=marker, color=color, edgecolor="white", linewidth=0.4, zorder=4)
        handles.append(Line2D([0], [0], marker=marker, color=color, linestyle="none",
                              markersize=5, label=f"{index:02d}  {METHOD_LABELS[row['method']]}"))
    axis.set_xlabel("Mean measured procedure time (s)\n(equal weight over 7 available tasks)", fontsize=8)
    axis.set_ylabel("Available-case scaled MSE\n(equal collection, then series)", fontsize=8)
    axis.grid(which="major", color="#DCE1E5", linewidth=0.5)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["bottom", "left"]].set_color("#9BA3AA")
    axis.tick_params(axis="both", labelsize=7.3)
    figure.text(0.51, 0.96, "Available natural accuracy and measured cost", ha="center", fontsize=9.5)
    figure.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.67, 0.90),
                  frameon=False, fontsize=7.0, handletextpad=0.55, labelspacing=0.58, borderaxespad=0)
    if not place_number_labels(axis, figure, rows):
        plt.close(figure)
        return {"omitted": True, "reason": "All seventeen numbered labels could not be placed without text/point overlap"}
    path = output / "natural_available_accuracy_cost.pdf"
    caption = (
        "Supplemental available-case natural accuracy and measured cost for all seventeen procedures. "
        "Accuracy averages A-seasonal-scaled MSE equally over series within each available collection, "
        "then equally over the three available collections (seven series). The horizontal coordinate "
        "is the arithmetic mean measured procedure time across those seven available tasks, with "
        "equal task weights and a logarithmic axis. Thus accuracy and time use explicitly different "
        "weighting rules. Time includes the implementation's charged fitting, selection and eager "
        "query dependencies and selected-state packing/serialization on this host; candidate "
        "dependencies are counted once for the non-KAHM policy. Downloads, process imports, data "
        "preparation and artifact writes are excluded. Numbered points identify all procedures; "
        "overlapping points share the same plotted coordinates. There are no uncertainty intervals "
        "or source-population claims. The planned four-collection aggregate remains undefined because "
        "the three frozen rideshare tasks were unavailable; this figure does not replace it."
    )
    save_vector_pdf(figure, path, "Available natural accuracy and measured procedure cost", caption)
    plt.close(figure)
    return {"file": file_record(path, output), "size_inches": [6.8, 4.2], "vector_pdf": True,
            "statistics_source": "summary.json: method_aggregates and cost_aggregates",
            "available_series": 7, "available_collections": sources,
            "accuracy_weighting": "equal available collection, then equal available series",
            "time_weighting": "equal weight over seven available tasks",
            "logarithmic_x_axis": True, "plotted_rows": rows, "caption": caption}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="completed result directory containing the publication audit")
    parser.add_argument("--output", type=Path, help="figure directory; defaults to RESULTS/figures")
    parser.add_argument("--include-natural", action="store_true", help="also try the optional all-method available-case cost display")
    arguments = parser.parse_args()
    results = arguments.results.resolve()
    summary, protocol, audit, inputs = load_completed_inputs(results)
    # All completion/audit checks precede matplotlib setup and output creation.
    primary_rows(summary)
    output = arguments.output.resolve() if arguments.output else results / "figures"
    output.mkdir(parents=True, exist_ok=True)
    matplotlib, plt = configure_matplotlib()
    figures = {"primary_contrasts": draw_primary(summary, output, plt)}
    if arguments.include_natural:
        try:
            figures["natural_available_accuracy_cost"] = draw_natural(summary, protocol, output, plt)
        except ValueError as error:
            figures["natural_available_accuracy_cost"] = {"omitted": True, "reason": str(error)}
    manifest = {
        "schema_version": "direct-reconstruction-display-v1",
        "scope": "Display only; no new fitting, model selection, loss calculation or statistical inference",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "public_freeze_commit": audit["public_freeze_commit"],
        "publication_audit_passed": True, "selected_models_replayed": True,
        "builder": file_record(Path(__file__)), "matplotlib_version": matplotlib.__version__,
        "input_directory": str(results), "inputs": inputs, "figures": figures,
        "outputs": [figure["file"] for figure in figures.values() if "file" in figure],
        "visual_review": "PDFs require visual inspection after rendering; automated label placement is not a substitute",
    }
    manifest_path = output / "FIGURE_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "outputs": manifest["outputs"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
