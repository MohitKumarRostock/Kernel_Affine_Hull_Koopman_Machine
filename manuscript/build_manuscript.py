"""Compile the article and check its LaTeX and bibliography diagnostics.

Run ``python3 build_manuscript.py`` with a configured TeX Live installation.
For an unpacked distribution, pass each TeX tree using ``--texmf-root PATH``.
The build requires no data download or model fitting.
"""

from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess


ROOT = Path(__file__).resolve().parent
BUILD = ROOT / "build"
JOB = "manuscript"


def run(command, name, env, cwd=ROOT):
    with (BUILD / name).open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT
        )
    if result.returncode:
        raise RuntimeError(
            f"{name}: command failed with exit code {result.returncode}; "
            "see the corresponding file in build/"
        )


def tex_environment(roots):
    env = os.environ.copy()
    env["BIBINPUTS"] = str(ROOT) + os.pathsep + env.get("BIBINPUTS", "")
    if not roots:
        return env, ["pdflatex"], "main.tex"

    trees = [Path(root).expanduser().resolve() for root in roots]
    if any(not tree.is_dir() for tree in trees):
        raise ValueError("Each --texmf-root must name an existing TeX tree")
    env["TEXINPUTS"] = ".:" + ":".join(str(p / "tex") + "//" for p in trees) + ":"
    for variable, subdir in (
        ("TFMFONTS", "tfm"), ("T1FONTS", "type1"), ("VFFONTS", "vf"),
        ("TEXFONTMAPS", "map"), ("ENCFONTS", "enc"),
    ):
        env[variable] = ":".join(str(p / "fonts" / subdir) + "//" for p in trees) + ":"
    env["BSTINPUTS"] = ".:" + ":".join(str(p / "bibtex" / "bst") + "//" for p in trees) + ":"
    ini = next(
        (p / "tex" / "latex" / "tex-ini-files" / "pdflatex.ini" for p in trees
         if (p / "tex" / "latex" / "tex-ini-files" / "pdflatex.ini").is_file()),
        None,
    )
    if ini is None:
        raise FileNotFoundError("pdflatex.ini was not found in the supplied TeX trees")
    fmt = BUILD / "pdflatex.fmt"
    run(
        ["pdftex", "-ini", "-etex", "-interaction=nonstopmode", "-halt-on-error",
         "-jobname=pdflatex", "-progname=pdflatex", f"-output-directory={BUILD}", str(ini)],
        "format.stdout.log", env,
    )
    source = (
        r"\pdfmapfile{+pdftex35.map}\pdfmapfile{+cm.map}\pdfmapfile{+symbols.map}"
        r"\pdfmapfile{+lm.map}\pdfmapfile{=font_embed.map}\input{main.tex}"
    )
    return env, ["pdftex", f"-fmt={fmt}", "-progname=pdflatex"], source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--texmf-root", action="append", default=[], metavar="PATH")
    args = parser.parse_args()
    BUILD.mkdir(exist_ok=True)
    env, texcmd, source = tex_environment(args.texmf_root)
    common = ["-interaction=nonstopmode", "-halt-on-error", "-file-line-error",
              f"-jobname={JOB}", f"-output-directory={BUILD}"]
    run(texcmd + common + [source], "compile_pass1.stdout.log", env)
    run(["bibtex", JOB], "bibtex.stdout.log", env, cwd=BUILD)
    for index in (2, 3, 4):
        run(texcmd + common + [source], f"compile_pass{index}.stdout.log", env)

    latex_log = (BUILD / f"{JOB}.log").read_text(errors="replace")
    bib_log = (BUILD / f"{JOB}.blg").read_text(errors="replace")
    fatal = [line for line in latex_log.splitlines()
             if line.startswith("!") or "Undefined control sequence" in line]
    unresolved = [line for line in latex_log.splitlines()
                  if "undefined" in line.lower() or "multiply defined" in line.lower()]
    overfull = [line for line in latex_log.splitlines() if "Overfull" in line]
    warnings = [line for line in latex_log.splitlines() if "Warning" in line]
    bib_warnings = [line for line in bib_log.splitlines() if "Warning--" in line]
    passed = not (fatal or unresolved or overfull or warnings or bib_warnings)
    output = ROOT / f"{JOB}.pdf"
    if passed:
        shutil.copy2(BUILD / f"{JOB}.pdf", output)
    report = {
        "status": "PASS" if passed else "NEEDS_REVIEW",
        "pdf": output.name if passed else f"build/{JOB}.pdf",
        "fatal": fatal,
        "unresolved_references_or_citations": unresolved,
        "overfull_boxes": overfull,
        "latex_warnings": warnings,
        "bibliography_warnings": bib_warnings,
        "sha256": hashlib.sha256((BUILD / f"{JOB}.pdf").read_bytes()).hexdigest(),
    }
    (BUILD / "build_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
