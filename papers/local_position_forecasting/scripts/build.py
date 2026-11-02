#!/usr/bin/env python3
"""Build the manuscript and supplement with the supplied TMLR files."""
from pathlib import Path
import shutil
import subprocess
import sys

root=Path(__file__).resolve().parents[1]
latex=shutil.which('pdflatex')
bibtex=shutil.which('bibtex') or shutil.which('bibtex.original')
if not latex:
    sys.exit('pdfLaTeX is required. Install a TeX distribution before building.')
for name in ('main','supplement'):
    command=[latex,'-interaction=nonstopmode','-halt-on-error',name+'.tex']
    subprocess.run(command,cwd=root,check=True)
    if bibtex:
        subprocess.run([bibtex,name],cwd=root,check=True)
    elif not (root/(name+'.bbl')).is_file():
        sys.exit('BibTeX or a supplied .bbl file is required.')
    subprocess.run(command,cwd=root,check=True)
    subprocess.run(command,cwd=root,check=True)
print('Built main.pdf and supplement.pdf')
