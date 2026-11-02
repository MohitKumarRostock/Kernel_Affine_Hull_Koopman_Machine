#!/usr/bin/env python3
"""Compile the article and supplement from supplied scientific tables."""
from pathlib import Path
import shutil,subprocess,sys
root=Path(__file__).resolve().parents[1]
latex=shutil.which('pdflatex');bib=shutil.which('bibtex.original') or shutil.which('bibtex')
if not latex:sys.exit('pdfLaTeX is required.')
for name in ('main','supplement'):
 cmd=[latex,'-interaction=nonstopmode','-halt-on-error',name+'.tex']
 subprocess.run(cmd,cwd=root,check=True)
 if bib:subprocess.run([bib,name],cwd=root,check=True)
 elif not (root/(name+'.bbl')).exists():sys.exit('BibTeX or a supplied bibliography (.bbl) is required.')
 for _ in range(2):subprocess.run(cmd,cwd=root,check=True)
print('Built main.pdf and supplement.pdf.')
