================================================================
   How to compile Milestone3_Report.tex into a PDF
================================================================

The source file is `Milestone3_Report.tex` and the figures it
references live in the `figures/` subdirectory. You have two
options for compiling.

----------------------------------------------------------------
OPTION 1: Overleaf (recommended, no installation needed)
----------------------------------------------------------------
1. Go to https://www.overleaf.com/ and sign in.
2. Click "New Project" -> "Upload Project".
3. Zip the entire `report/` folder (including the `figures/`
   subdirectory) and upload the zip.
4. Overleaf will detect `Milestone3_Report.tex` as the main file
   automatically. If not, set it via Menu -> Main document.
5. Click "Recompile". The PDF appears in the right pane.
6. Download the PDF via the download arrow at the top.

----------------------------------------------------------------
OPTION 2: Local install (Windows: MiKTeX)
----------------------------------------------------------------
1. Install MiKTeX from https://miktex.org/download (default
   options are fine - "install missing packages on the fly").
2. Open PowerShell in this folder.
3. Run:
       pdflatex Milestone3_Report.tex
       pdflatex Milestone3_Report.tex
   (Two passes are needed for the table of contents and
   cross-references to resolve.)
4. The PDF is produced as `Milestone3_Report.pdf` in this folder.

----------------------------------------------------------------
What the file uses
----------------------------------------------------------------
The document uses only standard CTAN packages:
    geometry, graphicx, booktabs, amsmath, amssymb, listings,
    xcolor, caption, subcaption, float, enumitem, hyperref,
    url, array, tabularx, inputenc, fontenc.
All of these ship with a default MiKTeX or TeX Live installation
and are available out of the box on Overleaf.

Figures are referenced as `figures/<name>.pdf` (vector versions)
and the corresponding `.png` versions are also in `figures/` if
you ever need to switch.
