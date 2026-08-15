#!/usr/bin/env bash
# Build the paper. Markdown sections are the source of truth; pandoc converts
# them to .tex fragments that main.tex \input's. Nothing in build/ is hand-edited.
#
#   bash scripts/build_paper.sh          # convert + compile if latex present
#   bash scripts/build_paper.sh --tex    # convert only
set -euo pipefail
cd "$(dirname "$0")/../docs/paper"
mkdir -p build

SECTIONS=(01_abstract 02_introduction 03_background_related 04_methodology
          05_results 05b_power 06_threats 07_discussion
          08_appendix_conduct 09_appendix_incidents 10_appendix_tables)

for s in "${SECTIONS[@]}"; do
  if [ ! -f "$s.md" ]; then
    echo "% section $s.md not written yet" > "build/$s.tex"
    echo "  SKIP $s (no markdown yet)"
    continue
  fi
  # --natbib keeps \cite{} keys intact so citations resolve against references.bib.
  # --top-level-division=section maps markdown h1 to \section.
  pandoc "$s.md" -o "build/$s.tex" \
    --natbib --top-level-division=section --wrap=preserve
  echo "  ok   $s"
done

# pandoc 2.9 emits `height=\textheight` alongside any width attribute, which
# lets a wide figure scale until it fills the page vertically. Drop the height
# constraint so width alone governs, and pin figures near their text.
sed -i 's/,height=\\textheight//g' build/*.tex
sed -i 's/\\begin{figure}$/\\begin{figure}[htbp]/' build/*.tex

# Strip the abstract fragment's section heading: it sits inside \begin{abstract}.
if [ -f build/01_abstract.tex ]; then
  sed -i '/^\\section{/d;/^\\hypertarget/d;/^}\\label{/d' build/01_abstract.tex
fi

[ "${1:-}" = "--tex" ] && { echo "tex fragments in docs/paper/build/"; exit 0; }

if ! command -v pdflatex >/dev/null; then
  echo
  echo "pdflatex not installed, stopping after tex generation."
  echo "  sudo apt-get install -y texlive-latex-recommended texlive-fonts-recommended texlive-latex-extra"
  exit 0
fi

pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
bibtex main >/dev/null 2>&1 || true
pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
echo "built docs/paper/main.pdf"
