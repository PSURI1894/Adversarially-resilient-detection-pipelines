# ══════════════════════════════════════════════════════════════
# Makefile — Build the ARDP research paper
# Requires: pdflatex + bibtex (TeX Live / MiKTeX)
# ══════════════════════════════════════════════════════════════

PAPER   = main
LATEX   = pdflatex
BIBTEX  = bibtex
FLAGS   = -interaction=nonstopmode -halt-on-error

.PHONY: all pdf clean distclean figures

# Default target
all: pdf

# Full two-pass build with bibliography
pdf: figures
	$(LATEX) $(FLAGS) $(PAPER).tex
	$(BIBTEX) $(PAPER)
	$(LATEX) $(FLAGS) $(PAPER).tex
	$(LATEX) $(FLAGS) $(PAPER).tex
	@echo "✓  $(PAPER).pdf built successfully"

# Copy experiment figures into paper/figures/
figures:
	@python -c "import shutil, pathlib; \
	src=pathlib.Path('../experiments/results/figures'); \
	dst=pathlib.Path('figures'); \
	dst.mkdir(exist_ok=True); \
	[shutil.copy2(f, dst/f.name) for f in src.glob('*.pdf')] if src.exists() else None; \
	print('Figures synced.')"

# Quick single pass (no bibliography)
quick:
	$(LATEX) $(FLAGS) $(PAPER).tex

# Remove build artifacts
clean:
	rm -f $(PAPER).aux $(PAPER).bbl $(PAPER).blg \
	       $(PAPER).log $(PAPER).out $(PAPER).toc \
	       $(PAPER).lof $(PAPER).lot texput.log

# Remove everything including the PDF
distclean: clean
	rm -f $(PAPER).pdf
