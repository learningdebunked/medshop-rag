# MedShop-RAG — IEEE LaTeX Build
#
# Requirements: TeX Live 2020+ (or MiKTeX 2020+) with IEEEtran, TikZ, booktabs
#
# pdflatex is run twice to resolve cross-references and figure/table numbers.

MAIN = medshop_rag_ieee
PDFLATEX = pdflatex -interaction=nonstopmode -halt-on-error

.PHONY: all clean

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex
	$(PDFLATEX) $(MAIN).tex
	$(PDFLATEX) $(MAIN).tex   # second pass for cross-references

clean:
	rm -f $(MAIN).aux $(MAIN).log $(MAIN).out \
	      $(MAIN).toc $(MAIN).lof $(MAIN).lot \
	      $(MAIN).bbl $(MAIN).blg
