# MedShop-RAG IEEE LaTeX Source

IEEE 2-column conference paper for ISIR-eCom 2026 Workshop at IEEE BigData 2026.

## Files

| File | Description |
|------|-------------|
| `medshop_rag_ieee.tex` | Main LaTeX source (single file, self-contained) |
| `Makefile` | Build script |

## Requirements

- TeX Live 2020+ (or MiKTeX 2020+)
- Packages: `IEEEtran`, `amsmath`, `amssymb`, `booktabs`, `array`, `tabularx`,
  `multirow`, `graphicx`, `xcolor`, `hyperref`, `tikz` (with `shapes.geometric`,
  `arrows.meta`, `positioning`, `fit`, `backgrounds`), `cite`, `url`, `microtype`

All packages are included in a standard TeX Live full installation.

## Build

```bash
make          # produces medshop_rag_ieee.pdf (runs pdflatex twice)
make clean    # remove auxiliary files
```

Or manually:

```bash
pdflatex medshop_rag_ieee.tex
pdflatex medshop_rag_ieee.tex   # second pass for cross-refs
```

## Before Submission

- [ ] Replace `[Author Name(s) — To be Added]` with actual author names
- [ ] Replace `[Affiliation — To be Added]` with actual affiliation
- [ ] Replace `[email@institution.edu]` with actual email
- [ ] Fill in `\section*{Acknowledgments}` if applicable
- [ ] Verify all 10 DOI links are correct
- [ ] Confirm `SYNTHETIC_DATA` flags are intentional (they are — do not remove)
- [ ] Check page count does not exceed 10 pages in IEEE 2-column format
- [ ] Upload to IEEE PDF eXpress for compliance check before submission

## Key Design Notes

- `\synflag` macro prints `[SYNTHETIC_DATA]` in red bold — required by project integrity rules
- `\simflag{...}` prints simulation flags in red italic
- Figure 1 is a TikZ flowchart (no external image file needed)
- All three tables use `booktabs` rules (`\toprule`, `\midrule`, `\bottomrule`)
- References use `\begin{thebibliography}` (no .bib file needed)
- Cite keys match the inline reference list: `medagent`, `shoppingcomp`, `sia`,
  `toolcalling`, `pgxrag`, `medibot`, `cartintel`, `sidekick`, `splextraction`, `drugbench`
