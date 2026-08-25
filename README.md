# deepseek-ocr-pdf

Replace a PDF's text layer with DeepSeek-OCR-2 output, positioned where the
text actually appears on the page. Everything runs locally through Ollama.

## Install

```bash
brew install ocrmypdf qpdf          # ghostscript and tesseract come along
ollama pull hf.co/sabafallah/DeepSeek-OCR-2-GGUF:Q8_0

# venv for development and tests
uv venv && uv pip install -e ".[dev]"
# then either activate it or use uv run:
source .venv/bin/activate
# or
uv run deepseek-ocr-pdf --help

# global CLI (puts deepseek-ocr-pdf on your PATH via ~/.local/bin)
uv tool install -e .
```

## Use

```bash
deepseek-ocr-pdf scanned.pdf searchable.pdf
```

Defaults to `--redo-ocr`: any existing invisible text layer is stripped and replaced, but the original page image is preserved losslessly (verified: image JPEG MD5 identical before/after). To rasterize every page and discard all text (e.g. to fix vector-text PDFs), pass `--force-ocr`. To OCR only pages with no text at all, pass `--skip-text`.

Also defaults to `--output-type pdf`, which rewrites nothing but the text layer. ocrmypdf's own default is `--output-type auto`: it attempts a declarations-only PDF/A conversion with pikepdf, validates it with veraPDF, and falls back to Ghostscript when veraPDF is missing or the file does not validate.

That Ghostscript pass costs less image quality than it appears to. With ocrmypdf's settings (`LeaveColorUnchanged`, `auto` compression), JPEG streams pass through byte for byte, and a Flate bilevel image re-encoded to CCITT G4 came back pixel-identical. What it does change is `/Interpolate`, which PDF/A forbids and Ghostscript strips from every image — scanners including the ScanSnap set it, and dropping it makes some viewers render a zoomed scan more harshly. It also rebuilds the document structure around the images.

So the choice here is conservatism, not rescue: a tool that replaces an invisible text layer should not also rewrite the file. Archival output stays one flag away — pass `--output-type pdfa` (or `auto`, `pdfa-1`, `pdfa-2`, `pdfa-3`).

Unrecognized options go straight to ocrmypdf, so `--sidecar out.txt`,
`--jobs 4`, and `--deskew` work as usual.

| Option | Default | Purpose |
| --- | --- | --- |
| `--model` | `hf.co/sabafallah/DeepSeek-OCR-2-GGUF:Q8_0` | Ollama model tag |
| `--ollama-host` | `http://localhost:11434` | Ollama base URL |
| `--timeout` | `300` | Seconds allowed per page |
| `--max-dim` | none | Shrink pages to this longest edge before OCR |
| `--no-coverage-guard` | off | Skip the dropped-text repair pass |
| `--no-line-split-pass` | off | Measure lines with Tesseract instead of a second model pass. Halves the time, moves the odd line break by a word |
| `--output-type` | `pdf` | Passed to ocrmypdf. `pdf` leaves the file untouched apart from the text layer; `pdfa` and friends rebuild it via Ghostscript and strip `/Interpolate` |

Leave `--max-dim` alone unless pages are so large they exhaust memory.
Shrinking costs accuracy for nothing: at 1024px the model merges several lines
into one box, at 1700px it returns one box per line, and both take the same
time because it retiles internally either way.

## How it works

ocrmypdf rasterizes each page and applies the existing-text policy. The plugin
then makes two model calls per page, because no single prompt gives both halves
of what a positioned text layer needs:

- `Convert the document to markdown.` reads the text. This is the text of
  record; nothing else is trusted to spell a word.
- `OCR this image.` measures the printed lines. It returns one box per line,
  which the markdown prompt does not.

The markdown text is then laid back onto the measured lines, each line taking a
share of the words proportional to the characters read on it.

Tesseract runs as a *detector only* — its text is thrown away — and does two
jobs with the same boxes: it finds text both prompts missed, which is re-read as
tight crops and merged, and it supplies line geometry wherever the line pass
came up short. The result becomes an `OcrElement` tree, which ocrmypdf's fpdf2
renderer writes as an invisible text layer.

## Why there are two prompts

The markdown prompt reflows a wrapped paragraph into one run of text under one
box. Build a text layer from that and the whole paragraph becomes a single
selectable run, rendered at paragraph height, so a viewer highlights the entire
block when you drag across one line of it.

Measured on 2026-08-25, on a page holding two wrapped paragraphs: the markdown
prompt returned 4 boxes, `OCR this image.` returned 11 — one per printed line,
including a footer the markdown pass dropped entirely. The line pass is also
faster (2.7s against 12.1s on that page).

It is not a replacement, only a ruler. Its recognized text loses the spaces
between words (`TheCommitteehasreviewed`) and misreads more often
(`acne.example.com` for `acme.example.com`), which is why it is used for
geometry and never for text.

It also stops early sometimes, exactly as the markdown pass does. Over twelve
runs of one page it returned every printed line ten times and stopped short of
a whole paragraph twice, which left that paragraph as one selectable run again.
Tesseract's boxes fill those gaps, so `--no-coverage-guard` costs line geometry
now as well as dropped text.

Tesseract can in fact carry the whole job on a clean page: with
`--no-line-split-pass` the same page still came out one line per printed line,
in half the time, with one break landing a word early. The model pass stays the
default because Tesseract's line detection is the thing that degrades on real
scans and on text it cannot segment — which is the case this tool exists for.
Turn it off when the pages are clean and the time matters.

## Why the coverage guard exists

DeepSeek-OCR-2 stops early when it reaches an isolated element after a large
blank region, silently omitting it. Measured on 2026-08-24: a footer at 86% page
height was dropped on every full-page run at both Q4_K_M and Q8_0, with
`done_reason: stop` and no truncation — yet read perfectly from a tight crop.
For a tool that replaces a text layer, silently losing a line is the worst
failure mode, because nothing about the output looks wrong.

`tests/test_integration.py` pins that exact page.

## Known limits

- `--force-ocr` rasterizes born-digital pages: bigger files, no vector text. Default `--redo-ocr` preserves the original image and only replaces the invisible layer; use `--force-ocr` only when vector text must be flattened.
- Asking for PDF/A output hands the file to Ghostscript, which rebuilds it and strips `/Interpolate`. Image data survives, but the file is no longer byte-comparable to the input. Installing veraPDF lets ocrmypdf try its pikepdf path instead, which only adds an sRGB OutputIntent and XMP metadata — but that path is only used when the result validates, and a scan carrying `/Interpolate true` will not.
- Word boxes within a line are synthesized by character count, so a
  highlight's left and right edges are approximate. Line boxes are measured.
- Where a line break falls inside a paragraph is inferred from the characters
  the line pass read on each line. It lands on the right word almost always,
  and one word off occasionally; the word is still on the right page, one line
  from where it is printed.
- Multi-column pages work because every box the model returns is checked
  against the measured lines, not just the ones that look like paragraphs — the
  model gives one box per column there. Reading order stays column by column.
- The guard only catches text Tesseract can find. Text both engines miss is
  still lost.
- The Q8_0 GGUF is a community conversion, not an official DeepSeek release.
- Roughly 10–20s per page for the two passes, plus a third on pages needing
  repair. `--no-line-split-pass` drops back to one.
- Both prompts stop early on the same pages, so `--no-coverage-guard` now costs
  line geometry as well as dropped text: with neither guard, a wrapped
  paragraph is one selectable run again.

## Tests

```bash
.venv/bin/pytest -m "not slow"   # fast, no Ollama needed
.venv/bin/pytest -m slow         # end-to-end, needs Ollama and the model
```
