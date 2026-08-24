# src/deepseek_ocr_pdf/cli.py
"""Command-line entry point wrapping ocrmypdf."""

from __future__ import annotations

import argparse
import logging
import sys

from deepseek_ocr_pdf.ollama_client import (
    DEFAULT_HOST,
    DEFAULT_MODEL,
    DEFAULT_TIMEOUT,
)
from deepseek_ocr_pdf.plugin import DeepSeekOcrEngine

#: ocrmypdf options that decide the fate of existing text. They are mutually
#: exclusive, so naming any of them must suppress our default --force-ocr.
POLICY_FLAGS = ("--redo-ocr", "--skip-text", "--force-ocr")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deepseek-ocr-pdf",
        description=(
            "Replace a PDF's text layer using DeepSeek-OCR-2 via Ollama. "
            "Unrecognized options are passed through to ocrmypdf."
        ),
    )
    parser.add_argument("input_pdf")
    parser.add_argument("output_pdf")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model tag")
    parser.add_argument(
        "--ollama-host", default=DEFAULT_HOST, help="Ollama base URL"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help="Seconds allowed for one page",
    )
    parser.add_argument(
        "--max-dim",
        type=int,
        default=None,
        help="Shrink pages to this longest edge before OCR (default: no resize)",
    )
    parser.add_argument(
        "--no-coverage-guard",
        action="store_true",
        help="Skip the Tesseract pass that finds text the model dropped",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.set_defaults(force_ocr=True)
    return parser


def resolve_policy(args: argparse.Namespace, passthrough: list[str]) -> None:
    """Drop the default --force-ocr when the user named a policy themselves."""
    if any(flag in passthrough for flag in POLICY_FLAGS):
        args.force_ocr = False


def configure_engine(args: argparse.Namespace) -> None:
    """Apply CLI settings to the engine class ocrmypdf will instantiate.

    ocrmypdf constructs the engine itself through the plugin hook, so there is
    no instance to configure. Settings live on the class.
    """
    DeepSeekOcrEngine.model = args.model
    DeepSeekOcrEngine.host = args.ollama_host
    DeepSeekOcrEngine.timeout = args.timeout
    DeepSeekOcrEngine.max_dim = args.max_dim
    DeepSeekOcrEngine.coverage_guard = not args.no_coverage_guard


def _as_kwargs(flags: list[str]) -> dict:
    """Translate passthrough flags into ocrmypdf.ocr keyword arguments.

    ocrmypdf.ocr takes keywords rather than an argv list, so ``--sidecar
    out.txt`` becomes ``sidecar="out.txt"`` and a bare ``--deskew`` becomes
    ``deskew=True``.
    """
    kwargs: dict = {}
    index = 0
    while index < len(flags):
        flag = flags[index]
        if not flag.startswith("--"):
            index += 1
            continue
        name = flag[2:].replace("-", "_")
        if index + 1 < len(flags) and not flags[index + 1].startswith("--"):
            kwargs[name] = flags[index + 1]
            index += 2
        else:
            kwargs[name] = True
            index += 1
    return kwargs


def main(argv: list[str] | None = None) -> int:
    import ocrmypdf

    args, passthrough = build_parser().parse_known_args(argv)
    resolve_policy(args, passthrough)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    configure_engine(args)

    flags = list(passthrough)
    if args.force_ocr:
        flags.append("--force-ocr")

    try:
        return ocrmypdf.ocr(
            args.input_pdf,
            args.output_pdf,
            plugins=["deepseek_ocr_pdf.plugin"],
            **_as_kwargs(flags),
        )
    except Exception as exc:  # noqa: BLE001 - surface one clean line, not a trace
        print(f"error: {exc}", file=sys.stderr)
        return 1
