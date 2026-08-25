# tests/test_cli.py
import pytest

from deepseek_ocr_pdf.cli import (
    build_parser,
    configure_engine,
    resolve_output_type,
    resolve_policy,
)
from deepseek_ocr_pdf.ollama_client import DEFAULT_HOST, DEFAULT_MODEL
from deepseek_ocr_pdf.plugin import DeepSeekOcrEngine


@pytest.fixture(autouse=True)
def reset_engine():
    """Engine settings live on the class, so every test must restore them."""
    yield
    DeepSeekOcrEngine.model = DEFAULT_MODEL
    DeepSeekOcrEngine.host = DEFAULT_HOST
    DeepSeekOcrEngine.max_dim = None
    DeepSeekOcrEngine.coverage_guard = True
    DeepSeekOcrEngine.line_split_pass = True


def _parse(argv):
    args, passthrough = build_parser().parse_known_args(argv)
    resolve_policy(args, passthrough)
    resolve_output_type(args, passthrough)
    return args, passthrough


def test_input_and_output_are_required():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_redo_ocr_is_the_default_policy():
    args, _ = _parse(["in.pdf", "out.pdf"])
    assert args.redo_ocr is True


def test_explicit_redo_ocr_disables_default_redo():
    args, passthrough = _parse(["in.pdf", "out.pdf", "--redo-ocr"])
    assert args.redo_ocr is False
    assert "--redo-ocr" in passthrough


def test_skip_text_disables_redo_ocr():
    args, _ = _parse(["in.pdf", "out.pdf", "--skip-text"])
    assert args.redo_ocr is False


def test_explicit_force_ocr_is_not_added_twice():
    args, passthrough = _parse(["in.pdf", "out.pdf", "--force-ocr"])
    assert args.redo_ocr is False
    assert "--force-ocr" in passthrough


def test_unknown_arguments_pass_through_to_ocrmypdf():
    _, passthrough = _parse(
        ["in.pdf", "out.pdf", "--sidecar", "out.txt", "--jobs", "4"]
    )
    assert passthrough == ["--sidecar", "out.txt", "--jobs", "4"]


def test_configure_engine_applies_model_and_host():
    args, _ = _parse(
        [
            "in.pdf",
            "out.pdf",
            "--model",
            "custom:tag",
            "--ollama-host",
            "http://box:1234",
        ]
    )
    configure_engine(args)
    assert DeepSeekOcrEngine.model == "custom:tag"
    assert DeepSeekOcrEngine.host == "http://box:1234"


def test_configure_engine_can_disable_the_coverage_guard():
    args, _ = _parse(["in.pdf", "out.pdf", "--no-coverage-guard"])
    configure_engine(args)
    assert DeepSeekOcrEngine.coverage_guard is False


def test_line_split_pass_is_on_by_default():
    args, _ = _parse(["in.pdf", "out.pdf"])
    configure_engine(args)
    assert DeepSeekOcrEngine.line_split_pass is True


def test_configure_engine_can_disable_the_line_split_pass():
    args, _ = _parse(["in.pdf", "out.pdf", "--no-line-split-pass"])
    configure_engine(args)
    assert DeepSeekOcrEngine.line_split_pass is False


def test_max_dim_defaults_to_no_resizing():
    args, _ = _parse(["in.pdf", "out.pdf"])
    configure_engine(args)
    assert DeepSeekOcrEngine.max_dim is None


def test_max_dim_is_applied_when_given():
    args, _ = _parse(["in.pdf", "out.pdf", "--max-dim", "1200"])
    configure_engine(args)
    assert DeepSeekOcrEngine.max_dim == 1200


def test_plain_pdf_is_the_default_output_type():
    """Replacing a text layer should not rewrite the whole document.

    ocrmypdf defaults to --output-type auto, which falls back to a Ghostscript
    PDF/A conversion when veraPDF is absent. Measured with ocrmypdf's own
    settings, that pass is kinder than it first appears: JPEG streams survive
    byte for byte and a Flate bilevel image re-encoded to CCITT G4 came back
    pixel-identical. What it does change is /Interpolate, which PDF/A forbids
    and Ghostscript therefore strips from every image, and the document
    structure around them. Nobody asked for archival output, so the default is
    the pass that leaves the file alone.
    """
    args, _ = _parse(["in.pdf", "out.pdf"])
    assert args.plain_pdf is True


def test_explicit_output_type_is_respected():
    args, passthrough = _parse(["in.pdf", "out.pdf", "--output-type", "pdfa"])
    assert args.plain_pdf is False
    assert passthrough == ["--output-type", "pdfa"]


def test_explicit_plain_output_type_is_not_added_twice():
    args, passthrough = _parse(["in.pdf", "out.pdf", "--output-type", "pdf"])
    assert args.plain_pdf is False
    assert passthrough.count("--output-type") == 1
