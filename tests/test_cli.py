# tests/test_cli.py
import pytest

from deepseek_ocr_pdf.cli import build_parser, configure_engine, resolve_policy
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


def _parse(argv):
    args, passthrough = build_parser().parse_known_args(argv)
    resolve_policy(args, passthrough)
    return args, passthrough


def test_input_and_output_are_required():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_force_ocr_is_the_default_policy():
    args, _ = _parse(["in.pdf", "out.pdf"])
    assert args.force_ocr is True


def test_explicit_redo_ocr_disables_force_ocr():
    args, passthrough = _parse(["in.pdf", "out.pdf", "--redo-ocr"])
    assert args.force_ocr is False
    assert "--redo-ocr" in passthrough


def test_skip_text_disables_force_ocr():
    args, _ = _parse(["in.pdf", "out.pdf", "--skip-text"])
    assert args.force_ocr is False


def test_explicit_force_ocr_is_not_added_twice():
    args, passthrough = _parse(["in.pdf", "out.pdf", "--force-ocr"])
    assert args.force_ocr is False
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


def test_max_dim_defaults_to_no_resizing():
    args, _ = _parse(["in.pdf", "out.pdf"])
    configure_engine(args)
    assert DeepSeekOcrEngine.max_dim is None


def test_max_dim_is_applied_when_given():
    args, _ = _parse(["in.pdf", "out.pdf", "--max-dim", "1200"])
    configure_engine(args)
    assert DeepSeekOcrEngine.max_dim == 1200
