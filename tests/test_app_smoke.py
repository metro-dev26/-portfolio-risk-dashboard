from pathlib import Path

from streamlit.testing.v1 import AppTest

# Resolve app.py against the repo root, not the caller's cwd or the test-file
# directory. AppTest.from_file() resolves a *relative* path against the file
# that calls it (tests/), which breaks in CI; an absolute path is portable
# across streamlit versions and working directories.
APP = str(Path(__file__).resolve().parent.parent / "app.py")


def test_app_runs_without_exception():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception


def test_model_validation_section_present():
    at = AppTest.from_file(APP, default_timeout=60).run()
    body = " ".join(str(m.value) for m in at.markdown)
    assert "Model validation" in body


def test_factor_section_present():
    at = AppTest.from_file(APP, default_timeout=60).run()
    body = " ".join(str(m.value) for m in at.markdown)
    assert "Factor" in body and ("exposure" in body.lower() or "tilt" in body.lower())
