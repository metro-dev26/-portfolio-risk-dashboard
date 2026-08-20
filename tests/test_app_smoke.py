from streamlit.testing.v1 import AppTest


def test_app_runs_without_exception():
    at = AppTest.from_file("app.py", default_timeout=60).run()
    assert not at.exception


def test_model_validation_section_present():
    at = AppTest.from_file("app.py", default_timeout=60).run()
    body = " ".join(str(m.value) for m in at.markdown)
    assert "Model validation" in body


def test_factor_section_present():
    at = AppTest.from_file("app.py", default_timeout=60).run()
    body = " ".join(str(m.value) for m in at.markdown)
    assert "Factor" in body and ("exposure" in body.lower() or "tilt" in body.lower())
