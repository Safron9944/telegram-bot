import json
from pathlib import Path

import pytest


@pytest.fixture
def load_fixture_json():
    fixtures = Path(__file__).parent / "fixtures"
    return lambda name: json.loads((fixtures / name).read_text(encoding="utf-8"))
