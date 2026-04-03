import pytest

import app.api.evaluation as evaluation


@pytest.mark.component
def test_evaluation_api_router_is_available():
    assert evaluation.router.tags == ["evaluation"]
