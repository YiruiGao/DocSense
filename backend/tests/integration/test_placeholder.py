import pytest


@pytest.mark.integration
def test_no_integration_suite_configured_yet():
    pytest.skip("No integration tests are configured yet")
