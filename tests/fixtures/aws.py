import pytest


@pytest.fixture
def fx_boto3_session(mocker):
    yield mocker.patch("boto3.Session").return_value
