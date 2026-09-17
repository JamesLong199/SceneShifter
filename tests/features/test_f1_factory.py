import pytest

from sceneshifter.router import create_router
from tests.conftest import canonical_config


@pytest.mark.feature
def test_disabled_returns_none():
    cfg = canonical_config(enabled=False)
    assert create_router(cfg, validate=True) is None


@pytest.mark.feature
def test_enabled_returns_router():
    router = create_router(canonical_config(), backend="multitalk", validate=True)
    assert router is not None
    assert router.enabled is True
    assert router.num_segments == 2
