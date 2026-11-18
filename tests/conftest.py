import pytest

from contact_learning.environments.gripper_lift import EnvConfig, GripperLiftEnv


@pytest.fixture(scope="session")
def env():
    return GripperLiftEnv(EnvConfig())


@pytest.fixture(scope="session")
def h1cfg():
    return {"T": 60, "K": 4, "H": 10, "magnitudes": [0.0, 0.1, 0.3], "strata": ["near_onset", "uniform"]}
