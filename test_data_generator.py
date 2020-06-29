import pytest

from DataGenerator import *
from Parameters import *
from utils import *
import numpy as np

@pytest.fixture
def params():
    settings_yaml = load_settings_yaml("runs/run.yaml")
    params = Parameters(settings_yaml)
    params.data_type = np.float32 if params.data_type == "np.float32" else np.float32       # must be done here, due to the json, not accepting this kind of if statement in the parameter class.
    return params

@pytest.fixture
def dg(params):
    return DataGenerator(params)


def test_receives_and_stores_params(params):
    dg = DataGenerator(params)
    assert dg.params.__dict__ == params.__dict__
    assert dg.PSF_r == dg.compute_PSF_r()

def test_computes_correct_psf_r(dg):
    