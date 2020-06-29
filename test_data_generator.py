import pytest

from DataGenerator import *
from Parameters import *
from utils import *
import numpy as np
import scipy
from astropy.io import fits

num_imgs = 10

@pytest.fixture
def params():
    settings_yaml = load_settings_yaml("runs/run.yaml")
    params = Parameters(settings_yaml)
    params.data_type = np.float32 if params.data_type == "np.float32" else np.float32       # must be done here, due to the json, not accepting this kind of if statement in the parameter class.
    return params

@pytest.fixture
def dg(params):
    return DataGenerator(params)

@pytest.fixture
def lenses(dg):
   return dg.get_data_array(dg.params.img_dims,
                            path=dg.params.sources_path,
                            fraction_to_load=num_imgs,
                            are_sources=True,
                            normalize_dat=dg.params.normalize)

def test_receives_and_stores_params(params):
    dg = DataGenerator(params)
    assert dg.params.__dict__ == params.__dict__
    assert np.array_equal(dg.PSF_r, dg.compute_PSF_r())

def test_computes_correct_psf_r(dg):
    PSF_r = dg.compute_PSF_r()
    assert np.amin(PSF_r) == 0.0
    assert np.amax(PSF_r) == 0.04781075159040512
    assert np.shape(PSF_r) == (101, 101)

def test_get_data_array_returns_lenses(dg):
    data_array = dg.get_data_array(dg.params.img_dims,
                                   path=dg.params.lenses_path,
                                   fraction_to_load=num_imgs,
                                   are_sources=False,
                                   normalize_dat=dg.params.normalize)
    assert np.shape(data_array) == (num_imgs, 101, 101, 1)
    max_val, min_val = (np.amax(data_array), np.amin(data_array))
    assert max_val == 1.0
    assert min_val == 0.0

    
def test_data_array_returns_negatives(dg):
    data_array = dg.get_data_array(dg.params.img_dims,
                                   path=dg.params.negatives_path,
                                   fraction_to_load=num_imgs,
                                   are_sources=False,
                                   normalize_dat=dg.params.normalize)
    assert np.shape(data_array) == (num_imgs, 101, 101, 1)
    max_val, min_val = (np.amax(data_array), np.amin(data_array))
    assert max_val == 1.0
    assert min_val == 0.0


def test_get_data_array_returns_sources(dg):
    data_array = dg.get_data_array(dg.params.img_dims,
                                   path=dg.params.sources_path,
                                   fraction_to_load=num_imgs,
                                   are_sources=True,
                                   normalize_dat=dg.params.normalize)
    assert np.shape(data_array) == (num_imgs, 101, 101, 1)
    max_val, min_val = (np.amax(data_array), np.amin(data_array))
    assert max_val == 1.0
    assert min_val == 0.0

def test_split_train_test_data(dg, lenses):
    splits = dg.split_train_test_data(lenses,
                                      0.0,
                                      test_fraction=dg.params.test_fraction)
    X_train_lenses, X_test_lenses, y_train_lenses, y_test_lenses = splits
    assert np.shape(X_train_lenses)[0] == int(num_imgs * (1 - dg.params.test_fraction))
    assert np.shape(X_test_lenses)[0] == int(num_imgs * dg.params.test_fraction)

    assert np.shape(y_train_lenses)[0] == int(num_imgs * (1 - dg.params.test_fraction))
    assert np.shape(y_test_lenses)[0] == int(num_imgs * dg.params.test_fraction)

def test_it_loads_a_chunk_of_size_N(dg, lenses):
    N = 2
    chunk = dg.load_chunk(N, lenses, lenses, lenses, np.float32, (0.02, 0.3))
    assert np.shape(chunk)[0] == N
    max_val, min_val = (np.amax(chunk), np.amin(chunk))
    assert max_val == 1.0
    assert min_val == 0.0

def test_it_merges_lenses_and_sources_correctly(dg):
    raise "not implemented"

def test_merge_lense_and_source(dg):
    # Merge a single lens and source together into a mock lens.

    lens = dg.normalize_img(np.expand_dims(fits.getdata("data/training/lenses/KIDS_216.0_1.5_ra_215.885864819_dec_1.37332389553__OCAM_r_.fits"), axis=2).astype(np.float32))
    source = fits.getdata("data/training/sources/1/1.fits").astype(data_type)
    source = dg.normalize_img(np.expand_dims(scipy.signal.fftconvolve(source, PSF_r, mode="same"), axis=2))
    mock_lens_alpha_scaling = (0.1, 0.1)
    
    mock_lens = dg.merge_lens_and_source(lens, source, mock_lens_alpha_scaling)
