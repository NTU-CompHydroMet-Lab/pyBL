
from pybl.timeseries import IndexedSnapshot
import numpy as np
import numpy.typing as npt

def initialize_isnap(time, intensity):
    isnap = IndexedSnapshot(time, intensity)
    return isnap

def check_mean(isnap: IndexedSnapshot, time: npt.NDArray, intensity: npt.NDArray):
    assert np.isclose(isnap.mean(), np.nanmean(intensity), rtol=1e-10, equal_nan=True)
    assert np.isclose(isnap.mean(), np.nanmean(isnap.unpack()[1]), rtol=1e-10, equal_nan=True)
    
def check_sse(isnap: IndexedSnapshot, time: npt.NDArray, intensity: npt.NDArray):
    assert np.isclose(isnap.sum_squared_error(), np.nansum((intensity - np.nanmean(intensity))**2) * (time[1] - time[0]), rtol=1e-10, equal_nan=True)
    assert np.isclose(isnap.sum_squared_error(), np.nansum((isnap.unpack()[1] - np.nanmean(isnap.unpack()[1]))**2), rtol=1e-10, equal_nan=True)
    
def check_variance(isnap: IndexedSnapshot, time: npt.NDArray, intensity: npt.NDArray):
    
    # In isnap. The denominator of var is time[-1] - time[0]. But in real data, the denominator of var is number of data points.
    #assert np.isclose(isnap.variance(biased=False), np.nanvar(intensity, ddof=0), rtol=1e-10, equal_nan=True)
    
    assert np.isclose(isnap.variance(biased=False), np.nanvar(isnap.unpack()[1], ddof=1), rtol=1e-10, equal_nan=True)
    assert np.isclose(isnap.variance(biased=True), np.nanvar(intensity), rtol=1e-10, equal_nan=True)
    assert np.isclose(isnap.variance(biased=True), np.nanvar(isnap.unpack()[1]), rtol=1e-10, equal_nan=True)

def check_skew(isnap: IndexedSnapshot, time: npt.NDArray, intensity: npt.NDArray):
    np_std = np.nanstd(intensity, ddof=0)
    if np.isnan(np_std) or np.isclose(np_std, 0, atol=1e-10):
        np_skew = np.nan
    else:
        np_skew = np.nanmean((intensity - np.nanmean(intensity))**3) / np.nanstd(intensity, ddof=0)**3    
    
    unpack_intensity = isnap.unpack()[1]
    np_unpack_std = np.nanstd(unpack_intensity, ddof=0)
    if np.isnan(np_unpack_std) or np.isclose(np_unpack_std, 0, atol=1e-10):
        np_unpack_skew = np.nan
    else:
        np_unpack_skew = np.nanmean((unpack_intensity - np.nanmean(unpack_intensity))**3) / np.nanstd(unpack_intensity, ddof=0)**3

    assert np.isclose(isnap.skewness(), np_skew, rtol=1e-10, equal_nan=True)
    assert np.isclose(isnap.skewness(), np_unpack_skew, rtol=1e-10, equal_nan=True)
    
def check_acf(isnap: IndexedSnapshot, time: npt.NDArray, intensity: npt.NDArray):
    mean = np.nanmean(intensity)
    x_i = intensity[:-1]
    y_i = intensity[1:]
    xi_minus_mean = x_i - mean
    yi_minus_mean = y_i - mean
    numerator = np.nansum(xi_minus_mean * yi_minus_mean)
    denominator = np.nansum((intensity - mean) ** 2)
    if np.isclose(denominator, 0, atol=1e-10):
        autocorr_coef = np.nan
    else:
        autocorr_coef = numerator / denominator
    
    lag = time[1] - time[0]
    assert np.isclose(isnap.autocorr_coef(lag), autocorr_coef, rtol=1e-10, equal_nan=True)
    
def check_rescale_cycle(isnap: IndexedSnapshot, time: npt.NDArray, intensity: npt.NDArray):
    isnap_rescale = isnap.rescale(1/6)
    isnap_rescale = isnap_rescale.rescale(3)
    isnap_rescale = isnap_rescale.rescale(2)

    #assert np.isclose(isnap.intensity.size, isnap_rescale.intensity.size, rtol=1e-10, equal_nan=True)
    assert np.isclose(isnap.mean(), isnap_rescale.mean(), rtol=1e-10, equal_nan=True)
    assert np.isclose(isnap.sum_squared_error(), isnap_rescale.sum_squared_error(), rtol=1e-10, equal_nan=True)
    assert np.isclose(isnap.variance(biased=False), isnap_rescale.variance(biased=False), rtol=1e-10, equal_nan=True)
    assert np.isclose(isnap.variance(biased=True), isnap_rescale.variance(biased=True), rtol=1e-10, equal_nan=True)
    assert np.isclose(isnap.skewness(), isnap_rescale.skewness(), rtol=1e-10, equal_nan=True)
    assert np.isclose(isnap.autocorr_coef(1), isnap_rescale.autocorr_coef(1), rtol=1e-10, equal_nan=True)

def test_short_input_stats(fuzzy_short_data):
    for (time, intensity) in fuzzy_short_data:
        isnap = initialize_isnap(time, intensity)
        check_mean(isnap, time, intensity)
        check_sse(isnap, time, intensity)
        check_variance(isnap, time, intensity)
        check_skew(isnap, time, intensity)
        check_acf(isnap, time, intensity)
        check_rescale_cycle(isnap, time, intensity)
        
def test_long_input_stats(fuzzy_long_data):
    for (time, intensity) in fuzzy_long_data:
        isnap = initialize_isnap(time, intensity)
        check_mean(isnap, time, intensity)
        check_sse(isnap, time, intensity)
        check_variance(isnap, time, intensity)
        check_skew(isnap, time, intensity)
        check_acf(isnap, time, intensity)
        check_rescale_cycle(isnap, time, intensity)