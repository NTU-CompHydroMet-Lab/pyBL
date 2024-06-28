from pybl.timeseries.indexsnapshot import (
    _isnapshot_sum_and_duration,
    _isnapshot_mean,
    _isnapshot_sum_squared_error,
    _isnapshot_variance,
    _isnapshot_standard_deviation,
    _isnapshot_coef_var,
    _isnapshot_skew,
    _isnapshot_pDry,
    _isnapshot_acf,
    _isnapshot_rescale,
)
from pybl.timeseries import IndexedSnapshot
import numpy as np
import pytest


def test_sum_core_vs_isnap(elmdon_data_np):
    """
    Test the sum of the intensity snapshot using the core function and the IndexedSnapshot class.
    """

    elm_time, elm_intensity = elmdon_data_np

    # The intensity is in the unit of mm/hr. So we divide the time by 3600 to convert it to hours.
    elm_isnap = IndexedSnapshot(elm_time / 3600, elm_intensity)
    elm_isnap_sum = elm_isnap.sum()

    elm_raw_time = elm_time / 3600
    elm_raw_sum, _ = _isnapshot_sum_and_duration(
        np.append(elm_raw_time, elm_raw_time[-1] + 1), np.append(elm_intensity, np.nan)
    )

    assert np.isclose(elm_isnap_sum, elm_raw_sum, rtol=1e-10)


def test_sum_core_vs_np_pd(elmdon_data_np, elmdon_data_pd):
    """
    Test the sum of the intensity snapshot using the core function and the numpy and pandas functions.
    """

    elm_time, elm_intensity = elmdon_data_np

    elm_raw_time = elm_time / 3600
    elm_raw_sum, _ = _isnapshot_sum_and_duration(
        np.append(elm_raw_time, elm_raw_time[-1] + 1), np.append(elm_intensity, np.nan)
    )

    assert np.isclose(elm_raw_sum, elm_intensity.sum(), rtol=1e-10)
    assert np.isclose(elm_raw_sum, elmdon_data_pd.sum(), rtol=1e-10)


def test_sum_isnap_with_nan_core_vs_isnap(bochum_data_np, bochum_data_pd):
    """
    Test the sum of the intensity snapshot with NaN values using the core function and the IndexedSnapshot class.
    """

    boc_time, boc_intensity = bochum_data_np

    boc_isnap = IndexedSnapshot(boc_time / 300, boc_intensity)
    boc_isnap_sum = boc_isnap.sum()

    boc_raw_time = boc_time / 300
    boc_raw_sum, _ = _isnapshot_sum_and_duration(
        np.append(boc_raw_time, boc_raw_time[-1] + 1), np.append(boc_intensity, np.nan)
    )

    assert np.isclose(boc_isnap_sum, boc_raw_sum, rtol=1e-10)


def test_sum_isnap_with_nan_core_vs_np_pd(bochum_data_np, bochum_data_pd):
    """
    Test the sum of the intensity snapshot with NaN values using the core function and the numpy and pandas functions.
    """

    boc_time, boc_intensity = bochum_data_np

    boc_raw_time = boc_time / 300
    boc_raw_sum, _ = _isnapshot_sum_and_duration(
        np.append(boc_raw_time, boc_raw_time[-1] + 1), np.append(boc_intensity, np.nan)
    )

    assert np.isclose(boc_raw_sum, np.nansum(boc_intensity), rtol=1e-10)
    assert np.isclose(boc_raw_sum, bochum_data_pd.sum(), rtol=1e-10)


def test_mean_core_vs_isnap(elmdon_data_np):
    elm_time, elm_intensity = elmdon_data_np

    # The intensity is in the unit of mm/hr. So we divide the time by 3600 to convert it to hours.
    elm_isnap = IndexedSnapshot(elm_time / 3600, elm_intensity)
    elm_isnap_mean = elm_isnap.mean()

    elm_raw_time = elm_time / 3600
    elm_raw_mean = _isnapshot_mean(
        np.append(elm_raw_time, elm_raw_time[-1] + 1), np.append(elm_intensity, np.nan)
    )

    assert np.isclose(elm_isnap_mean, elm_raw_mean, rtol=1e-10)


def test_mean_core_vs_np_pd(elmdon_data_np, elmdon_data_pd):
    elm_time, elm_intensity = elmdon_data_np

    elm_raw_time = elm_time / 3600
    elm_raw_mean = _isnapshot_mean(
        np.append(elm_raw_time, elm_raw_time[-1] + 1), np.append(elm_intensity, np.nan)
    )

    assert np.isclose(elm_raw_mean, elm_intensity.mean(), rtol=1e-10)
    assert np.isclose(elm_raw_mean, elmdon_data_pd.mean(), rtol=1e-10)


def test_mean_with_nan_core_vs_isnap(bochum_data_np):
    boc_time, boc_intensity = bochum_data_np

    boc_isnap = IndexedSnapshot(boc_time / 300, boc_intensity)
    boc_isnap_mean = boc_isnap.mean()

    boc_raw_time = boc_time / 300
    boc_raw_mean = _isnapshot_mean(
        np.append(boc_raw_time, boc_raw_time[-1] + 1), np.append(boc_intensity, np.nan)
    )

    assert np.isclose(boc_isnap_mean, boc_raw_mean, rtol=1e-10)


def test_mean_with_nan_core_vs_np_pd(bochum_data_np, bochum_data_pd):
    boc_time, boc_intensity = bochum_data_np

    boc_raw_time = boc_time / 300
    boc_depth, boc_duration = _isnapshot_sum_and_duration(
        np.append(boc_raw_time, boc_raw_time[-1] + 1), np.append(boc_intensity, np.nan)
    )
    print(boc_depth, boc_duration)
    boc_raw_mean = _isnapshot_mean(
        np.append(boc_raw_time, boc_raw_time[-1] + 1), np.append(boc_intensity, np.nan)
    )
    print(boc_raw_mean)

    assert np.isclose(boc_raw_mean, np.nanmean(boc_intensity), rtol=1e-10)
    assert np.isclose(boc_raw_mean, bochum_data_pd.mean(), rtol=1e-10)


def test_square_sum_error_core_vs_isnap(elmdon_data_np):
    elm_time, elm_intensity = elmdon_data_np

    # The intensity is in the unit of mm/hr. So we divide the time by 3600 to convert it to hours.
    elm_isnap = IndexedSnapshot(elm_time / 3600, elm_intensity)
    elm_isnap_sum_squared_error = elm_isnap.sum_squared_error()

    elm_raw_time = elm_time / 3600
    elm_raw_sum_squared_error = _isnapshot_sum_squared_error(
        np.append(elm_raw_time, elm_raw_time[-1] + 1), np.append(elm_intensity, np.nan)
    )

    assert np.isclose(
        elm_isnap_sum_squared_error, elm_raw_sum_squared_error, rtol=1e-10
    )


def test_square_sum_error_core_vs_np_pd(elmdon_data_np, elmdon_data_pd):
    elm_time, elm_intensity = elmdon_data_np

    elm_raw_time = elm_time / 3600
    elm_raw_sum_squared_error = _isnapshot_sum_squared_error(
        np.append(elm_raw_time, elm_raw_time[-1] + 1), np.append(elm_intensity, np.nan)
    )

    assert np.isclose(
        elm_raw_sum_squared_error,
        np.sum(np.square(elm_intensity - elm_intensity.mean())),
        rtol=1e-10,
    )
    assert np.isclose(
        elm_raw_sum_squared_error,
        np.sum(np.square(elmdon_data_pd - elmdon_data_pd.mean())),
        rtol=1e-10,
    )


def test_square_sum_error_with_nan_core_vs_isnap(bochum_data_np):
    boc_time, boc_intensity = bochum_data_np

    boc_isnap = IndexedSnapshot(boc_time / 300, boc_intensity)
    boc_isnap_sum_squared_error = boc_isnap.sum_squared_error()

    boc_raw_time = boc_time / 300
    boc_raw_sum_squared_error = _isnapshot_sum_squared_error(
        np.append(boc_raw_time, boc_raw_time[-1] + 1), np.append(boc_intensity, np.nan)
    )

    assert np.isclose(
        boc_isnap_sum_squared_error, boc_raw_sum_squared_error, rtol=1e-10
    )


def test_square_sum_error_with_nan_core_vs_np_pd(bochum_data_np, bochum_data_pd):
    boc_time, boc_intensity = bochum_data_np

    boc_raw_time = boc_time / 300
    boc_raw_sum_squared_error = _isnapshot_sum_squared_error(
        np.append(boc_raw_time, boc_raw_time[-1] + 1), np.append(boc_intensity, np.nan)
    )

    assert np.isclose(
        boc_raw_sum_squared_error,
        np.nansum(np.square(boc_intensity - np.nanmean(boc_intensity))),
        rtol=1e-10,
    )
    assert np.isclose(
        boc_raw_sum_squared_error,
        np.nansum(np.square(bochum_data_pd - bochum_data_pd.mean())),
        rtol=1e-10,
    )


def test_variance_core_vs_isnap(elmdon_data_np):
    elm_time, elm_intensity = elmdon_data_np

    # The intensity is in the unit of mm/hr. So we divide the time by 3600 to convert it to hours.
    elm_isnap = IndexedSnapshot(elm_time / 3600, elm_intensity)
    elm_isnap_coef_var = elm_isnap.variance()

    elm_raw_time = elm_time / 3600
    elm_raw_coef_var = _isnapshot_variance(
        np.append(elm_raw_time, elm_raw_time[-1] + 1), np.append(elm_intensity, np.nan), biased=False
    )

    assert np.isclose(elm_isnap_coef_var, elm_raw_coef_var, rtol=1e-10)


def test_variance_core_vs_np_pd(elmdon_data_np, elmdon_data_pd):
    elm_time, elm_intensity = elmdon_data_np

    elm_raw_time = elm_time / 3600
    elm_raw_coef_var = _isnapshot_variance(
        np.append(elm_raw_time, elm_raw_time[-1] + 1),
        np.append(elm_intensity, np.nan),
        biased=False,
    )

    assert np.isclose(
        elm_raw_coef_var,
        np.var(elm_intensity, ddof=1),
        rtol=1e-10,
    )
    assert np.isclose(
        elm_raw_coef_var,
        np.var(elmdon_data_pd, ddof=1),
        rtol=1e-10,
    )
    
def test_variance_with_nan_core_vs_isnap(bochum_data_np):
    boc_time, boc_intensity = bochum_data_np

    boc_isnap = IndexedSnapshot(boc_time / 300, boc_intensity)
    boc_isnap_coef_var = boc_isnap.variance()

    boc_raw_time = boc_time / 300
    boc_raw_coef_var = _isnapshot_variance(
        np.append(boc_raw_time, boc_raw_time[-1] + 1), np.append(boc_intensity, np.nan), biased=False
    )

    assert np.isclose(boc_isnap_coef_var, boc_raw_coef_var, rtol=1e-10)
    
def test_variance_with_nan_core_vs_np_pd(bochum_data_np, bochum_data_pd):
    boc_time, boc_intensity = bochum_data_np

    boc_raw_time = boc_time / 300
    boc_raw_coef_var = _isnapshot_variance(
        np.append(boc_raw_time, boc_raw_time[-1] + 1),
        np.append(boc_intensity, np.nan),
        biased=False,
    )

    assert np.isclose(
        boc_raw_coef_var,
        np.nanvar(boc_intensity, ddof=1),
        rtol=1e-10,
    )
    assert np.isclose(
        boc_raw_coef_var,
        np.nanvar(bochum_data_pd, ddof=1),
        rtol=1e-10,
    )
    
def test_coef_var_core_vs_isnap(elmdon_data_np):
    elm_time, elm_intensity = elmdon_data_np

    # The intensity is in the unit of mm/hr. So we divide the time by 3600 to convert it to hours.
    elm_isnap = IndexedSnapshot(elm_time / 3600, elm_intensity)
    elm_isnap_coef_var = elm_isnap.coef_variation()

    elm_raw_time = elm_time / 3600
    elm_raw_coef_var = _isnapshot_coef_var(
        np.append(elm_raw_time, elm_raw_time[-1] + 1), np.append(elm_intensity, np.nan), biased=False
    )

    assert np.isclose(elm_isnap_coef_var, elm_raw_coef_var, rtol=1e-10)
    
def test_coef_var_core_vs_np_pd(elmdon_data_np, elmdon_data_pd):
    elm_time, elm_intensity = elmdon_data_np

    elm_raw_time = elm_time / 3600
    elm_raw_coef_var = _isnapshot_coef_var(
        np.append(elm_raw_time, elm_raw_time[-1] + 1),
        np.append(elm_intensity, np.nan),
        biased=False,
    )

    assert np.isclose(
        elm_raw_coef_var,
        np.std(elm_intensity, ddof=1) / np.mean(elm_intensity),
        rtol=1e-10,
    )
    assert np.isclose(
        elm_raw_coef_var,
        np.std(elmdon_data_pd, ddof=1) / np.mean(elmdon_data_pd),
        rtol=1e-10,
    )
    
def test_coef_var_with_nan_core_vs_isnap(bochum_data_np):
    boc_time, boc_intensity = bochum_data_np

    boc_isnap = IndexedSnapshot(boc_time / 300, boc_intensity)
    boc_isnap_coef_var = boc_isnap.coef_variation()

    boc_raw_time = boc_time / 300
    boc_raw_coef_var = _isnapshot_coef_var(
        np.append(boc_raw_time, boc_raw_time[-1] + 1), np.append(boc_intensity, np.nan), biased=False
    )

    assert np.isclose(boc_isnap_coef_var, boc_raw_coef_var, rtol=1e-10)
    
def test_coef_var_with_nan_core_vs_np_pd(bochum_data_np, bochum_data_pd):
    boc_time, boc_intensity = bochum_data_np

    boc_raw_time = boc_time / 300
    boc_raw_coef_var = _isnapshot_coef_var(
        np.append(boc_raw_time, boc_raw_time[-1] + 1),
        np.append(boc_intensity, np.nan),
        biased=False,
    )

    assert np.isclose(
        boc_raw_coef_var,
        np.nanstd(boc_intensity, ddof=1) / np.nanmean(boc_intensity),
        rtol=1e-10,
    )
    assert np.isclose(
        boc_raw_coef_var,
        np.nanstd(bochum_data_pd, ddof=1) / np.nanmean(bochum_data_pd),
        rtol=1e-10,
    )
    
def test_skewness_core_vs_isnap(elmdon_data_np):
    elm_time, elm_intensity = elmdon_data_np

    # The intensity is in the unit of mm/hr. So we divide the time by 3600 to convert it to hours.
    elm_isnap = IndexedSnapshot(elm_time / 3600, elm_intensity)
    elm_isnap_skew = elm_isnap.skewness()

    elm_raw_time = elm_time / 3600
    elm_raw_skew = _isnapshot_skew(
        np.append(elm_raw_time, elm_raw_time[-1] + 1), np.append(elm_intensity, np.nan), biased=True
    )

    assert np.isclose(elm_isnap_skew, elm_raw_skew, rtol=1e-10)
    
def test_skewness_core_vs_np_pd(elmdon_data_np, elmdon_data_pd):
    elm_time, elm_intensity = elmdon_data_np

    elm_raw_time = elm_time / 3600
    elm_raw_skew_biased = _isnapshot_skew(
        np.append(elm_raw_time, elm_raw_time[-1] + 1),
        np.append(elm_intensity, np.nan),
        biased=True
    )
    elm_raw_skew_unbiased = _isnapshot_skew(
        np.append(elm_raw_time, elm_raw_time[-1] + 1),
        np.append(elm_intensity, np.nan),
        biased=False
    )

    assert np.isclose(
        elm_raw_skew_biased,
        np.mean((elm_intensity - np.mean(elm_intensity))**3) / np.std(elm_intensity, ddof=0)**3,
        rtol=1e-10,
    )
    
    # Pandas calculate skewness using biased standard deviation and Fisher-Pearson coefficient
    assert np.isclose(
        elm_raw_skew_unbiased,
        elmdon_data_pd.skew(),
        rtol=1e-10,
    )
    
def test_skewness_with_nan_core_vs_isnap(bochum_data_np):
    boc_time, boc_intensity = bochum_data_np

    boc_isnap = IndexedSnapshot(boc_time / 300, boc_intensity)
    boc_isnap_skew = boc_isnap.skewness()

    boc_raw_time = boc_time / 300
    boc_raw_skew = _isnapshot_skew(
        np.append(boc_raw_time, boc_raw_time[-1] + 1), np.append(boc_intensity, np.nan), biased=True
    )

    assert np.isclose(boc_isnap_skew, boc_raw_skew, rtol=1e-10)
    
def test_skewness_with_nan_core_vs_np_pd(bochum_data_np, bochum_data_pd):
    boc_time, boc_intensity = bochum_data_np

    boc_raw_time = boc_time / 300
    boc_raw_skew_biased = _isnapshot_skew(
        np.append(boc_raw_time, boc_raw_time[-1] + 1),
        np.append(boc_intensity, np.nan),
        biased=True
    )
    boc_raw_skew_unbiased = _isnapshot_skew(
        np.append(boc_raw_time, boc_raw_time[-1] + 1),
        np.append(boc_intensity, np.nan),
        biased=False
    )

    assert np.isclose(
        boc_raw_skew_biased,
        np.nanmean((boc_intensity - np.nanmean(boc_intensity))**3) / np.nanstd(boc_intensity, ddof=0)**3,
        rtol=1e-10,
    )
    assert np.isclose(
        boc_raw_skew_unbiased,
        bochum_data_pd.skew(),
        rtol=1e-10,
    )

def test_acf_core_vs_isnap(elmdon_data_np):
    elm_time, elm_intensity = elmdon_data_np

    # The intensity is in the unit of mm/hr. So we divide the time by 3600 to convert it to hours.
    elm_isnap = IndexedSnapshot(elm_time / 3600, elm_intensity)
    elm_isnap_acf = elm_isnap.autocorr_coef()

    elm_raw_time = elm_time / 3600
    elm_raw_acf = _isnapshot_acf(
        np.append(elm_raw_time, elm_raw_time[-1] + 1), np.append(elm_intensity, np.nan), lag=1
    )

    assert np.allclose(elm_isnap_acf, elm_raw_acf, rtol=1e-10)

def test_acf_core_vs_np_pd(elmdon_data_np, elmdon_data_pd):
    # We don't comapre with pandas.autocorr. Because it uses Pearson correlation coefficient.
    elm_time, elm_intensity = elmdon_data_np

    elm_raw_time = elm_time / 3600
    elm_raw_acf = _isnapshot_acf(
        np.append(elm_raw_time, elm_raw_time[-1] + 1), np.append(elm_intensity, np.nan), lag=1
    )

    mean = np.mean(elm_intensity)
    x_i = elm_intensity[:-1]
    y_i = elm_intensity[1:]
    xi_minus_mean = x_i - mean
    yi_minus_mean = y_i - mean
    numerator = np.sum(xi_minus_mean * yi_minus_mean)
    denominator = np.sum((elm_intensity - mean) ** 2)

    assert np.allclose(elm_raw_acf, numerator/denominator, rtol=1e-10)

def test_acf_with_nan_core_vs_isnap(bochum_data_np):
    boc_time, boc_intensity = bochum_data_np

    boc_isnap = IndexedSnapshot(boc_time / 300, boc_intensity)
    boc_isnap_acf = boc_isnap.autocorr_coef()

    boc_raw_time = boc_time / 300
    boc_raw_acf = _isnapshot_acf(
        np.append(boc_raw_time, boc_raw_time[-1] + 1), np.append(boc_intensity, np.nan), lag=1
    )

    assert np.allclose(boc_isnap_acf, boc_raw_acf, rtol=1e-10)

def test_acf_with_nan_core_vs_pd(bochum_data_np, bochum_data_pd):
        # We don't comapre with pandas.autocorr. Because it uses Pearson correlation coefficient.

    boc_time, boc_intensity = bochum_data_np

    boc_raw_time = boc_time / 300
    boc_raw_acf = _isnapshot_acf(
        np.append(boc_raw_time, boc_raw_time[-1] + 1), np.append(boc_intensity, np.nan), lag=1
    )

    mean = np.nanmean(boc_intensity)
    x_i = boc_intensity[:-1]
    y_i = boc_intensity[1:]
    xi_minus_mean = x_i - mean
    yi_minus_mean = y_i - mean
    numerator = np.nansum(xi_minus_mean * yi_minus_mean)
    denominator = np.nansum((boc_intensity - mean) ** 2)

    assert np.allclose(boc_raw_acf, numerator/denominator, rtol=1e-10)

def test_rescale_core_vs_isnap(elmdon_data_np):
    elm_time, elm_intensity = elmdon_data_np

    # The intensity is in the unit of mm/hr. So we divide the time by 3600 to convert it to hours.
    elm_isnap = IndexedSnapshot(elm_time / 3600, elm_intensity)
    elm_isnap_rescaled_3 = elm_isnap.rescale(3, 1e-10)
    elm_isnap_rescaled_6 = elm_isnap.rescale(6, 1e-10)
    elm_isnap_rescaled_12 = elm_isnap.rescale(12, 1e-10)
    elm_isnap_rescaled_24 = elm_isnap.rescale(24, 1e-10)

    elm_raw_time = elm_time / 3600
    elm_raw_time_rescaled_3, elm_raw_intensity_rescaled_3 = _isnapshot_rescale(
        np.append(elm_raw_time, elm_raw_time[-1] + 1), np.append(elm_intensity, np.nan), 3, 1e-10
    )
    elm_raw_time_rescaled_6, elm_raw_intensity_rescaled_6 = _isnapshot_rescale(
        np.append(elm_raw_time, elm_raw_time[-1] + 1), np.append(elm_intensity, np.nan), 6, 1e-10
    )
    elm_raw_time_rescaled_12, elm_raw_intensity_rescaled_12 = _isnapshot_rescale(
        np.append(elm_raw_time, elm_raw_time[-1] + 1), np.append(elm_intensity, np.nan), 12, 1e-10
    )
    elm_raw_time_rescaled_24, elm_raw_intensity_rescaled_24 = _isnapshot_rescale(
        np.append(elm_raw_time, elm_raw_time[-1] + 1), np.append(elm_intensity, np.nan), 24, 1e-10
    )

    # Intensity is different due to floating point error. The amout of addition is different for isnap intensity and raw intensity.
    print(len(elm_isnap_rescaled_3.time), len(elm_raw_time_rescaled_3))
    print(elm_isnap_rescaled_3.time[-3:], elm_isnap_rescaled_3.intensity[-3:])
    print(elm_raw_time_rescaled_3[-3:], elm_raw_intensity_rescaled_3[-3:])
    for i in range(len(elm_raw_time_rescaled_3)):
        if not np.isclose(elm_isnap_rescaled_3.time[i], elm_raw_time_rescaled_3[i], rtol=1e-10):
            print(i)
            print(elm_isnap_rescaled_3.time[i], elm_raw_time_rescaled_3[i], np.isclose(elm_isnap_rescaled_3.time[i], elm_raw_time_rescaled_3[i], rtol=1e-10))
    assert np.all(elm_isnap_rescaled_3.time == elm_raw_time_rescaled_3)
    assert np.allclose(elm_isnap_rescaled_3.intensity[:-1], elm_raw_intensity_rescaled_3[:-1], rtol=1e-10)
    assert np.all(elm_isnap_rescaled_6.time == elm_raw_time_rescaled_6)
    assert np.allclose(elm_isnap_rescaled_6.intensity[:-1], elm_raw_intensity_rescaled_6[:-1], rtol=1e-10)
    assert np.all(elm_isnap_rescaled_12.time == elm_raw_time_rescaled_12)
    assert np.allclose(elm_isnap_rescaled_12.intensity[:-1], elm_raw_intensity_rescaled_12[:-1], rtol=1e-10)
    assert np.all(elm_isnap_rescaled_24.time == elm_raw_time_rescaled_24)
    assert np.allclose(elm_isnap_rescaled_24.intensity[:-1], elm_raw_intensity_rescaled_24[:-1], rtol=1e-10)
    assert np.isnan(elm_isnap_rescaled_3.intensity[-1])
    assert np.isnan(elm_raw_intensity_rescaled_3[-1])
    assert np.isnan(elm_isnap_rescaled_6.intensity[-1])
    assert np.isnan(elm_raw_intensity_rescaled_6[-1])
    assert np.isnan(elm_isnap_rescaled_12.intensity[-1])
    assert np.isnan(elm_raw_intensity_rescaled_12[-1])
    assert np.isnan(elm_isnap_rescaled_24.intensity[-1])
    assert np.isnan(elm_raw_intensity_rescaled_24[-1])


def test_rescale_core_vs_pd(elmdon_data_np, elmdon_data_pd):
    elm_time, elm_intensity = elmdon_data_np

    elm_isnap = IndexedSnapshot(elm_time / 3600, elm_intensity)
    _, isnap_intensity_3 = elm_isnap.rescale(3, 1e-10).unpack()
    _, isnap_intensity_6 = elm_isnap.rescale(6, 1e-10).unpack()
    _, isnap_intensity_12 = elm_isnap.rescale(12, 1e-10).unpack()
    _, isnap_intensity_24 = elm_isnap.rescale(24, 1e-10).unpack()


    # Calculate time difference between 3 index in the pandas data
    time_dif = elmdon_data_pd.index[1] - elmdon_data_pd.index[0]
    elm_pd_3 = elmdon_data_pd.resample(time_dif*3).sum()
    elm_pd_6 = elmdon_data_pd.resample(time_dif*6).sum()
    elm_pd_12 = elmdon_data_pd.resample(time_dif*12).sum()
    elm_pd_24 = elmdon_data_pd.resample(time_dif*24).sum()

    assert np.allclose(isnap_intensity_3, elm_pd_3.values, rtol=1e-10)
    assert np.allclose(isnap_intensity_6, elm_pd_6.values, rtol=1e-10)
    assert np.allclose(isnap_intensity_12, elm_pd_12.values, rtol=1e-10)
    assert np.allclose(isnap_intensity_24, elm_pd_24.values, rtol=1e-10)

def test_rescale_with_nan_core_vs_isnap(bochum_data_np):
    boc_time, boc_intensity = bochum_data_np

    boc_isnap = IndexedSnapshot(boc_time / 300, boc_intensity)
    boc_isnap_rescaled_3 = boc_isnap.rescale(3, 1e-10)
    boc_isnap_rescaled_6 = boc_isnap.rescale(6, 1e-10)
    boc_isnap_rescaled_12 = boc_isnap.rescale(12, 1e-10)
    boc_isnap_rescaled_24 = boc_isnap.rescale(24, 1e-10)

    boc_raw_time = boc_time / 300
    boc_raw_time_rescaled_3, boc_raw_intensity_rescaled_3 = _isnapshot_rescale(
        np.append(boc_raw_time, boc_raw_time[-1] + 1), np.append(boc_intensity, np.nan), 3, 1e-10
    )
    boc_raw_time_rescaled_6, boc_raw_intensity_rescaled_6 = _isnapshot_rescale(
        np.append(boc_raw_time, boc_raw_time[-1] + 1), np.append(boc_intensity, np.nan), 6, 1e-10
    )
    boc_raw_time_rescaled_12, boc_raw_intensity_rescaled_12 = _isnapshot_rescale(
        np.append(boc_raw_time, boc_raw_time[-1] + 1), np.append(boc_intensity, np.nan), 12, 1e-10
    )
    boc_raw_time_rescaled_24, boc_raw_intensity_rescaled_24 = _isnapshot_rescale(
        np.append(boc_raw_time, boc_raw_time[-1] + 1), np.append(boc_intensity, np.nan), 24, 1e-10
    )

    # Intensity is different due to floating point error. The amout of addition is different for isnap intensity and raw intensity.
    for i in range(len(boc_raw_time_rescaled_3)):
        if not np.isclose(boc_isnap_rescaled_3.time[i], boc_raw_time_rescaled_3[i], rtol=1e-10):
            print(i)
            print(boc_isnap_rescaled_3.time[i-2: i+3], boc_raw_time_rescaled_3[i-2: i+3], np.isclose(boc_isnap_rescaled_3.time[i], boc_raw_time_rescaled_3[i], rtol=1e-10))
    assert np.all(boc_isnap_rescaled_3.time == boc_raw_time_rescaled_3)
    assert np.allclose(boc_isnap_rescaled_3.intensity[:-1], boc_raw_intensity_rescaled_3[:-1], rtol=1e-10, equal_nan=True)
    assert np.all(boc_isnap_rescaled_6.time == boc_raw_time_rescaled_6)
    assert np.allclose(boc_isnap_rescaled_6.intensity[:-1], boc_raw_intensity_rescaled_6[:-1], rtol=1e-10, equal_nan=True)
    assert np.all(boc_isnap_rescaled_12.time == boc_raw_time_rescaled_12)
    assert np.allclose(boc_isnap_rescaled_12.intensity[:-1], boc_raw_intensity_rescaled_12[:-1], rtol=1e-10, equal_nan=True)
    assert np.all(boc_isnap_rescaled_24.time == boc_raw_time_rescaled_24)
    assert np.allclose(boc_isnap_rescaled_24.intensity[:-1], boc_raw_intensity_rescaled_24[:-1], rtol=1e-10, equal_nan=True)
    assert np.isnan(boc_isnap_rescaled_3.intensity[-1])
    assert np.isnan(boc_raw_intensity_rescaled_3[-1])
    assert np.isnan(boc_isnap_rescaled_6.intensity[-1])
    assert np.isnan(boc_raw_intensity_rescaled_6[-1])
    assert np.isnan(boc_isnap_rescaled_12.intensity[-1])
    assert np.isnan(boc_raw_intensity_rescaled_12[-1])
    assert np.isnan(boc_isnap_rescaled_24.intensity[-1])
    assert np.isnan(boc_raw_intensity_rescaled_24[-1])


def test_rescale_with_nan_core_vs_pd(bochum_data_np, bochum_data_pd):
    boc_time, boc_intensity = bochum_data_np
    print(f"length of boc_time: {len(boc_time)}")
    print(f"length of boc_intensity: {len(boc_intensity)}")
    print(f"length of bochum_data_pd: {len(bochum_data_pd)}")
    print(f"Is there nan in numpy intensity? Ans: ", np.any(np.isnan(boc_intensity)))
    print(f"Is there nan in Series intensity? Ans: ", np.any(np.isnan(bochum_data_pd)))

    boc_isnap = IndexedSnapshot(boc_time / 300, boc_intensity)
    _, isnap_intensity_5m = boc_isnap.rescale(1, 1e-10).unpack()
    _, isnap_intensity_1h = boc_isnap.rescale(12, 1e-10).unpack()
    _, isnap_intensity_6h = boc_isnap.rescale(72, 1e-10).unpack()
    _, isnap_intensity_24h = boc_isnap.rescale(288, 1e-10).unpack()

    # Calculate time difference between 3 index in the pandas data
    time_dif = bochum_data_pd.index[1] - bochum_data_pd.index[0]
    boc_pd_5m = bochum_data_pd.resample(time_dif*1).sum(min_count=1)
    boc_pd_1h = bochum_data_pd.resample(time_dif*12).sum(min_count=1)
    boc_pd_6h = bochum_data_pd.resample(time_dif*72).sum(min_count=1)
    boc_pd_24h = bochum_data_pd.resample(time_dif*288).sum(min_count=1)

    assert np.allclose(isnap_intensity_5m, boc_pd_5m.values, rtol=1e-10, equal_nan=True)
    assert np.allclose(isnap_intensity_1h, boc_pd_1h.values, rtol=1e-10, equal_nan=True)
    assert np.allclose(isnap_intensity_6h, boc_pd_6h.values, rtol=1e-10, equal_nan=True)
    assert np.allclose(isnap_intensity_24h, boc_pd_24h.values, rtol=1e-10, equal_nan=True)