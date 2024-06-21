from pybl.utils.timeseries import preprocess_classic
import numpy as np


def test_elmdon_stats_weight(elmdon_data_np, elmdon_stats_weight):
    timescale = np.array([1, 3, 6, 24], dtype=np.float64)*3600
    elm_time, elm_intensity = elmdon_data_np
    calc_target, calc_weight = preprocess_classic(
        elm_time, elm_intensity / 3600, timescale=timescale
    )

    true_stats, true_weight = elmdon_stats_weight

    assert np.allclose(calc_target, true_stats, rtol=1e-10)
    assert np.allclose(calc_weight, true_weight, rtol=1e-10)


def test_bochum_stats_weight(bochum_data_np, bochum_stats_weight):
    timescale = np.array([1, 12, 72, 288], dtype=np.float64)*300
    boc_time, boc_intensity = bochum_data_np
    calc_target, calc_weight = preprocess_classic(
        boc_time, boc_intensity / 300, timescale=timescale
    )

    true_stats, true_weight = bochum_stats_weight

    assert np.allclose(calc_target, true_stats, rtol=1e-10)
    assert np.allclose(calc_weight, true_weight, rtol=1e-10)


