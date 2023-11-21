from pybl.models import BLRPRx, Stat_Props, BLRPRx_params
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

## Experiment configuration
sample_size = 1000
sample_duration = 1000000
rng = np.random.default_rng(100)
params = BLRPRx_params(
    lambda_=0.016679733103341976,
    phi=0.08270236178820184,
    kappa=0.34970877070925505,
    alpha=9.017352714561754,
    nu=0.9931496975448589,
    sigmax_mux=1.0,
    iota=0.971862948182735,
)
model = BLRPRx(params=params, sampling_rng=rng)

## Calculate the theoretical mean, cvar, ar1, skewness
model_df = pd.DataFrame(
    columns=["Mean", "CVaR", "AR1", "Skewness"], index=[1, 3, 6, 24]
)
for i, scale in enumerate([1, 3, 6, 24]):
    model_df.loc[scale, "Mean"] = model.get_prop(prop=Stat_Props.MEAN, timescale=scale)
    model_df.loc[scale, "CVaR"] = model.get_prop(prop=Stat_Props.CVAR, timescale=scale)
    model_df.loc[scale, "AR1"] = model.get_prop(prop=Stat_Props.AR1, timescale=scale)
    model_df.loc[scale, "Skewness"] = model.get_prop(
        prop=Stat_Props.SKEWNESS, timescale=scale
    )

## Start sampling
result_df = np.empty(sample_size, dtype=object)
result = np.empty((sample_size, 4, 4), dtype=np.float64)
for exp in range(sample_size):
    ts = model.sample(sample_duration)
    ts_rescale = []
    ts_rescale.append(ts.rescale(1))
    ts_rescale.append(ts_rescale[0].rescale(3))
    ts_rescale.append(ts_rescale[1].rescale(2))
    ts_rescale.append(ts_rescale[2].rescale(4))

    result_df[exp] = pd.DataFrame(
        columns=["Mean", "CVaR", "AR1", "Skewness"], index=[1, 3, 6, 24]
    )

    for i, scale in enumerate([1, 3, 6, 24]):
        result_df[exp].loc[scale, "Mean"] = ts_rescale[i].mean()
        result_df[exp].loc[scale, "CVaR"] = ts_rescale[i].cvar()
        result_df[exp].loc[scale, "AR1"] = ts_rescale[i].acf(1)
        result_df[exp].loc[scale, "Skewness"] = ts_rescale[i].skewness()

    result[exp] = result_df[exp].to_numpy()
    print(f"Experiment {exp} completed")

## Plot the results
for i, scale in enumerate([1, 3, 6, 24]):
    # Set the font size
    plt.rcParams.update({"font.size": 6})
    # Plot bin chart for mean, cvar, ar1, skewness
    fig, axs = plt.subplots(2, 2)
    fig.suptitle(
        f"Scale: {scale}, Sample size {sample_size}, Duration {sample_duration}hr"
    )
    axs[0, 0].hist(result[:, i, 0], bins=100)
    axs[0, 0].set_title(
        f'Mean, median={np.median(result[:, i, 0]):.5f}, theoretical={model_df.loc[scale, "Mean"]:.5f}'
    )
    axs[0, 1].hist(result[:, i, 1], bins=100)
    axs[0, 1].set_title(
        f'CVaR, median={np.median(result[:, i, 1]):.5f}, theoretical={model_df.loc[scale, "CVaR"]:.5f}'
    )
    axs[1, 0].hist(result[:, i, 2], bins=100)
    axs[1, 0].set_title(
        f'AR1, median={np.median(result[:, i, 2]):.5f}, theoretical={model_df.loc[scale, "AR1"]:.5f}'
    )
    axs[1, 1].hist(result[:, i, 3], bins=100)
    axs[1, 1].set_title(
        f'Skewness, median={np.median(result[:, i, 3]):.5f}, theoretical={model_df.loc[scale, "Skewness"]:.5f}'
    )
    fig.tight_layout()
    plt.savefig(f"scale_{scale}.png")
    plt.close()
