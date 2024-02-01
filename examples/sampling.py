from pybl.models import BLRPRx, Stat_Props, BLRPRx_params
from pybl.raincell.rcimodel import ExponentialRCIModel
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

## Experiment configuration
sample_size = 100
sample_duration = 876000  # 100 years
rng = np.random.default_rng(100)

para_data = pd.read_csv('./5Y_1_Theta.csv')

paras = pd.DataFrame()
#1
paras['lambda_'] = para_data[list(para_data.keys())[0]]
#2
paras['phi'] = para_data[list(para_data.keys())[-2]]
#3
paras['kappa'] = para_data[list(para_data.keys())[-3]]
#4
paras['alpha'] = para_data[list(para_data.keys())[2]]
#5
paras['nu'] = para_data[list(para_data.keys())[2]]/para_data[list(para_data.keys())[3]]
#6
paras['sigma_mux'] = para_data[list(para_data.keys())[-1]]
#7
paras['iota'] = para_data[list(para_data.keys())[1]]

print(paras)

single_month = paras.iloc[0]

#params = BLRPRx_params(
#    lambda_ = single_month[0].item(),
#    phi = single_month[1].item(),
#    kappa = single_month[2].item(),
#    alpha = single_month[3].item(),
#    nu = single_month[4].item(),
#    sigmax_mux = single_month[5].item(),
#    iota = single_month[6].item()
#)

## elmdon January prameters
#params = BLRPRx_params(
#    lambda_=0.016679733103341976,
#    phi=0.08270236178820184,
#    kappa=0.34970877070925505,
#    alpha=9.017352714561754,
#    nu=0.9931496975448589,
#    sigmax_mux=1.0,
#    iota=0.971862948182735,
#)

params = BLRPRx_params(
    lambda_=0.016679733103341976,
    phi=0.08270236178820184,
    kappa=0.04970877070925505,
    alpha=0.5052714561754,
    nu=0.2031496975448589,
    sigmax_mux=1.0,
    iota=0.971862948182735,
)

model = BLRPRx(params=params, sampling_rng=rng, rci_model=ExponentialRCIModel())
print(params)

## Calculate the theoretical mean, cvar, ar1, skewness
model_df = pd.DataFrame(
    columns=["Mean", "CVaR", "AR1", "Skewness"], index=[1/12, 1, 3, 6, 24]
)
for i, scale in enumerate([1/12, 1, 3, 6, 24]):
    model_df.loc[scale, "Mean"] = model.get_prop(prop=Stat_Props.MEAN, timescale=scale)
    model_df.loc[scale, "CVaR"] = model.get_prop(prop=Stat_Props.CVAR, timescale=scale)
    model_df.loc[scale, "AR1"] = model.get_prop(prop=Stat_Props.AR1, timescale=scale)
    model_df.loc[scale, "Skewness"] = model.get_prop(
        prop=Stat_Props.SKEWNESS, timescale=scale
    )

## Start sampling
result_df = np.empty(sample_size, dtype=object)
result = np.empty((sample_size, 5, 4), dtype=np.float64)
mean = np.empty(sample_size, dtype=np.float64)
for exp in range(sample_size):
    ts = model.sample(sample_duration)
    
    ts_rescale = []
    ts_rescale.append(ts.rescale(1/12))
    ts_rescale.append(ts.rescale(1))
    ts_rescale.append(ts.rescale(3))
    ts_rescale.append(ts.rescale(6))
    ts_rescale.append(ts.rescale(24))

    result_df[exp] = pd.DataFrame(
        columns=["Mean", "CVaR", "AR1", "Skewness"], index=[1/12, 1, 3, 6, 24]
    )

    for i, scale in enumerate([1/12, 1, 3, 6, 24]):
        result_df[exp].loc[scale, "Mean"] = ts_rescale[i].mean()
        result_df[exp].loc[scale, "CVaR"] = ts_rescale[i].cvar()
        result_df[exp].loc[scale, "AR1"] = ts_rescale[i].acf(1)
        result_df[exp].loc[scale, "Skewness"] = ts_rescale[i].skewness()

    result[exp] = result_df[exp].to_numpy()
    print(f"Experiment {exp} completed")

# Plot the results
for i, scale in enumerate([1/12, 1, 3, 6, 24]):
    # Set the font size
    plt.rcParams.update({"font.size": 6})
    # Plot bin chart for mean, cvar, ar1, skewness
    fig, axs = plt.subplots(2, 2)
    fig.suptitle(
        f"Scale: {scale}, Sample size {sample_size}, Duration {sample_duration}hr"
    )
    # axs[0, 0].hist(result[:, i, 0], bins=100)
    axs[0, 0].boxplot(result[:, i, 0])
    axs[0, 0].scatter(1, model_df.loc[scale, "Mean"], color="red")
    axs[0, 0].set_title(
        f'Mean, median={np.median(result[:, i, 0]):.5f}, theoretical={model_df.loc[scale, "Mean"]:.5f}'
    )
    # axs[0, 1].hist(result[:, i, 1], bins=100)
    axs[0, 1].boxplot(result[:, i, 1])
    axs[0, 1].scatter(1, model_df.loc[scale, "CVaR"], color="red")
    axs[0, 1].set_title(
        f'CVaR, median={np.median(result[:, i, 1]):.5f}, theoretical={model_df.loc[scale, "CVaR"]:.5f}'
    )
    # axs[1, 0].hist(result[:, i, 2], bins=100)
    axs[1, 0].boxplot(result[:, i, 2])
    axs[1, 0].scatter(1, model_df.loc[scale, "AR1"], color="red")
    axs[1, 0].set_title(
        f'AR1, median={np.median(result[:, i, 2]):.5f}, theoretical={model_df.loc[scale, "AR1"]:.5f}'
    )
    # axs[1, 1].hist(result[:, i, 3], bins=100)
    axs[1, 1].boxplot(result[:, i, 3])
    axs[1, 1].scatter(1, model_df.loc[scale, "Skewness"], color="red")
    axs[1, 1].set_title(
        f'Skewness, median={np.median(result[:, i, 3]):.5f}, theoretical={model_df.loc[scale, "Skewness"]:.5f}'
    )
    fig.tight_layout()
    plt.savefig(f"scale_{scale}.png")
    plt.close()
