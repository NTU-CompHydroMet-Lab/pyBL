from abc import ABC, abstractmethod


class Fitting:
    def __init__(self, args):
        prop = pd.read_csv(args.IO.config_path, index_col=0, header=None)
        self.properties_to_fit = prop.loc["prop"].dropna().to_numpy()
        self.timescale_to_fit = prop.loc["time"].dropna().to_numpy()
        self.stats_path = args.IO.stats_file_path
        self.theta_initial = args.fitting.theta_initial

    def fit(self, args):
        num_month = pd.read_csv(self.stats_path, index_col=0, header=0).shape[0]

        if self.theta_initial is None:
            theta[0] = 0.01  # lambda #
            theta[1] = 0.1  # iota #
            theta[2] = 1.0  # sigma/mu
            theta[3] = 1e-6  # spare space
            theta[4] = 2.1  # alpha #
            theta[5] = 7.1  # alpha/nu
            theta[6] = 0.1  # kappa #
            theta[7] = 0.1  # phi #
            theta[8] = 1.0  # c
        else:
            theta = self.theta_initial
