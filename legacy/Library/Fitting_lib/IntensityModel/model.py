from abc import ABC, abstractmethod


class root_RCI_model(ABC):
    @abstractmethod
    def get_f1(self):
        pass

    @abstractmethod
    def get_f2(self):
        pass

    @abstractmethod
    def sample_intensity(self):
        pass


class gamma_RCI_model(root_RCI_model):
    def get_f1(self, a):
        return (a + 1.0) / a

    def get_f2(self, a):
        a2 = a * a
        return (a2 + 3.0 * a + 2.0) / a2

    def sample_intensity(para_ins):
        """
        :param para_ins: list, contains shape and scale parameters of gamma distribution to generate random rainfall intensity
        :return rainfall_intensity: float
        """

        # cell intensity (mm/h)
        mux, I_shape = para_ins[0], para_ins[1]

        I_scale = mux / I_shape

        rainfall_intensity = gamma(I_shape, I_scale)

        return rainfall_intensity


class exponential_RCI_model(root_RCI_model):
    def get_f1(self, a):
        return 2.0

    def get_f2(self, a):
        return 6.0

    def sample_intensity(para_ins):
        """
        :param para_ins: list, contains shape and scale parameters of gamma distribution to generate random rainfall intensity
        :return rainfall_intensity: float
        """

        # cell intensity (mm/h)
        mux, sigmax_mux = para_ins[0], para_ins[1]

        I_shape = 1.0 / sigmax_mux / sigmax_mux
        I_scale = sigmax_mux * sigmax_mux * mux

        rainfall_intensity = gamma(I_shape, I_scale)

        return rainfall_intensity
