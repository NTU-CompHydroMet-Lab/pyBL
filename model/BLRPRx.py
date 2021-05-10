import math
import scipy.special as sc
from IntensityModel.model import gamma_RCI_model, exponential_RCI_model
from datetime import datetime


class BLRPRx():
    def __init__(self, mode):
        super().__init__()
        if mode == 'gamma':
            self.RCI_model = gamma_RCI_model()
        elif mode == 'exponential':
            self.RCI_model = exponential_RCI_model()

    def sample_storm(self, theta: list, simulation_hours: int, start_time: datetime) -> list:
        """
        :param intensity_model: str, rainfall intensity model
        :param theta: list, a list of parameters
        :param simulation_hours: int, total time length with hour as unit of simulation
        :param start_time: time object, start time of simulation
        :return storms: list, a list of Storm object
        """
        # all operations are working with unit = hour

        ld = theta[0]  # storm arrival rate
        iota = theta[1]  # ratio of mean cell intensity to eta
        alpha = theta[2]  # shape parameter for gamma distribution of eta
        nu = alpha / theta[3]  # scale parameter for gamma distribution, thetas[3]=alpha/nu
        kappa = theta[4]  # cell arrival rate
        phi = theta[5]  # storm termination rate

        # the number of storms within the 'simulation' hours - a Poisson distributed random number
        n_storms = poisson.rvs(ld * simulation_hours)
        print("%d samples of storms are generated." % n_storms)

        storms = list()

        # generate random variables for each storm
        etas = [gamma(alpha, 1 / nu) for s in range(n_storms)]
        gamas = [phi * etas[s] for s in range(n_storms)]
        betas = [kappa * etas[s] for s in range(n_storms)]
        muxs = [iota * etas[s] for s in range(n_storms)]

        # generate each storm's duration
        duration_storm_hrs = [expon.rvs(scale=1 / gamas[s]) for s in range(n_storms)]

        # generate each storm's number of cells
        n_cells_list = [1 + poisson.rvs(betas[s] * duration_storm_hrs[s]) for s in range(n_storms)]

        # generate each storm's start time and calculate each storm's end time
        sDT_storm_hrs = [start_time + timedelta(hours=(simulation_hours * uniform.rvs())) for s in range(n_storms)]
        eDT_storm_hrs = [sDT_storm_hrs[s] + timedelta(hours=duration_storm_hrs[s]) for s in range(n_storms)]

        sigmax_mux = 1.0

        # single storm simulation
        for s in range(n_storms):
            storm = Storm(n_cells_list[s])
            storm.sDT, storm.eDT = sDT_storm_hrs[s], eDT_storm_hrs[s]

            # generate each cell's duration
            duration_cell_hrs = expon.rvs(scale=1 / etas[s], size=n_cells_list[s])

            # generate each cell's start time and calculate each cell's end time
            cell_sDTs = [sDT_storm_hrs[s] + timedelta(hours=(duration_storm_hrs[s] * uniform.rvs())) for c in
                        range(n_cells_list[s])]
            cell_sDTs[0] = sDT_storm_hrs[s]
            cell_eDTs = [cell_sDTs[c] + timedelta(hours=duration_cell_hrs[c]) for c in range(n_cells_list[s])]

            # generate each cell's rainfall intensity
            para_xis = [[muxs[s], sigmax_mux] for c in range(n_cells_list[s])]
            cell_Depths = [self.RCI_model.sample_intensity(para_xis[c]) for c in range(n_cells_list[s])]

            for c in range(n_cells_list[s]):
                storm.cells[c] = Cell()
                storm.cells[c].sDT = cell_sDTs[c]
                storm.cells[c].eDT = cell_eDTs[c]
                storm.cells[c].Depth = cell_Depths[c]

            storms.append(storm)
        return storms

    def kernal(self, k, x, nu, alpha):
        alpha = (alpha + 0.00000001) if ((alpha <= 4.0)
                                         and (math.modf(alpha) == 0.0)) else alpha

        if (alpha - k) >= 171.0 or alpha >= 171.0:
            return 1000000000.0

        else:
            try:
                return math.pow((nu / (nu + x)), alpha) * math.pow((nu + x), k) * math.gamma(alpha - k) / math.gamma(alpha)
            except:
                return 1000000000.0

    def Mean(self, theta):
        the_lambda = theta[1]
        iota = theta[2]
        alpha = theta[5]
        nu = alpha / theta[6]
        kappa = theta[7]
        phi = theta[8]
        c = theta[9]
        h = theta[0]
        muc = 1.0 + kappa / phi

        return the_lambda * h * iota * muc * self.kernal(1.0 - c, 0, nu, alpha)

    def Var(self, theta):
        the_lambda = theta[1]
        iota = theta[2]
        alpha = theta[5]
        nu = alpha / theta[6]
        kappa = theta[7]
        phi = theta[8]
        c = theta[9]
        h = theta[0]
        muc = 1.0 + kappa / phi

        phi2 = phi*phi
        phi3 = phi*phi2

        r = theta[3]
        f1 = self.RCI_model.get_f1(r)

        k = 3.0 - 2.0*c
        kappa_phi2_1 = kappa / (phi2 - 1.0)
        kappa_phi2_phi2_1 = kappa_phi2_1 / phi2

        var_p1 = self.kernal(2.0 - 2.0*c, 0, nu, alpha)*(f1 + (kappa / phi)) * h
        var_p2 = self.kernal(k, 0.0, nu, alpha) * \
            ((kappa_phi2_phi2_1 * (1.0 - phi3)) - f1)
        var_p3 = self.kernal(k, phi*h, nu, alpha)*(kappa_phi2_phi2_1)
        var_p4 = self.kernal(k, h, nu, alpha)*(f1 + (kappa_phi2_1*phi))

        # std::cout << kernal(2.0 - 2.0*c, 0, nu, alpha) << ' ' << kernal(k, 0.0, nu, alpha) << ' ' << kernal(k, phi*h, nu, alpha) << ' ' << kernal(k, h, nu, alpha) << std::endl
        # std::cout << var_p1 << ' ' << var_p2 << ' ' << var_p3 << ' ' << var_p4 << std::endl
        ans = 2.0*the_lambda*muc*iota*iota*(var_p1 + var_p2 - var_p3 + var_p4)
        return ans if ans >= 0 else 99999999.0

    def Cov(self, theta, lag=1):
        the_lambda = theta[1]
        iota = theta[2]
        alpha = theta[5]
        nu = alpha / theta[6]
        kappa = theta[7]
        phi = theta[8]
        c = theta[9]
        h = theta[0]
        muc = 1.0 + kappa / phi

        phi2 = phi*phi
        phi3 = phi*phi2

        r = theta[3]
        f1 = self.RCI_model.get_f1(r)

        k = 3.0 - 2.0*c

        cov_p1 = (f1 + kappa*phi / (phi2 - 1.0))*(self.kernal(k, h*(lag - 1.0), nu, alpha)
                                                  - 2.0 * self.kernal(k, lag*h, nu, alpha) + self.kernal(k, h*(lag + 1.0), nu, alpha))

        cov_p2 = (kappa / phi2 / (phi2 - 1.0))*(self.kernal(k, phi*(lag - 1.0)*h, nu, alpha)
                                                - 2.0*self.kernal(k, phi*lag*h, nu, alpha) + self.kernal(k, phi*(lag + 1.0)*h, nu, alpha))

        # /*std::cout << kernal(k, h*(lag - 1.0), nu, alpha)
        #     - 2.0 * kernal(k, lag*h, nu, alpha)
        #     + kernal(k, h*(lag + 1.0), nu, alpha)
        #     << ' ' << kernal(k, phi*(lag - 1.0)*h, nu, alpha)
        #     - 2.0*kernal(k, phi*lag*h, nu, alpha)
        #     + kernal(k, phi*(lag + 1.0)*h, nu, alpha) << endl*/

        # std::cout << kernal(k, h*(lag - 1.0), nu, alpha) << ' ' << kernal(k, lag*h, nu, alpha) << ' ' << kernal(k, phi*(lag - 1.0)*h, nu, alpha) << std::endl

        return the_lambda*muc*iota*iota*(cov_p1 - cov_p2)

    def Cov2(self, theta, eta0=0.001, lag=1):
        the_lambda = theta[1]
        iota = theta[2]
        alpha = theta[5]
        nu = alpha / theta[6]
        kappa = theta[7]
        phi = theta[8]
        c = theta[9]
        h = theta[0]
        muc = 1.0 + kappa / phi

        phi2 = phi*phi
        phi3 = phi*phi2

        r = theta[3]
        f1 = self.RCI_model.get_f1(r)

        s = alpha - 1
        x0 = (nu + phi*(lag - 1.0)*h)
        x = eta0 * x0
        upper_cov_c1 = sc.gammaincc(s, x) * sc.gamma(s) * math.pow(
            nu, alpha) / gsl_sf_gamma(alpha) / math.pow(x0, alpha-1.0)
        x0 = (nu + (lag - 1.0)*h)
        x = eta0 * x0
        upper_cov_c1_phi_1 = sc.gammaincc(
            s, x) * sc.gamma(s)*math.pow(nu, alpha) / gsl_sf_gamma(alpha) / math.pow(x0, alpha - 1.0)

        x0 = (nu + phi*lag*h)
        x = eta0 * x0
        upper_cov_c2 = -2.0 * sc.gammaincc(s, x) * sc.gamma(s)*math.pow(
            nu, alpha) / gsl_sf_gamma(alpha) / math.pow(x0, alpha - 1.0)
        x0 = (nu + lag*h)
        x = eta0 * x0
        upper_cov_c2_phi_1 = -2.0 * sc.gammaincc(s, x) * sc.gamma(s)*math.pow(
            nu, alpha) / gsl_sf_gamma(alpha) / math.pow(x0, alpha - 1.0)

        x0 = (nu + phi*(lag + 1.0)*h)
        x = eta0 * x0
        upper_cov_c3 = sc.gammaincc(
            s, x) * sc.gamma(s)*math.pow(nu, alpha) / gsl_sf_gamma(alpha) / math.pow(x0, alpha - 1.0)
        x0 = (nu + (lag + 1.0)*h)
        x = eta0 * x0
        upper_cov_c3_phi_1 = sc.gammaincc(
            s, x)*sc.gamma(s) * math.pow(nu, alpha) / gsl_sf_gamma(alpha) / math.pow(x0, alpha - 1.0)

        # sl_sf_gamma_inc_P
        s = alpha + 1.0
        x = eta0*nu
        lower_cov = sc.gammainc(s, x)*math.gamma(alpha + 1.0) * \
            phi2*h*h / nu / math.gamma(alpha)
        lower_cov_phi_1 = sc.gammainc(
            s, x)*math.gamma(alpha + 1.0)*h*h / nu / math.gamma(alpha)

        # /*std::cout << upper_cov_c1_phi_1 + upper_cov_c2_phi_1 + upper_cov_c3_phi_1 + lower_cov_phi_1
        #     << ' ' << upper_cov_c1 + upper_cov_c2 + upper_cov_c3 + lower_cov << std::endl*/

        # cout << lower_cov_phi_1 << ' ' << lower_cov << endl

        return the_lambda*muc*iota*iota*(
            (f1+(kappa*phi/(phi2-1.0)))*(upper_cov_c1_phi_1 +
                                         upper_cov_c2_phi_1 + upper_cov_c3_phi_1 + lower_cov_phi_1)
            - ((kappa/phi2/(phi2-1.0))*(upper_cov_c1 + upper_cov_c2 + upper_cov_c3 + lower_cov)))

    def Mom3(self, theta):
        the_lambda = theta[1]
        iota = theta[2]
        alpha = theta[5]
        nu = alpha / theta[6]
        kappa = theta[7]
        phi = theta[8]
        c = theta[9]
        h = theta[0]
        muc = 1.0 + kappa / phi

        phi2 = phi*phi
        phi3 = phi*phi2
        phi4 = phi*phi3
        phi5 = phi*phi4
        phi6 = phi*phi5
        phi7 = phi*phi6
        phi8 = phi*phi7
        phi9 = phi*phi8

        kappa2 = kappa*kappa

        r = theta[3]
        f1 = self.RCI_model.get_f1(r)
        f2 = self.RCI_model.get_f2(r)

        if f2 < 0.0:
            return 10000000000000.0

        k1 = 4.0 - 3.0*c
        k2 = 3.0 - 3.0*c

        m3_p0 = (1.0 + 2.0*phi + phi2)*(phi4 - 2.0 *
                                        phi3 - 3.0*phi2 + 8.0*phi - 4.0)*(phi3)
        m3_p1 = self.kernal(k1, h, nu, alpha)*(12.0*phi7*kappa2 - 24.0*f1*phi2*kappa - 18.0*phi4*kappa2
                                          + 24.0*f1*phi3*kappa - 132.0*f1*phi6*kappa + 150.0*f1*phi4*kappa
                                          - 42.0*phi5*kappa2 - 6.0*f1*phi5*kappa + 108.0*phi5*f2
                                          - 72.0*phi7*f2 - 48.0*phi3*f2 + 24.0*f1*phi8*kappa
                                          + 12.0*phi3*kappa2 + 12.0*phi9*f2)
        m3_p2 = self.kernal(k2, h, nu, alpha)*(24.0*f1*phi4*h*kappa + 6.0*phi9*h*f2 - 30.0*f1*phi6*h*kappa + 6.0*f1*phi8*h*kappa
                                          + 54.0*phi5*h*f2 - 24.0*h*f2*phi3 - 36.0*phi7*h*f2)
        m3_p3 = self.kernal(k1, phi*h, nu, alpha)*(-48.0*kappa2 + 6.0*f1*phi4*kappa - 48.0*phi*f1*kappa + 6.0*phi5*kappa2
                                              - 24.0*f1*phi2*kappa + 36.0*f1*phi3*kappa - 6.0*f1*phi5*kappa
                                              + 84.0*phi2*kappa2 + 12.0*phi3*kappa2 - 18.0*phi4*kappa2)
        m3_p4 = self.kernal(k2, phi*h, nu, alpha)*(h*kappa2) * \
            (-24.0*phi + 30.0*phi3 - 6.0*phi5)
        m3_p5 = self.kernal(k1, 0.0, nu, alpha)*(72.0*phi7*f2 + 48.0*phi*f1*kappa + 24.0*f1*phi2*kappa - 36.0*f1*phi3*kappa
                                            - 84.0*phi2*kappa2 + 6.0*f1*phi5*kappa + 117.0*f1*phi6*kappa + 39.0*phi5*kappa2
                                            - 12.0*phi9*f2 - 138.0*f1*phi4*kappa + 48.0*kappa2
                                            - 9.0*phi7*kappa2 + 48.0*phi3*f2 + 18.0*phi4*kappa2
                                            - 21.0*phi8*f1*kappa - 12.0*phi3*kappa2 - 108.0*phi5*f2)
        m3_p6 = self.kernal(k2, 0.0, nu, alpha)*(h)*(-24.0*phi*kappa2 - 72.0*f1*phi6*kappa - 36.0*phi5*kappa2
                                                + 54.0*phi3*kappa2 + 6.0*phi7*kappa2 + 54.0*phi5*f2
                                                - 36.0*phi7*f2 - 24.0*phi3*f2 - 48.0*f1*phi2*kappa
                                                + 12.0*f1*phi8*kappa + 6.0*phi9*f2 + 108.0*f1*phi4*kappa)
        m3_p7 = self.kernal(k1, 2.0*h, nu, alpha)*(-12.0*f1*phi4*kappa - 3.0*f1*phi8*kappa + 15.0*f1*phi6*kappa
                                              - 3.0*phi7*kappa2 + 3.0*phi5*kappa2)
        m3_p8 = self.kernal(k1, h*(1.0 + phi), nu, alpha)*(-24.0*f1*phi3*kappa - 6.0*f1*phi4*kappa + 6.0*f1*phi5*kappa + 24.0*f1*phi2*kappa
                                                      + 18.0*phi4*kappa2 - 12.0*phi3*kappa2 - 6.0*phi5*kappa2)

        return (the_lambda*muc*iota*iota*iota / m3_p0) * (m3_p1 + m3_p2 + m3_p3 + m3_p4 + m3_p5 + m3_p6 + m3_p7 + m3_p8)

    def Ph(self, theta):
        the_lambda = theta[1]
        # iota = theta[2]
        alpha = theta[5]
        nu = alpha / theta[6]
        kappa = theta[7]
        phi = theta[8]
        # c = theta[9]
        h = theta[0]
        # muc = 1.0 + kappa / phi

        # std::cout << "mut" << std::endl
        integ2 = integral2(kappa, phi)
        mut = nu / (alpha - 1.0) * (phi*integ2 + (1.0 / phi))
        # std::cout << mut << ' ' << integ2 << std::endl

        # std::cout << "Gp" << std::endl
        Gp = nu*gsl_sf_exp(-1.0*kappa) / (alpha - 1.0)*integral(kappa, phi)
        # std::cout << alpha << ' ' << nu << ' ' << kappa << ' ' << phi << ' ' << mut << ' ' << Gp << std::endl

        # std::cout << "ph" << std::endl
        ph_tmp = (-1.0*the_lambda)*(h + mut) + the_lambda*Gp*(phi + kappa *
                                                              math.pow((nu / (nu + (kappa + phi)*h)), (alpha - 1.0))) / (phi + kappa)
        ph = 0.0

        if ph_tmp < -708396418532264:
            ph = 10000000.0
        elif ph_tmp > 887228390520684:
            ph = 10000000.0
        else:
            ph = math.exp(ph_tmp)

        # std::cout << ph << std::endl

        return ph
