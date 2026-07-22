import numpy as np
from scipy.interpolate import interp1d


class laser_field:
    def __init__(
        self,
        wavelength,
        intensity,
        ncyc,
        CEP=0.0,
        fwhm_cyc=None,
        offset_ratio=2,
        env_type="trapezoidal",
    ):
        self.wavelength = wavelength
        self.intensity = intensity
        self.ncyc = ncyc
        self.CEP = CEP / 180 * np.pi

        self.tofmt = 0.0241889
        self.toene = 1239.84190
        self.toev = 27.2113845
        self.toamp = 35.0944506
        # from Toru Morishita

        self.omega = self.toene / (self.wavelength * self.toev)
        self.amplitude = np.sqrt(intensity / 1e15 / self.toamp)
        self.period = 2 * np.pi / self.omega
        self.env_type = env_type.lower()

        self.pulse_center = 0.0
        self.fwhm_cyc = fwhm_cyc  # intensity FWHM in terms of T
        self.offset_ratio = (
            offset_ratio  # pulse center: t_center = offset_ratio * sigma
        )
        if env_type == "gauss2":
            sigma_intensity = self.fwhm_cyc * self.period / (np.sqrt(8 * np.log(2)))
            self.sigma = np.sqrt(2) * sigma_intensity
            self.pulse_center = self.offset_ratio * self.sigma

        # print(self.omega)
        # print(self.amplitude)

    def electric_field(self, time, delay=0.0):
        t0 = time - delay
        envelop = self.get_envelop(t0)
        return (
            envelop
            * self.amplitude
            * np.sin(self.omega * (t0 - self.pulse_center) + self.CEP)
        )

    def get_envelop(self, time):
        if self.env_type == "sin2":
            very_small = 1e-15
            if time < 0.0 - very_small:
                envelop = 0.0
            elif np.abs(time) < self.period * self.ncyc:
                envelop = np.sin(np.pi * time / (self.ncyc * self.period)) ** 2
            else:
                envelop = 0.0
        elif self.env_type == "cos2":
            if np.abs(time) < self.period * self.ncyc * 0.5:
                envelop = np.cos(np.pi * time / (self.ncyc * self.period)) ** 2
            else:
                envelop = 0.0
        elif self.env_type == "sin2_const":
            if time < 0.0:
                envelop = np.cos(np.pi * time / (self.ncyc * self.period)) ** 2
            else:
                envelop = 1.0
        elif self.env_type == "gauss":
            FWHM = self.ncyc * self.period * 0.25
            scale = FWHM**2 / (4 * np.log(2))
            t_center = 0.5 * self.ncyc * self.period
            envelop = np.exp(-((time - t_center) ** 2) / scale)
        elif self.env_type == "gauss2":
            envelop = np.exp(-((time - self.pulse_center) ** 2) / (2 * self.sigma**2))
        elif self.env_type == "trapezoidal":
            if time < 0 or time > self.ncyc * self.period:
                envelop = 0.0
            elif time < 0.5 * self.period:
                envelop = time / (0.5 * self.period)
            elif time < (self.ncyc - 0.5) * self.period:
                envelop = 1.0
            else:
                envelop = -(time - self.ncyc * self.period) / (0.5 * self.period)
        else:
            envelop = 1.0
        return envelop

    def _generate_A(self, t_list, delay=0.0):
        Agrad = lambda t, A: -self.electric_field(t, delay)

        dt0 = t_list[1] - t_list[0]
        small = 2 * np.pi / (self.toene / (800 * self.toev)) / 100000

        numdt = 1
        while dt0 / numdt > small:
            numdt *= 10

        self.numdt_calib = numdt
        self.dt_calib = dt0 / numdt
        self.t_calib = np.linspace(t_list[0], t_list[-1], (len(t_list) - 1) * numdt + 1)

        self.A_calib = np.zeros_like(self.t_calib)
        for istep in range(1, len(self.t_calib)):
            t1 = self.t_calib[istep - 1]
            k1 = -self.electric_field(t1, delay)
            k2 = -self.electric_field(t1 + 0.5 * self.dt_calib, delay)
            k3 = -self.electric_field(t1 + self.dt_calib, delay)
            self.A_calib[istep] = self.A_calib[istep - 1] + self.dt_calib / 6 * (
                k1 + 4 * k2 + k3
            )
        self.vector_potential = interp1d(self.t_calib, self.A_calib)
