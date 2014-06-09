"""
This module is used to analyse the hypercolumn structure of preference
maps. Currently thsi file offers a means to estimate the hypercolumn
distance from the Fourier power spectrum but different types of
analysis (eg. wavelet analysis) may be introduced in future.
"""

__author__ = "Jean-Luc Stevens"

import math
import itertools
import numpy as np
from scipy.optimize import curve_fit

import param
from dataviews.operation  import ViewOperation
from imagen.analysis import fft_power_spectrum

from dataviews import Dimension, Curve, Points, Table, Annotation, Histogram
from dataviews.options import options, StyleOpts

try: # 2.7+
    gamma = math.gamma
except:
    import scipy.special as ss
    gamma = ss.gamma



class PowerSpectrumAnalysis(ViewOperation):
    """
    Estimation of hypercolumn distance in a cyclic preference map from
    the size of the ring in the fourier power spectrum, following the
    methods described in the in the supplementary materials of
    ``Universality in the Evolution of Orientation Columns in the
    Visual Cortex'', Kaschube et al. 2010.

    If supplied with a preference overlayed with pinwheels, the
    pinwheel_density is computed from kmax (the wavenumber of highest
    power) using the equation:

    rho = pinwheel count/(kmax**2)

    This is then used to generate a map quality estimate (with unit
    range) based on the pi-pinwheel density criterion.
    """


    init_fit = param.Dict(default=None, allow_None=True, doc="""
       If set to None, an initial fit is automatically selected for
       the curve fitting procedure. Otherwise, this is a dictionary of
       the initial coefficients for equation (7) from the 2010 Science
       paper mentioned above (supplementary materials). For instance,
       the values used in the GCAL (Stevens et al. 2013):

       init_fit = dict(a0=0.35, a1=3.8, a2=1.3, a3=0.15, a4=-0.003, a5=0)

       These coefficients may be understood as follows:

        a0 => Gaussian height.
        a1 => Peak x-axis position.
        a2 => Gaussian spread (ie. variance).
        a3 => Baseline value (without falloff).
        a4 => Linear falloff.
        a5 => Quadratic falloff.
       """)

    averaging_fn = param.Callable(default=np.mean, doc="""
      The averaging function used to collapse the power spectrum at each
      wavenumber down to a scalar value. By default, finds the mean
      power for each wavenumber.""")

    fit_table = param.Boolean(default=False, doc="""
      Whether or not to add table listing the fit coefficients at the
      end of the output layout.""")

    gamma_k= param.Number(default=1.8, doc="""
      The degree to which the gamma kernel is heavily tailed when
      squashing the pinwheel density into a unit map metric.""")

    label = param.String(None, allow_None=True, precedence=-1, constant=True,
     doc="""Label suffixes are fixed as there are too many labels to specify.""")


    def _process(self, view):

        [pref] = self.get_views(view, 'Preference')
        pinwheel_views = self.get_views(view, 'Pinwheels', Points)

        pinwheel_count = sum([pw_view.data.shape[0] for pw_view in pinwheel_views], 0)

        xlabel, ylabel = Dimension('Wavenumber', unit='k'), 'FFT Power'
        (l, b, r, t) = pref.bounds.lbrt()
        (dim1, dim2) = pref.data.shape
        xdensity = dim1 / abs(r-l)
        ydensity = dim2 / abs(t-b)

        if xdensity != ydensity:
            raise Exception("SheetView must have matching x- and y-density")

        self._density = xdensity

        power_spectrum = fft_power_spectrum(pref).data
        (amplitudes, edges), fit, info = self.estimate_hypercolumn_distance(power_spectrum)

        kmax = info['kmax']
        if pinwheel_views != []:
            info['rho'] = pinwheel_count / (kmax ** 2)
            info['rho_metric'] = self.gamma_metric(info['rho'])

        info_table = Table(info, label='Hypercolumn Analysis')

        if fit is not None:
            samples = self.fit_samples(dim1/2, 100, fit)
        else:
            samples = zip([0, dim1/2], [0.0, 0.0])

        curve = Curve(samples, dimensions=[xlabel], label=ylabel, value=ylabel,
                      title='{label} Histogram Fit')
        hist = Histogram(amplitudes, edges, dimensions=[xlabel],
                         label=ylabel, value=ylabel, title='FFT Histogram')
        annotation = Annotation(vlines=[kmax], label='KMax', title='{label} Line')

        views = [hist * curve * annotation, info_table]
        if self.p.fit_table and fit is None:
            fit = dict(('a%i' % i, '-') for i in range(6))

        if self.p.fit_table:
            fit_table = Table(fit, label='Hypercolumn Analysis Fit')
            views.append(fit_table)
        return views


    def gamma_dist(self, x, k, theta):
        "The gamma distribution used for the gamma metric"
        return (1.0/theta**k)*(1.0/gamma(k)) * x**(k-1) * np.exp(-(x/theta))


    def gamma_metric(self, pwd):
        """
        The heavily-tailed gamma kernel used to squash the pinwheel
        density into unit range. The maximum value of unity is
        attained when the input pinwheel density is pi.
        """
        theta = math.pi / (self.p.gamma_k -1) # Mode: (k - 1)* theta
        norm = self.gamma_dist(math.pi, self.p.gamma_k, theta)
        return (1.0/norm)*self.gamma_dist(pwd, self.p.gamma_k, theta)


    def wavenumber_spectrum(self, spectrum):
        """
        Bins the power values in the 2D FFT power spectrum as a
        function of wavenumber (1D). Requires square FFT spectra with
        an odd dimension to work to ensure there is a central sample
        corresponding to the DC component (wavenumber zero).
        """
        dim, _dim = spectrum.shape
        assert dim == _dim, "This approach only supports square FFT spectra"
        if not dim % 2:
            self.warning("Slicing data to nearest odd dimensions for centered FFT.")
            spectrum = spectrum[:None if dim % 2 else -1,
                                :None if _dim % 2 else -1]
            dim, _ = spectrum.shape

        # Invert as power_spectrum returns black (low values) for high amplitude
        spectrum = 1 - spectrum
        pixel_bins = range(0, (dim / 2) + 1)
        lower = -(dim / 2)
        upper = (dim / 2) + 1

        # Grid of coordinates relative to central DC component (0,0)
        x, y = np.mgrid[lower:upper, lower:upper]
        flat_pixel_distances = ((x ** 2 + y ** 2) ** 0.5).flatten()
        flat_spectrum = spectrum.flatten()

        # Indices in pixel_bins to which the distances belong
        bin_allocation = np.digitize(flat_pixel_distances, pixel_bins)
        # The bin allocation zipped with actual fft power values
        spectrum_bins = zip(bin_allocation, flat_spectrum)
        grouped_bins = itertools.groupby(sorted(spectrum_bins), lambda x: x[0])
        hist_values = [([sval for (_, sval) in it], bin)
                       for (bin, it) in grouped_bins]
        (power_values, bin_boundaries) = zip(*hist_values)
        averaged_powers = [self.p.averaging_fn(power) for power in power_values]
        assert len(bin_boundaries) == len(pixel_bins)
        return averaged_powers, pixel_bins


    def KaschubeFit(self, k, a0, a1, a2, a3, a4, a5):
        """
        Fitting function used by Kaschube for finding the hypercolumn
        distance from the Fourier power spectrum. These values should
        match the init_fit defaults of pinwheel_analysis below.
        """
        exponent = - ((k - a1)**2) / (2 * a2**2)
        return a0 * np.exp(exponent) + a3 + a4*k + a5*np.power(k,2)


    def fit_samples(self, max_k, samples, fit):
        "Compute a curve based from the fit coefficients"
        ks = np.linspace(0, max_k, max_k)
        values = [self.KaschubeFit(k, **fit) for k in ks]
        return np.array(zip(ks,values))


    def estimate_hypercolumn_distance(self, power_spectrum):
        """
        Estimating the hypercolumn distance by fitting Equation 7 of
        Kaschube et al. 2010 Equation 7 (supplementary
        material). Returns the analysed values as a dictionary.
        """
        amplitudes, edges = self.wavenumber_spectrum(power_spectrum)
        ks = np.array(range(len(amplitudes)))
        try:
            wavenumber_power = amplitudes[:]
            kmax_argmax = float(np.argmax(wavenumber_power[1:]) + 1)
            baseline = np.mean(wavenumber_power)
            height = wavenumber_power[int(kmax_argmax)] - baseline

            if self.p.init_fit is None:
                init_fit = [height, kmax_argmax, 4.0, baseline, 0, 0]
            else:
                init_fit = self.p.init_fit

            fit_vals, _ = curve_fit(self.KaschubeFit,
                                    ks, np.array(amplitudes),
                                    init_fit, maxfev=10000)
            fit = dict(zip(['a0', 'a1', 'a2', 'a3', 'a4', 'a5'], fit_vals))
            valid_fit = (fit['a1'] > 0)
        except:
            valid_fit = False

        kmax_argmax = np.argmax(amplitudes[1:]) + 1
        kmax = fit['a1'] if valid_fit else float(kmax_argmax)

        # The amplitudes begins with k=0 (DC component), k=1 for one
        # period per map, k=2 for two periods per map etc. The units per
        # hypercolumn is the total number of units across the map divided
        # by kmax. If k <= 1.0, the full map width is reported.
        (dim, _) = power_spectrum.shape
        units_per_hypercolumn = dim if (kmax <= 1.0) else dim / float(kmax)
        cycles = self._density / units_per_hypercolumn

        return ((amplitudes, edges),
                fit if valid_fit else None,
                {'kmax': float(kmax),
                'k_delta': float(kmax - float(kmax_argmax)),
                'units_per_hc': float(units_per_hypercolumn),
                'cycles': float(cycles)})


# Defining styles
options.Power_Curve = StyleOpts(color='r', linewidth=3)
options.KMax_Annotation =    StyleOpts(color='g', linewidth=3)
options.Power_Histogram =     StyleOpts(fc='w', ec='k')
