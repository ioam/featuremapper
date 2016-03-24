import math

import numpy as np
from scipy import special as ss
from scipy.optimize import curve_fit

import param

from holoviews import OrderedDict
from holoviews import Curve, ItemTable, ElementOperation, TreeOperation


#====================================#
# Spatial constant conversion methods
#====================================#

def idog_conv(sc):
    """
    Conversion of iDoG spatial constants to extents.
    """
    return math.sqrt(sc*2)

def fr2sp(fr):
    """
    Convert spatial frequency to spatial constant.
    """
    return (math.sqrt(2)/(2*math.pi*fr))


class TuningCurveAnalysis(ElementOperation):

    feature = param.String()

    def _validate_curve(self, curve):
        if not isinstance(curve, Curve):
            raise Exception('Supplied views need to be curves.')
        elif not self.p.feature in curve.key_dimensions[0].name:
            raise Exception('Analysis requires %s response curves.' % self.feature)



class OrientationContrastAnalysis(TuningCurveAnalysis):

    feature = param.String(default='OrientationSurround')

    def _process(self, curve, key=None):
        self._validate_curve(curve)
        ydata = curve.dimension_values(1)
        n_ors = len(curve)
        r0_index = int(n_ors/2)
        r0 = ydata[r0_index]
        rorth = ydata[0]
        try:
            ocsi = (r0 - rorth) / r0
        except:
            ocsi = np.NaN
        data = OrderedDict([('OCSI', ocsi)])
        return ItemTable(data, group='Orientation Contrast Suppression',
                         label=curve.label)



class FrequencyTuningAnalysis(TuningCurveAnalysis):
    """
    Analyzes frequency-tuning curve to find the preferred frequency, lower and
    upper cutoff frequencies as well as the Q-Factor and bandwidth.
    """

    feature = param.String(default='Frequency')

    def _process(self, curve, key=None):
        self._validate_curve(curve)
        xdata = curve.dimension_values(0)
        ydata = curve.dimension_values(1)

        peak_strength = np.max(ydata)
        peak_idx = np.argmax(ydata)
        peak_freq = xdata[peak_idx]
        cutoff_value = peak_strength * 0.707
        cutoff_diff = ydata - cutoff_value
        lower_cutoff_idx = np.argmin(cutoff_diff[:peak_idx]) if peak_idx else 0
        upper_cutoff_idx = peak_idx + np.argmin(cutoff_diff[peak_idx:])
        lower_cutoff = xdata[lower_cutoff_idx]
        upper_cutoff = xdata[upper_cutoff_idx]
        qfactor = peak_freq / (upper_cutoff - lower_cutoff) if peak_idx else 0

        table_data = {'Peak': peak_freq, 'Lower': lower_cutoff,
                      'Upper': upper_cutoff, 'QFactor': qfactor,
                      'Bandwidth': upper_cutoff - lower_cutoff}
        return ItemTable(OrderedDict(table_data), label='Frequency Tuning Analysis')


class SizeTuningPeaks(TuningCurveAnalysis):
    """
    Analysis size-tuning curve to find peak facilitation, peak suppression
    and peak counter-suppression values, which can be used to derive metrics
    like contrast dependent size tuning shifts and counter suppression
    indices.
    """

    feature = param.String(default='Size')

    def _process(self, curve, key=None):
        self._validate_curve(curve)
        xdata = curve.dimension_values(0)
        ydata = curve.dimension_values(1)

        peak_idx = np.argmax(ydata)
        min_idx = np.argmin(ydata[peak_idx:]) + peak_idx
        counter_idx = np.argmax(ydata[min_idx:]) + min_idx

        max_response = np.max(ydata)
        peak_size = xdata[peak_idx]
        r_max = ydata[peak_idx]
        suppression_size = xdata[min_idx]
        r_min = ydata[min_idx]
        counter_size = xdata[counter_idx]
        r_cs = ydata[counter_idx]

        table_data = OrderedDict({'Peak Size': peak_size, 'Suppression Size': suppression_size,
                                  'CS Size': counter_size, 'Max Response': max_response})
        if not r_max == 0:
            table_data['SI'] = (r_max-r_min)/r_max
            table_data['CSI'] = (r_cs-r_min)/r_max
        else:
            table_data['SI'] = 0
            table_data['CSI'] = 0
        return ItemTable(table_data, label='Size Tuning Analysis')



class SizeTuningShift(ElementOperation):
    """
    Takes an overlay of two curves as input and computes the contrast-dependent
    size tuning shift. Assumes the first curve is low contrast and the second
    high contrast.
    """

    def _process(self, overlay, key=None):
        low_contrast = overlay.values()[0]
        high_contrast = overlay.last

        low_table = SizeTuningPeaks(low_contrast)
        high_table = SizeTuningPeaks(high_contrast)

        try:
            shift = low_table['Peak Size'] / high_table['Peak Size']
        except:
            shift = np.NaN
        data = OrderedDict([('CSS', shift), ('Low', low_table['Peak Size']),
                            ('High', high_table['Peak Size'])])
        return ItemTable(data, group='Contrast Dependent Size Tuning Shift',
                         label=low_contrast.label)


class DoGModelFit(TuningCurveAnalysis):
    """
    Baseclass to implement basic size tuning curve fitting procedures.
    Subclasses have to implement the _function method with the function
    that is to be fit to the supplied curve.
    """

    K_c = param.Number(default=0, doc="Center excitatory kernel strength.")

    K_s = param.Number(default=0, doc="Surround inhibitory kernel strength.")

    a = param.Number(default=0, doc="Center excitatory space constant.")

    b = param.Number(default=0, doc="Surround inhibitory space constant.")

    max_iterations = param.Number(default=100000, doc="""
       Number of iterations to optimize the fit.""")

    fit_labels = ['K_c', 'K_s', 'a', 'b']

    feature = param.String(default='Size')

    def _function(self):
        raise NotImplementedError

    def _fit_curve(self, curve):
        xdata = curve.dimension_values(0)
        ydata = curve.dimension_values(1)
        init_fit = [self.p.get(l, self.defaults()[l]) for l in self.fit_labels]
        try:
            table = SizeTuningPeaks(curve)

            if self.a == self.p.a:
                init_fit[self.fit_labels.index('a')] = table['Peak Size']/2.
            if self.b == self.p.b:
                init_fit[self.fit_labels.index('b')] = table['Suppression Size']/2.
        except:
            pass

        try:
            fit, pcov = curve_fit(self._function, xdata, ydata,
                                  init_fit, maxfev=self.p.max_iterations)
            fit_data = dict(zip(self.fit_labels, fit))
            K_s = fit[self.fit_labels.index('K_s')]
            b = fit[self.fit_labels.index('b')]
            K_c = fit[self.fit_labels.index('K_c')]
            a = fit[self.fit_labels.index('a')]
            fitted_ydata = self._function(xdata, *fit)

            peak_idx = np.argmax(fitted_ydata)
            min_idx = np.argmin(fitted_ydata[peak_idx:]) + peak_idx
            counter_idx = np.argmax(fitted_ydata[min_idx:]) + min_idx

            max_response = np.max(ydata)
            peak_size = xdata[peak_idx]
            r_max = fitted_ydata[peak_idx]
            suppression_size = xdata[min_idx]
            r_min = fitted_ydata[min_idx]
            counter_size = xdata[counter_idx]
            r_cs = fitted_ydata[counter_idx]

            fit_data['SI'] = (r_max-r_min)/r_max
            fit_data['Peak'] = peak_size

            fitted_curve = Curve(zip(xdata, fitted_ydata), group='Response',
                                 label='Size Tuning Fit', kdims=curve.kdims,
                                 vdims=curve.vdims)(style=dict(color='k', linestyle='-.'))
        except:
            fitted_curve = Curve(zip(xdata, np.zeros(len(xdata))),
                                 kdims=curve.kdims, vdims=curve.vdims)
            fit_data = dict(zip(self.fit_labels, [0]*len(self.fit_labels)))
            fit_data['SI'] = 0
        return [fitted_curve, fit_data]


class Size_iDoGModel(DoGModelFit):
    """
    iDoG model response function to sine grating disk stimulus
    with optimal spatial frequency and varying disk radius (r).
    Ref: Sceniak et al. (2006) - page 3476
    Fitting parameters: R_0 - Steady-state response
                        K_c - Center strength
                        a   - Center spatial constant
                        K_s - Surround Strength
                        b   - Surround spatial constant
    """

    R_0 = param.Number(default=0, doc="Baseline response.")

    label = param.String(default='IDoG Model Fit')

    fit_labels = ['R_0', 'K_c', 'K_s', 'a', 'b']

    feature = param.String(default='Size')

    def _process(self, curve, key=None):
        self._validate_curve(curve)
        fitted_curve, fit_data = self._fit_curve(curve)
        return curve*fitted_curve + ItemTable(OrderedDict(fit_data), label=self.p.label)

    @classmethod
    def _function(cls, d, R_0, K_c, K_s, a, b):
        R_e = K_c * (1-np.exp(-(2*d/a)**2))
        R_i = K_s * (1-np.exp(-(2*d/b)**2))

        return R_0 + R_e - R_i


class Size_DivDoGModel(DoGModelFit):
    """
    iDoG model response function to sine grating disk stimulus
    with optimal spatial frequency and varying disk radius (r).
    Ref: Sceniak et al. (2006) - page 3476
    Fitting parameters: R_0 - Steady-state response
                        K_c - Center strength
                        a   - Center spatial constant
                        K_s - Surround Strength
                        b   - Surround spatial constant
    """

    R_0 = param.Number(default=0, doc="Baseline response.")

    label = param.String(default='IDoG Model Fit')

    fit_labels = ['R_0', 'K_c', 'K_s', 'a', 'b']

    feature = param.String(default='Size')

    def _process(self, curve, key=None):
        self._validate_curve(curve)
        fitted_curve, fit_data = self._fit_curve(curve)
        return curve*fitted_curve + ItemTable(OrderedDict(fit_data), label=self.p.label)

    @classmethod
    def _function(cls, d, R_0, K_c, K_s, a, b):
        if a < 0:
            return 10**8
        R_e = K_c * (1-np.exp(-(2*d/a)**2))
        R_i = K_s * (1-np.exp(-(2*d/b)**2))
        return R_0 + R_e / (1+R_i)


class SF_DoGModel(DoGModelFit):
    """
    DoG model response function to sine grating disk stimulus
    with varying spatial frequency (f).
    Ref: Sceniak et al. (2006) - page 3476
    Fitting parameters: R_0 - Steady-state response
                        K_c - Center strength
                        a   - Center spatial constant
                        K_s - Surround Strength
                        b   - Surround spatial constant
    """

    R_0 = param.Number(default=0, doc="Baseline response.")

    label = param.String(default='DoG Model Fit')

    fit_labels = ['R_0', 'K_c', 'K_s', 'a', 'b']

    feature = param.String(default='Frequency')

    def _process(self, curve, key=None):
        if 'Contrast' in key:
            self.p.default_contrast = key['Contrast']
        self._validate_curve(curve)

        fitted_curve, fit_data = self._fit_curve(curve)

        return [curve*fitted_curve, ItemTable(fit_data, label=self.p.label)]


    def _function(self, f, R_0, K_c, K_s, a, b):
        # Fitting penalties for negative coefficients
        if (a <= 0) or (b <= 0) or (K_c <= 0) or (K_s <= 0) or (R_0 < 0):
            return 10000

        C = self.p.default_contrast
        if not isinstance(f, float):
            R = np.zeros(len(f))
            for i, fr in enumerate(f):
                R_c = C * K_c * (1.0 - np.exp(-(fr / 2.0 * a) ** 2.0))
                R_s = C * K_s * (1.0 - np.exp(-(fr / 2.0 * b) ** 2.0))
                R[i] = R_0 + R_c - R_s
        else:
            R_c = C * K_c * (1.0 - np.exp(-(f / 2.0 * a) ** 2.0))
            R_s = C * K_s * (1.0 - np.exp(-(f / 2.0 * b) ** 2.0))
            R = R_0 + R_c - R_s

        return R


class iDoG_DeAngelisModel(DoGModelFit):
    """
    Basic integrated difference of Gaussian response function
    for area summation curves.
    Ref: DeAngelis et al. 1994
    Fitting parameters: K_c - Center strength
                        a   - Center spatial constant
                        K_s - Surround Strength
                        b   - Surround spatial constant
                        R_0 - Steady-state response
    """

    R_0 = param.Number(default=0, doc="Baseline response.")

    label = param.String(default='IDoG Model Fit')

    fit_labels = ['R_0', 'K_c', 'K_s', 'a', 'b']

    feature = param.String(default='Size')

    def _function(self, d, R_0, K_c, K_s, a, b):
        if (a <= 0) or (b <= 0) or (K_c <= 0) or (K_s <= 0) or (
            R_0 < 0): return 10000

        r = d / 2.0
        R_c = 0.5 * a * math.sqrt(math.pi) * ss.erf(r / a)
        R_s = 0.5 * b * math.sqrt(math.pi) * ss.erf(r / b)

        return R_0 + (K_c * R_c) - (K_s * R_s)


    def _process(self, curve, key=None):
        self._validate_curve(curve)
        fitted_curve, fit_data = self._fit_curve(curve)
        return [curve*fitted_curve, ItemTable(fit_data, label=self.p.label)]



class NormalizationDoGModel(DoGModelFit):
    """
    Normalization model describing response of V1 neurons
    to sine grating disk stimuli of varying sizes.
    Ref: Sceniak et al. (200q1) - page 1875
    Fitting parameters: K_c -  Center strength
                        a   -  Center spatial constant
                        K_s -  Surround Strength
                        b   -  Surround spatial constant
                        beta - Arbitrary exponent
    """

    beta = param.Number(default=0, doc="Baseline response.")

    default_contrast = param.Number(default=1.0, doc="""
        Default contrast to use if supplied curve doesn't provide contrast.""")

    label = param.String(default='Normalization DoG Model Fit')

    fit_labels = ['beta', 'K_c', 'K_s', 'a', 'b']

    feature = param.String(default='Size')

    def _function(self, d, beta, K_c, K_s, a, b):
        # Fitting penalty
        if (a <= 0) or (b <= 0) or (b <= a) or (K_c <= 0) or (K_s <= 0):
            return 10000

        C = self.p.default_contrast
        r = d/2.0
        L_c = 0.5 * a * math.sqrt(math.pi) * ss.erf(2 * r / a)
        L_s = 0.5 * b * math.sqrt(math.pi) * ss.erf(2 * r / b)
        R = ((C * K_c * L_c) / (1 + C * K_s * L_s)) ** beta
        return R


    def _process(self, curve, key=None):
        self._validate_curve(curve)
        fitted_curve, fit_data = self._fit_curve(curve)
        return [curve*fitted_curve, ItemTable(fit_data, label=self.p.label)]
