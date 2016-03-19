"""
User-level measurement commands, typically for measuring or generating
SheetViews.

The file defines base classes for defining measurements and a large
set of predefined measurement commands.

Some of the commands are ordinary Python functions, but the rest are
ParameterizedFunctions, which act like Python functions but support
Parameters with defaults, bounds, inheritance, etc.  These commands
are usually grouped together using inheritance so that they share a
set of parameters and some code, and only the bits that are specific
to that particular plot or analysis appear below.  See the
superclasses for the rest of the parameters and code.
"""

import copy
import ImageDraw
import Image as PILImage

import numpy as np

import param
from param import ParameterizedFunction, ParamOverrides

from holoviews import HoloMap, Layout
from holoviews.interface.collector import AttrDict, Collector
from holoviews.element.raster import Image

import imagen
from imagen import random
from imagen import SineGrating, Gaussian, RawRectangle, Disk, Composite, \
    OrientationContrast
from imagen.deprecated import GaussiansCorner

from featuremapper import FeatureResponses, FeatureMaps, FeatureCurves, \
    ReverseCorrelation
import features as f
from metaparams import *
from distribution import DSF_MaxValue, DistributionStatisticFn, \
    DSF_WeightedAverage, DSF_BimodalPeaks


class PatternPresentingCommand(ParameterizedFunction):
    """Parameterized command for presenting input patterns"""

    durations = param.Parameter(default=[1.0], doc="""Times after presentation
        begins at which to record a measurement.""")

    measurement_prefix = param.String(default='', doc="""
        Optional prefix to add to the name under which results are stored as
        part of a measurement response.""")

    __abstract = True



class MeasureResponseCommand(PatternPresentingCommand):
    """Parameterized command for presenting input patterns and measuring
    responses."""

    inputs = param.List(default=[], doc="""Name of input supplied to the
        metadata_fns to filter out desired input.""")

    metafeature_fns = param.HookList(default=[contrast2scale], doc="""
        Metafeature_fns is a hooklist, which accepts any function, which applies
        coordinated changes to a set of inputs based on some parameter or feature
        value. Can be used to present different patterns to different inputs or
        to control complex features like contrast.""")

    offset = param.Number(default=0.0, softbounds=(-1.0, 1.0), doc="""
        Additive offset to input pattern.""")

    outputs = param.List(default=[], doc="""Name of output sources supplied
        to metadata_fns to filter out desired output.""")

    pattern_generator = param.Callable(default=None, instantiate=True, doc="""
        Callable object that will generate input patterns coordinated
        using a list of meta parameters.""")

    pattern_response_fn = param.Callable(default=None, instantiate=False, doc="""
        Callable object that will present a parameter-controlled pattern to a
        set of Sheets.  Needs to be supplied by a subclass or in the call.
        The attributes duration and apply_output_fns (if non-None) will
        be set on this object, and it should respect those if possible.""")

    preference_fn = param.ClassSelector(DistributionStatisticFn,
                                        default=DSF_MaxValue(), doc="""
        Function that will be used to analyze the distributions of unit
        responses.""")

    preference_lookup_fn = param.Callable(default=None, doc="""
        Callable object that will look up a preferred feature values.""",
                                          instantiate=True)

    scale = param.Number(default=1.0, softbounds=(0.0, 2.0), doc="""
        Multiplicative strength of input pattern.""")

    static_parameters = param.List(default=["scale", "offset"], doc="""
        List of names of parameters of this class to pass to the
        pattern_presenter as static parameters, i.e. values that will be fixed
        to a single value during measurement.""", class_=str)

    subplot = param.String(default='', doc="""
        Name of map to register as a subplot, if any.""")

    __abstract = True


    def __call__(self, **params):
        """Measure the response to the specified pattern and store the data
        in each sheet."""
        p = ParamOverrides(self, params, allow_extra_keywords=True)
        self._set_presenter_overrides(p)
        static_params = dict([(s, p[s]) for s in p.static_parameters])

        results = FeatureMaps(self._feature_list(p), durations=p.durations,
                              inputs=p.inputs,
                              metafeature_fns=p.metafeature_fns,
                              measurement_prefix=p.measurement_prefix,
                              outputs=p.outputs, static_features=static_params,
                              pattern_generator=p.pattern_generator,
                              pattern_response_fn=p.pattern_response_fn)

        self._restore_presenter_defaults()

        return results


    def _feature_list(self, p):
        """Return the list of features to vary; must be implemented by each
        subclass."""
        raise NotImplementedError


    def _set_presenter_overrides(self, p):
        """
        Overrides parameters of the pattern_response_fn and
        pattern_coordinator, using extra_keywords passed into the
        MeasurementResponseCommand and saves the default parameters to restore
        after measurement is complete.
        """
        self._pr_fn_params = p.pattern_response_fn.get_param_values()
        for override, value in p.extra_keywords().items():
            if override in p.pattern_response_fn.params():
                p.pattern_response_fn.set_param(override, value)


    def _restore_presenter_defaults(self):
        """
        Restores the default pattern_response_fn parameters after the
        measurement.
        """
        params = self.pattern_response_fn.params()
        for key, value in self._pr_fn_params:
            if not params[key].constant:
                self.pattern_response_fn.set_param(key, value)



class SinusoidalMeasureResponseCommand(MeasureResponseCommand):
    """
    Parameterized command for presenting sine gratings and measuring
    responses.
    """

    pattern_generator = param.Callable(default=SineGrating(), doc="""
        Pattern to be presented on the inputs.""", instantiate=True)

    frequencies = param.List(class_=float, default=[2.4], doc="""
        Sine grating frequencies to test.""")

    num_phase = param.Integer(default=18, bounds=(1, None), softbounds=(1, 48),
                              doc="Number of phases to test.")

    num_orientation = param.Integer(default=4, bounds=(1, None),
                                    softbounds=(1, 24),
                                    doc="Number of orientations to test.")

    scale = param.Number(default=0.3)

    preference_fn = param.ClassSelector(DistributionStatisticFn,
                                        default=DSF_WeightedAverage(), doc="""
        Function that will be used to analyze the distributions of unit
        responses.""")

    __abstract = True



class PositionMeasurementCommand(MeasureResponseCommand):
    """
    Parameterized command for measuring topographic position.
    """

    divisions = param.Integer(default=7, bounds=(1, None), doc="""
        The number of different positions to measure in X and in Y.""")

    x_range = param.NumericTuple((-0.5, 0.5), doc="""
        The range of X values to test.""")

    y_range = param.NumericTuple((-0.5, 0.5), doc="""
        The range of Y values to test.""")

    size = param.Number(default=0.5, bounds=(0, None), doc="""
        The size of the pattern to present.""")

    pattern_generator = param.Callable(
        default=Gaussian(aspect_ratio=1.0), doc="""
        Callable object that will present a parameter-controlled
        pattern to a set of Sheets.  For measuring position, the
        pattern_presenter should be spatially localized, yet also able
        to activate the appropriate neurons reliably.""")

    static_parameters = param.List(default=["scale", "offset", "size"])

    __abstract = True



class SingleInputResponseCommand(MeasureResponseCommand):
    """
    A callable Parameterized command for measuring the response to
    input on a specified Sheet.

    Note that at present the input is actually presented to all input
    sheets; the specified Sheet is simply used to determine various
    parameters.  In the future, it may be modified to draw the pattern
    on one input sheet only.
    """

    scale = param.Number(default=30.0)

    offset = param.Number(default=0.5)

    pattern_generator = param.Callable(default=RawRectangle(size=0.1,
                                                            aspect_ratio=1.0))

    static_parameters = param.List(default=["scale", "offset", "size"])

    __abstract = True



class FeatureCurveCommand(SinusoidalMeasureResponseCommand):
    """A callable Parameterized command for measuring tuning curves."""

    contrasts = param.List(default=[30, 60, 80, 90])

    num_orientation = param.Integer(default=12)

    # Make constant in subclasses?
    x_axis = param.String(default='orientation', doc="""
        Parameter to use for the x axis of tuning curves.""")

    static_parameters = param.List(default=[])

    __abstract = True


    def __call__(self, **params):
        """Measure the response to the specified pattern and store the data
        in each sheet."""
        p = ParamOverrides(self, params, allow_extra_keywords=True)
        self._set_presenter_overrides(p)
        results = self._compute_curves(p)
        self._restore_presenter_defaults()
        return results


    def _compute_curves(self, p):
        """
        Compute a set of curves for the specified sheet, using the
        specified val_format to print a label for each value of a
        curve_parameter.
        """
        static_params = dict([(s, p[s]) for s in p.static_parameters])

        return FeatureCurves(self._feature_list(p), durations=p.durations,
                             inputs=p.inputs, measurement_prefix=p.measurement_prefix,
                             metafeature_fns=p.metafeature_fns, outputs=p.outputs,
                             pattern_generator=p.pattern_generator,
                             pattern_response_fn=p.pattern_response_fn,
                             static_features=static_params, x_axis=p.x_axis)



    def _feature_list(self, p):
        return [f.Orientation(steps=p.num_orientation),
                f.Phase(steps=p.num_phase),
                f.Frequency(values=p.frequencies),
                f.Contrast(values=p.contrasts, preference_fn=None)]



class UnitCurveCommand(FeatureCurveCommand):
    """
    Measures tuning curve(s) of particular unit(s).
    """

    pattern_generator = param.Callable(
        default=SineGrating(mask_shape=Disk(smoothing=0.0, size=1.0)))

    metafeature_fns = param.HookList(
        default=[contrast2scale.instance(contrast_parameter='weber_contrast')])

    size = param.Number(default=0.5, bounds=(0, None), doc="""
        The size of the pattern to present.""")

    coords = param.List(default=[(0, 0)], doc="""
        List of coordinates of units to measure.""")

    __abstract = True


    def _populate_grid(self, results):
        trees = []
        for coord, viewgroup in results.items():
            for path, stack in viewgroup.data.items():
                grid_container = Layout()
                coord_map = stack.add_dimension(f.Y, 0, coord[1])
                coord_map = coord_map.add_dimension(f.X, 0, coord[0])
                grid_container.set_path(path, coord_map)
                trees.append(grid_container)
        return Layout.merge(trees)



class measure_response(FeatureResponses):

    input_patterns = param.Dict(default={}, doc="""
        Assigns patterns to different inputs overriding the
        pattern_generator parameter. If all inputs have not been
        assigned a pattern, remaining inputs will be presented a
        blank pattern.""")

    pattern_generator = param.Callable(default=Gaussian(), instantiate=True,
                                       doc="""
        Callable object that will generate input patterns coordinated
        using a list of meta parameters.""")


    def __call__(self, **params):
        p = ParamOverrides(self, params, allow_extra_keywords=True)
        self._apply_cmd_overrides(p)
        self.metadata = AttrDict(p.metadata)
        for fn in p.metadata_fns:
            self.metadata.update(fn(p.inputs, p.outputs))

        output_names = self.metadata.outputs.keys()
        input_names = self.metadata.inputs.keys()
        inputs = dict.fromkeys(input_names)
        if p.input_patterns:
            for k, ip in p.input_patterns.items():
                inputs[k] = ip
            for name in [k for k, ip in inputs.items() if ip is None]:
                self.warning("No pattern specified for input %s, defaulting"
                             "to blank Constant pattern." % name)
                inputs[name] = imagen.Constant(scale=0)
        else:
            for k in inputs.keys():
                inputs[k] = copy.deepcopy(p.pattern_generator)


        for f in p.pre_presentation_hooks: f()

        responses = p.pattern_response_fn(inputs, output_names,
                                          durations=p.durations)

        for f in p.post_presentation_hooks: f()

        label = inputs.values()[0].__class__.__name__
        results = self._collate_results(responses, label)

        if p.measurement_storage_hook:
            p.measurement_storage_hook(results)

        return results


    def _collate_results(self, responses, label):
        time = self.metadata.timestamp
        dims = [f.Time, f.Duration]

        response_label = label + ' Response'
        results = Layout()
        for label, response in responses.items():
            name, duration = label
            path = (response_label.replace(' ', ''), name)
            label = ' '.join([name, response_label])
            metadata = self.metadata['outputs'][name]
            if path not in results:
                vmap = HoloMap(key_dimensions=dims)
                vmap.metadata = AttrDict(**metadata)
                results.set_path(path, vmap)

            im = Image(response, metadata['bounds'], label=label, group='Activity')
            im.metadata=AttrDict(timestamp=time)
            results[path][(time, duration)] = im
        return results


    def _apply_cmd_overrides(self, p):
        super(measure_response, self)._apply_cmd_overrides(p)
        for override, value in p.extra_keywords().items():
            if override in p.pattern_response_fn.params():
                p.pattern_response_fn.set_param(override, value)
            else:
                self.warning('%s not a parameter of measure_response '
                             'or the pattern_response_fn.' % override)



class measure_rfs(SingleInputResponseCommand):
    """
    Map receptive fields by reverse correlation.

    Presents a large collection of input patterns, typically white
    noise, keeping track of which units in the specified input_sheet
    were active when each unit in other Sheets in the simulation was
    active.  This data can then be used to plot receptive fields for
    each unit.  Note that the results are true receptive fields, not
    the connection fields usually presented in lieu of receptive
    fields, because they take all circuitry in between the input and
    the target unit into account.

    Note also that it is crucial to set the scale parameter properly
    when using units with a hard activation threshold (as opposed to a
    smooth sigmoid), because the input pattern used here may not be a
    very effective way to drive the unit to activate.  The value
    should be set high enough that the target units activate at least
    some of the time there is a pattern on the input.
    """

    static_parameters = param.List(default=["scale", "offset"])

    pattern_generator = param.Callable(default=random.UniformRandom(name='UniformNoise',
                                                                    time_dependent=False),
       doc="""Presented pattern for reverse correlation, usually white noise.""")

    presentations = param.Number(default=100, doc="""
       Number of presentations to run the reverse correlation for.""")

    roi = param.NumericTuple(default=(0, 0, 0, 0), doc="""
       If non-zero ROI bounds is specified only the RFs in that
       subregion are recorded.""")

    store_responses = param.Boolean(default=False, doc="""
       Whether to record the responses to individual patterns""")

    __abstract = True


    def __call__(self, **params):
        p = ParamOverrides(self, params, allow_extra_keywords=True)
        self._set_presenter_overrides(p)
        static_params = dict([(s, p[s]) for s in p.static_parameters])
        results = ReverseCorrelation(self._feature_list(p),
                                     durations=p.durations, inputs=p.inputs,
                                     outputs=p.outputs, roi=p.roi,
                                     static_features=static_params,
                                     pattern_response_fn=p.pattern_response_fn,
                                     pattern_generator=p.pattern_generator,
                                     store_responses=p.store_responses)
        self._restore_presenter_defaults()

        return results


    def _feature_list(self, p):
        return [f.Presentation(range=(0, p.presentations-1), steps=p.presentations)]



# Helper function for measuring direction maps
def compute_orientation_from_direction(current_values):
    """
    Return the orientation corresponding to the given direction.

    Wraps the value to be in the range [0,pi), and rounds it slightly
    so that wrapped values are precisely the same (to avoid biases
    caused by vector averaging with keep_peak=True).

    Note that in very rare cases (1 in 10^-13?), rounding could lead
    to different values for a wrapped quantity, and thus give a
    heavily biased orientation map.  In that case, just choose a
    different number of directions to test, to avoid that floating
    point boundary.
    """
    return np.round(((dict(current_values)['direction']) + (np.pi / 2)) % np.pi,
                    13)



class measure_sine_pref(SinusoidalMeasureResponseCommand):
    """
    Measure preferences for sine gratings in various combinations.
    Can measure orientation, spatial frequency, spatial phase,
    ocular dominance, horizontal phase disparity, color hue, motion
    direction, and speed of motion.

    In practice, this command is useful for any subset of the possible
    combinations, but if all combinations are included, the number of
    input patterns quickly grows quite large, much larger than the
    typical number of patterns required for an entire simulation.
    Thus typically this command will be used for the subset of
    dimensions that need to be evaluated together, while simpler
    special-purpose routines are provided below for other dimensions
    (such as hue and disparity).
    """

    max_speed = param.Number(default=2.0/24.0, bounds=(0, None), doc="""
        The maximum speed to measure (with zero always the minimum).""")

    metafeature_fns = param.HookList(default=[contrast2scale, direction2translation])

    num_ocularity = param.Integer(default=1, bounds=(1, None), softbounds=(1, 3), doc="""
        Number of ocularity values to test; set to 1 to disable or 2 to enable.""")

    num_disparity = param.Integer(default=1, bounds=(1, None), softbounds=(1, 48), doc="""
        Number of disparity values to test; set to 1 to disable or e.g. 12 to enable.""")

    num_hue = param.Integer(default=1, bounds=(1, None), softbounds=(1, 48), doc="""
        Number of hues to test; set to 1 to disable or e.g. 8 to enable.""")

    num_direction = param.Integer(default=0, bounds=(0, None),
                                  softbounds=(0, 48), doc="""
        Number of directions to test.  If nonzero, overrides num_orientation,
        because the orientation is calculated to be perpendicular to the direction.""")

    num_speeds = param.Integer(default=4, bounds=(0, None), softbounds=(0, 10),
                               doc="""
        Number of speeds to test (where zero means only static patterns).
        Ignored when num_direction=0.""")

    subplot = param.String("Orientation")


    def _feature_list(self, p):
        # Always varies frequency and phase; everything else depends on parameters.

        features = [f.Frequency(values=p.frequencies)]

        if p.num_direction == 0: 
            features += [f.Orientation(steps=p.num_orientation,
                                       preference_fn=self.preference_fn)]

        features += [f.Phase(steps=p.num_phase)]

        if p.num_ocularity > 1: features += \
            [f.Ocular(range=(0.0, 1.0), steps=p.num_ocularity)]

        if p.num_disparity > 1: features += \
            [f.PhaseDisparity(steps=p.num_disparity)]

        if p.num_hue > 1: features += \
            [f.Hue(steps=p.num_hue)]

        if p.num_direction > 0 and p.num_speeds == 0: features += \
            [f.Speed(values=[0])]

        if p.num_direction > 0 and p.num_speeds > 0: features += \
            [f.Speed(range=(0.0, p.max_speed), steps=p.num_speeds)]

        if p.num_direction > 0:
            # Compute orientation from direction
            dr = f.Direction(range=(0.0, 2*np.pi), steps=p.num_direction)
            or_values = list(set(
                [compute_orientation_from_direction([("direction", v)]) for v in
                 dr.values]))
            features += [dr, \
                         f.Orientation(values=or_values,
                                       compute_fn=compute_orientation_from_direction,
                                       preference_fn=self.preference_fn)]

        return features



class measure_or_pref(SinusoidalMeasureResponseCommand):
    """Measure an orientation preference map by collating the response to patterns."""

    subplot = param.String("Orientation")

    preference_fn = param.ClassSelector(DistributionStatisticFn,
                                        default=DSF_WeightedAverage(), doc="""
        Function that will be used to analyze the distributions of unit
        responses.""")


    def _feature_list(self, p):
        return [f.Frequency(values=p.frequencies),
                f.Orientation(steps=p.num_orientation,
                              preference_fn=self.preference_fn),
                f.Phase(steps=p.num_phase)]



class measure_od_pref(SinusoidalMeasureResponseCommand):
    """
    Measure an ocular dominance preference map by collating the response to patterns.
    """

    metafeature_fns = param.HookList(
        default=[contrast2scale, ocular2leftrightscale])


    def _feature_list(self, p):
        return [f.Frequency(values=p.frequencies),
                f.Orientation(steps=p.num_orientation,
                              preference_fn=self.preference_fn),
                f.Phase(steps=p.num_phase),
                f.Ocular(range=(0.0, 1.0), values=[0.0, 1.0])]



class measure_phasedisparity(SinusoidalMeasureResponseCommand):
    """
    Measure a phase disparity preference map by collating the response to patterns.
    """

    metafeature_fns = param.HookList(default=[contrast2scale,
                                              phasedisparity2leftrightphase])

    num_disparity = param.Integer(default=12, bounds=(1, None),
                                  softbounds=(1, 48),
                                  doc="Number of disparity values to test.")

    orientation = param.Number(default=np.pi / 2, softbounds=(0.0, 2 * np.pi),
                               doc="""
        Orientation of the test pattern; typically vertical to measure
        horizontal disparity.""")

    static_parameters = param.List(default=["orientation", "scale", "offset"])


    def _feature_list(self, p):
        return [f.Frequency(values=p.frequencies),
                f.Phase(steps=p.num_phase),
                f.PhaseDisparity(steps=p.num_disparity)]


class measure_dr_pref(SinusoidalMeasureResponseCommand):
    """Measure a direction preference map by collating the response to patterns."""

    metafeature_fns = param.HookList(default=[contrast2scale,
                                              direction2translation])

    num_phase = param.Integer(default=12)

    num_direction = param.Integer(default=6, bounds=(1, None),
                                  softbounds=(1, 48),
                                  doc="Number of directions to test.")

    num_speeds = param.Integer(default=4, bounds=(0, None), softbounds=(0, 10),
                               doc="""
        Number of speeds to test (where zero means only static patterns).""")

    max_speed = param.Number(default=2.0 / 24.0, bounds=(0, None), doc="""
        The maximum speed to measure (with zero always the minimum).""")

    subplot = param.String("Direction")

    preference_fn = param.ClassSelector(DistributionStatisticFn,
                                        default=DSF_WeightedAverage(), doc="""
        Function that will be used to analyze the distributions of
        unit responses. Sets value_scale to normalize direction
        preference values.""")


    def _feature_list(self, p):
        # orientation is computed from direction
        dr = f.Direction(steps=p.num_direction)
        or_values = list(set(
            [compute_orientation_from_direction([("direction", v)]) for v in
             dr.values]))

        return [f.Speed(values=[0]) if p.num_speeds is 0 else
                f.Speed(range=(0.0, p.max_speed), steps=p.num_speeds),
                f.Duration(values=[np.max(p.durations)]),
                f.Frequency(values=p.frequencies),
                f.Direction(steps=p.num_direction,
                            preference_fn=self.preference_fn),
                f.Phase(steps=p.num_phase),
                f.Orientation(values=or_values,
                              compute_fn=compute_orientation_from_direction)]


class measure_hue_pref(SinusoidalMeasureResponseCommand):
    """Measure a hue preference map by collating the response to patterns."""

    metafeature_fns = param.HookList(default=[contrast2scale,
                                              hue2rgbscale])

    num_phase = param.Integer(default=12)

    num_hue = param.Integer(default=8, bounds=(1, None), softbounds=(1, 48),
                            doc="Number of hues to test.")

    subplot = param.String("Hue")

    # For backwards compatibility; not sure why it needs to differ from the default
    static_parameters = param.List(default=[])


    def _feature_list(self, p):
        return [f.Frequency(values=p.frequencies),
                f.Orientation(steps=p.num_orientation),
                f.Hue(steps=p.num_hue),
                f.Phase(steps=p.num_phase)]



class measure_second_or_pref(SinusoidalMeasureResponseCommand):
    """Measure the secondary  orientation preference maps."""

    num_orientation = param.Integer(default=16, bounds=(1, None),
                                    softbounds=(1, 64),
                                    doc="Number of orientations to test.")

    true_peak = param.Boolean(default=True, doc="""If set the second
        orientation response is computed on the true second mode of the
	    orientation distribution, otherwise is just the second maximum
	    response""")

    subplot = param.String("Second Orientation")


    def _feature_list(self, p):
        fs = [f.Frequency(values=p.frequencies)]
        if p.true_peak:
            fs.append(f.Orientation(steps=p.num_orientation,
                                    preference_fn=DSF_BimodalPeaks()))
        else:
            fs.append(f.Orientation(steps=p.num_orientation,
                                    preference_fn=DSF_BimodalPeaks()))
            fs.append(f.Phase(steps=p.num_phase))

        return fs


gaussian_corner = Composite(
    operator=np.maximum, generators=[
        Gaussian(size=0.06, orientation=0, aspect_ratio=7, x=0.3),
        Gaussian(size=0.06, orientation=np.pi / 2, aspect_ratio=7, y=0.3)])


class measure_corner_or_pref(PositionMeasurementCommand):
    """Measure a corner preference map by collating the response to patterns."""

    scale = param.Number(default=1.0)

    divisions = param.Integer(default=11)

    pattern_generator = param.Callable(default=gaussian_corner)

    x_range = param.NumericTuple((-1.2, 1.2))

    y_range = param.NumericTuple((-1.2, 1.2))

    num_orientation = param.Integer(default=4, bounds=(1, None),
                                    softbounds=(1, 24),
                                    doc="Number of orientations to test.")

    # JABALERT: Presumably this should be omitted, so that size is included?
    static_parameters = param.List(default=["scale", "offset"])


    def _feature_list(self, p):
        return [f.X(range=p.x_range, steps=p.divisions,
                    preference_fn=self.preference_fn),
                f.Y(range=p.y_range, steps=p.divisions,
                    preference_fn=self.preference_fn),
                f.Orientation(range=(0, 2*np.pi), steps=p.num_orientation)]


class measure_corner_angle_pref(PositionMeasurementCommand):
    """Generate the preference map for angle shapes, by collating the response to patterns."""

    scale = param.Number(default=1.0)

    size = param.Number(default=0.2)

    positions = param.Integer(default=7)

    x_range = param.NumericTuple((-1.0, 1.0))

    y_range = param.NumericTuple((-1.0, 1.0))

    num_or = param.Integer(default=4, bounds=(1, None), softbounds=(1, 24), doc="""
        Number of orientations to test.""")

    angle_0 = param.Number(default=0.25 * np.pi, bounds=(0.0, np.pi),
                           softbounds=(0.0, 0.5 * np.pi), doc="""
                           First angle to test.""")

    angle_1 = param.Number(default=0.75 * np.pi, bounds=(0.0, np.pi),
                           softbounds=(0.5 * np.pi, np.pi), doc="""
                           Last angle to test.""")

    num_angle = param.Integer(default=4, bounds=(1, None), softbounds=(1, 12),
                              doc="Number of angles to test.")

    pattern_generator = param.Callable(
        default=GaussiansCorner(aspect_ratio=4.0, cross=0.85))

    static_parameters = param.List(default=["size", "scale", "offset"])


    def _feature_list(self, p):
        """Return the list of features to vary, generate hue code static image"""
        if p.angle_0 < p.angle_1:
            angle_0 = p.angle_0
            angle_1 = p.angle_1
        else:
            angle_0 = p.angle_1
            angle_1 = p.angle_0
        a_range = (angle_0, angle_1)
        self._make_key_image(p)
        return [f.X(range=p.x_range, steps=p.positions),
                f.Y(range=p.y_range, steps=p.positions),
                f.Orientation(range=(0, 2 * np.pi), steps=p.num_or),
                f.Angle(range=a_range, steps=p.num_angle)]


    def _make_key_image(self, p):
        """
        Generate the image with keys to hues used to code angles the image is
        saved on-the-fly, in order to fit the current choice of angle range
        """
        width = 60
        height = 300
        border = 6
        n_a = 7
        angle_0 = p.angle_0
        angle_1 = p.angle_1
        a_step = 0.5 * ( angle_1 - angle_0 ) / float(n_a)
        x_0 = border
        x_1 = ( width - border ) / 2
        x_a = x_1 + 2 * border
        y_use = height - 2 * border
        y_step = y_use / float(n_a)
        y_d = int(float(0.5 * y_step))
        y_0 = border + y_d
        l = 15

        hues = ["hsl(%2d,100%%,50%%)" % h for h in range(0, 360, 360 / n_a)]
        angles = [0.5 * angle_0 + a_step * a for a in range(n_a)]
        y_pos = [int(np.round(y_0 + y * y_step)) for y in range(n_a)]
        deltas = [(int(np.round(l * np.cos(a))),
                   int(np.round(l * np.sin(a)))) for a in angles]
        lb_img = PILImage.new("RGB", (width, height), "white")
        dr_img = ImageDraw.Draw(lb_img)

        for h, y, d in zip(hues, y_pos, deltas):
            dr_img.rectangle([(x_0, y - y_d), (x_1, y + y_d)], fill=h)
            dr_img.line([(x_a, y), (x_a + d[0], y + d[1])], fill="black")
            dr_img.line([(x_a, y), (x_a + d[0], y - d[1])], fill="black")

        lb_img.save(p.key_img_fname)

        return ( self.key_img_fname )


class measure_position_pref(PositionMeasurementCommand):
    """Measure a position preference map by collating the response to patterns."""

    scale = param.Number(default=0.3)

    def _feature_list(self, p):
        return [f.X(range=p.x_range, steps=p.divisions,
                    preference_fn=self.preference_fn),
                f.Y(range=p.y_range, steps=p.divisions,
                    preference_fn=self.preference_fn)]


class measure_or_tuning_fullfield(FeatureCurveCommand):
    """
    Measures orientation tuning curve(s) of a particular unit using a
    full-field sine grating stimulus.

    The curve can be plotted at various different values of the
    contrast (or actually any other parameter) of the stimulus.  If
    using contrast and the network contains an LGN layer, then one
    would usually specify michelson_contrast as the
    contrast_parameter. If there is no explicit LGN, then scale
    (offset=0.0) can be used to define the contrast.  Other relevant
    contrast definitions (or other parameters) can also be used,
    provided they are defined in CoordinatedPatternGenerator and the units
    parameter is changed as appropriate.
    """

    coords = param.Parameter(default=None,
                             doc="""Ignored; here just to suppress warning.""")

    pattern_generator = param.Callable(default=SineGrating())


class measure_or_tuning(UnitCurveCommand):
    """
    Measures orientation tuning curve(s) of a particular unit.

    Uses a circular sine grating patch as the stimulus on the
    retina.

    The curve can be plotted at various different values of the
    contrast (or actually any other parameter) of the stimulus.  If
    using contrast and the network contains an LGN layer, then one
    would usually specify weber_contrast as the contrast_parameter. If
    there is no explicit LGN, then scale (offset=0.0) can be used to
    define the contrast.  Other relevant contrast definitions (or
    other parameters) can also be used, provided they are defined in
    CoordinatedPatternGenerator and the units parameter is changed as
    appropriate.
    """

    num_orientation = param.Integer(default=12)

    static_parameters = param.List(default=["size", "x", "y"])


    def __call__(self, **params):
        p = ParamOverrides(self, params, allow_extra_keywords=True)
        self._set_presenter_overrides(p)
        results = {}
        for coord in p.coords:
            p.x = p.preference_lookup_fn('x', p.outputs[0], coord,
                                         default=coord[0])
            p.y = p.preference_lookup_fn('y', p.outputs[0], coord,
                                         default=coord[1])
            results[coord] = self._compute_curves(p)
        results = self._populate_grid(results)

        self._restore_presenter_defaults()
        return results



class measure_size_response(UnitCurveCommand):
    """
    Measure receptive field size of one unit of a sheet.

    Uses an expanding circular sine grating stimulus at the preferred
    orientation and retinal position of the specified unit.
    Orientation and position preference must be calulated before
    measuring size response.

    The curve can be plotted at various different values of the
    contrast (or actually any other parameter) of the stimulus.  If
    using contrast and the network contains an LGN layer, then one
    would usually specify weber_contrast as the contrast_parameter. If
    there is no explicit LGN, then scale (offset=0.0) can be used to
    define the contrast.  Other relevant contrast definitions (or
    other parameters) can also be used, provided they are defined in
    CoordinatedPatternGenerator and the units parameter is changed as
    appropriate.
    """
    size = None # Disabled unused parameter

    static_parameters = param.List(default=["orientation", "x", "y"])

    num_sizes = param.Integer(default=11, bounds=(1, None), softbounds=(1, 50),
                              doc="Number of different sizes to test.")

    max_size = param.Number(default=1.0, bounds=(0.1, None), softbounds=(1, 50),
                            doc="Maximum extent of the grating")

    x_axis = param.String(default="size", constant=True)


    def __call__(self, **params):
        p = ParamOverrides(self, params, allow_extra_keywords=True)
        self._set_presenter_overrides(p)
        results = {}
        for coord in p.coords:
            # Orientations are stored as a normalized value beween 0
            # and 1, so we scale them by pi to get the true orientations.
            p.orientation = p.preference_lookup_fn('orientation', p.outputs[0],
                                                   coord)
            p.x = p.preference_lookup_fn('x', p.outputs[0], coord,
                                         default=coord[0])
            p.y = p.preference_lookup_fn('y', p.outputs[0], coord,
                                         default=coord[1])
            results[coord] = self._compute_curves(p)
        results = self._populate_grid(results)

        self._restore_presenter_defaults()
        return results


    def _feature_list(self, p):
        return [f.Phase(steps=p.num_phase),
                f.Frequency(values=p.frequencies),
                f.Size(range=(0.0, p.max_size),
                        steps=p.num_sizes),
                f.Contrast(values=p.contrasts, preference_fn=None)]


class measure_contrast_response(UnitCurveCommand):
    """
    Measures contrast response curves for a particular unit.

    Uses a circular sine grating stimulus at the preferred
    orientation and retinal position of the specified unit.
    Orientation and position preference must be calulated before
    measuring contrast response.

    The curve can be plotted at various different values of the
    contrast (or actually any other parameter) of the stimulus.  If
    using contrast and the network contains an LGN layer, then one
    would usually specify weber_contrast as the contrast_parameter. If
    there is no explicit LGN, then scale (offset=0.0) can be used to
    define the contrast.  Other relevant contrast definitions (or
    other parameters) can also be used, provided they are defined in
    CoordinatedPatternGenerator and the units parameter is changed as
    appropriate.
    """

    static_parameters = param.List(default=["size", "x", "y"])

    contrasts = param.List(class_=int, default=[10, 20, 30, 40, 50,
                                                60, 70, 80, 90, 100])

    relative_orientations = param.List(class_=float, default=[0.0, np.pi / 6,
                                                              np.pi / 4,
                                                              np.pi / 2])

    x_axis = param.String(default='contrast', constant=True)


    def __call__(self, **params):
        p = ParamOverrides(self, params, allow_extra_keywords=True)
        self._set_presenter_overrides(p)
        results = {}
        for coord in p.coords:
            p.orientation = p.preference_lookup_fn('orientation', p.outputs[0],
                                                   coord)

            p.x = p.preference_lookup_fn('x', p.outputs[0], coord,
                                         default=coord[0])
            p.y = p.preference_lookup_fn('y', p.outputs[0], coord,
                                         default=coord[1])

            results[coord] = self._compute_curves(p)
        results = self._populate_grid(results)

        self._restore_presenter_defaults()
        return results


    def _feature_list(self, p):
        return [f.Phase(steps=p.num_phase),
                f.Frequency(values=p.frequencies),
                f.Contrast(values=p.contrasts),
                f.Orientation(preference_fn=None,
                              values=[p.orientation+ro
                                      for ro in p.relative_orientations])]


class measure_frequency_response(UnitCurveCommand):
    """
    Measure spatial frequency preference of one unit of a sheet.

    Uses an constant circular sine grating stimulus at the preferred
    with varying spatial frequency orientation and retinal position
    of the specified unit. Orientation and position preference must
    be calulated before measuring size response.

    The curve can be plotted at various different values of the
    contrast (or actually any other parameter) of the stimulus.  If
    using contrast and the network contains an LGN layer, then one
    would usually specify weber_contrast as the contrast_parameter. If
    there is no explicit LGN, then scale (offset=0.0) can be used to
    define the contrast.  Other relevant contrast definitions (or
    other parameters) can also be used, provided they are defined in one of the
    appropriate metaparameter_fns.
    """

    x_axis = param.String(default="frequency", constant=True)

    static_parameters = param.List(default=["orientation", "x", "y"])

    num_freq = param.Integer(default=21, bounds=(1, None), softbounds=(1, 50),
                             doc="Number of different sizes to test.")

    max_freq = param.Number(default=10.0, bounds=(0.1, None),
                            softbounds=(1, 50),
                            doc="Maximum extent of the grating")


    def __call__(self, **params):
        p = ParamOverrides(self, params, allow_extra_keywords=True)
        self._set_presenter_overrides(p)
        results = {}
        for coord in p.coords:
            # Orientations are stored as a normalized value beween 0
            # and 1, so we scale them by pi to get the true orientations.
            p.orientation = np.pi * p.preference_lookup_fn('orientation',
                                                           p.outputs[0], coord)
            p.x = p.preference_lookup_fn('x', p.outputs[0], coord,
                                         default=coord[0])
            p.y = p.preference_lookup_fn('y', p.outputs[0], coord,
                                         default=coord[1])

            results[coord] = self._compute_curves(p)
        results = self._populate_grid(results)

        self._restore_presenter_defaults()
        return results


    def _feature_list(self, p):
        return [f.Orientation(values=[p.orientation]),
                f.Phase(steps=p.num_phase),
                f.Frequency(range=(0.0, p.max_freq),
                            steps=p.num_freq),
                f.Size(values=[p.size]),
                f.Contrast(values=p.contrasts, preference_fn=None)]


class measure_orientation_contrast(UnitCurveCommand):
    """
    Measures the response to a center sine grating disk and a surround
    sine grating ring at different contrasts of the central disk.

    The central disk is set to the preferred orientation of the unit
    to be measured. The surround disk orientation (relative to the
    central grating) and contrast can be varied, as can the size of
    both disks.
    """

    metafeature_fns = param.HookList(
        default=[contrast2centersurroundscale.instance(contrast_parameter='weber_contrast')])

    pattern_generator = param.Callable(
        default=OrientationContrast(surround_orientation_relative=True))

    size = None # Disabled unused parameter
    # Maybe instead of the below, use size and some relative parameter, to allow easy scaling?

    sizecenter = param.Number(default=0.5, bounds=(0, None), doc="""
        The size of the central pattern to present.""")

    sizesurround = param.Number(default=1.0, bounds=(0, None), doc="""
        The size of the surround pattern to present.""")

    thickness = param.Number(default=0.5, bounds=(0, None), softbounds=(0, 1.5),
                             doc="""Ring thickness.""")

    contrastsurround = param.List(default=[30, 60, 80, 90],
                                  doc="Contrast of the surround.")

    contrastcenter = param.Number(default=100, bounds=(0, 100),
                                  doc="""Contrast of the center.""")

    x_axis = param.String(default='orientationsurround', constant=True)

    orientation_center = param.Number(default=0.0, softbounds=(0.0, np.pi), doc="""
        Orientation of the center grating patch""")

    num_orientation = param.Integer(default=9)

    static_parameters = param.List(
        default=["x", "y", "sizecenter", "sizesurround", "orientationcenter",
                 "thickness", "contrastcenter"])

    or_surrounds = []

    def __call__(self, **params):
        p = ParamOverrides(self, params, allow_extra_keywords=True)
        self._set_presenter_overrides(p)
        if not p.num_orientation % 2:
            raise Exception("Use odd number of surround orientation to ensure"
                            "the orthogonal to the preferred orientation is"
                            "covered.")

        results = {}
        for coord in p.coords:
            orientation = p.preference_lookup_fn('orientation', p.outputs[0],
                                                 coord, default=p.orientation_center)
            p.orientationcenter = orientation
            p.phase = p.preference_lookup_fn('phase', p.outputs[0],
                                             coord, default=p.orientation_center)
            p.x = p.preference_lookup_fn('x', p.outputs[0], coord,
                                         default=coord[0])
            p.y = p.preference_lookup_fn('y', p.outputs[0], coord,
                                         default=coord[1])

            self.or_surrounds = list(np.linspace(-np.pi/2, np.pi/2, p.num_orientation))
            results[coord] = self._compute_curves(p)
        results = self._populate_grid(results)

        self._restore_presenter_defaults()
        return results


    def _feature_list(self, p):
        return [f.Frequency(values=p.frequencies),
                f.Phase(steps=p.num_phase,
                        preference_fn=DSF_MaxValue()),
                f.OrientationSurround(values=self.or_surrounds,
                                      preference_fn=DSF_MaxValue()),
                f.ContrastSurround(values=p.contrastsurround,
                                   preference_fn=None)]



class test_measure(UnitCurveCommand):
    static_parameters = param.List(default=["size", "x", "y"])

    x_axis = param.String(default='contrast', constant=True)

    units = param.String(default=" rad")


    def __call__(self, **params):
        p = ParamOverrides(self, params, allow_extra_keywords=True)
        self.x = 0.0
        self.y = 0.0
        self._compute_curves(p)


    def _feature_list(self, p):
        return [f.Orientation(values=[1.0] * 22, preference_fn=None),
                f.Contrast(values=[100])]


__all__ = [
    "measure_corner_angle_pref",
    "measure_corner_or_pref",
    "measure_dr_pref",
    "measure_hue_pref",
    "measure_od_pref",
    "measure_or_pref",
    "measure_phasedisparity",
    "measure_response",
    "measure_rfs",
    "measure_second_or_pref",
    "measure_sine_pref",
    "measure_contrast_response",
    "measure_frequency_response",
    "measure_or_tuning",
    "measure_or_tuning_fullfield",
    "measure_orientation_contrast",
    "measure_position_pref",
    "measure_size_response",
    "test_measure"
]


#=================#
# Collector hooks #
#=================#


def array_hook(obj, *args, **kwargs):
    return None if obj is None else Image(obj.copy(), **kwargs)

def measurement_hook(obj, *args, **kwargs):
    return obj(*args, **kwargs)

def pattern_hook(obj,*args, **kwargs):
    return obj[:]

Collector.for_type(np.ndarray, array_hook)
Collector.for_type(measure_response, measurement_hook, mode='merge')
Collector.for_type(MeasureResponseCommand,  measurement_hook, mode='merge')
Collector.for_type(imagen.PatternGenerator, pattern_hook)
