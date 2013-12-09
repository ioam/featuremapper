"""
FeatureResponses and associated functions and classes.

These classes implement map and tuning curve measurement based
on measuring responses while varying features of an input pattern.
"""

import copy

from collections import defaultdict

import numpy as np
from itertools import product

import param
from param.parameterized import ParamOverrides, bothmethod

from imagen.ndmapping import AttrDict
from imagen.views import SheetView, SheetStack, ProjectionGrid, NdMapping

from distribution import Distribution, DistributionStatisticFn, DSF_WeightedAverage

activity_dtype = np.float64

class PatternDrivenAnalysis(param.ParameterizedFunction):
    """
    Abstract base class for various stimulus-response types of analysis.

    This type of analysis consists of presenting a set of input
    patterns and collecting the responses to each one, which one will
    often want to do in a way that does not affect the current state
    of the network.

    To achieve this, the class defines several types of hooks where
    arbitrary function objects (i.e., callables) can be registered.
    These hooks are generally used to ensure that unrelated previous
    activity is eliminated, that subsequent patterns do not interact,
    and that the initial state is restored after analysis.

    Any subclasses must ensure that these hook lists are run at the
    appropriate stage in their processing, using e.g.
    "for f in some_hook_list: f()".
    """

    __abstract = True

    pre_analysis_session_hooks = param.HookList(default=[],instantiate=False,doc="""
        List of callable objects to be run before an analysis session begins.""")

    pre_presentation_hooks = param.HookList(default=[],instantiate=False,doc="""
        List of callable objects to be run before each pattern is presented.""")

    post_presentation_hooks = param.HookList(default=[],instantiate=False,doc="""
        List of callable objects to be run after each pattern is presented.""")

    post_analysis_session_hooks = param.HookList(default=[],instantiate=False,doc="""
        List of callable objects to be run after an analysis session ends.""")


# CB: having a class called DistributionMatrix with an attribute
# distribution_matrix to hold the distribution matrix seems silly.
# Either rename distribution_matrix or make DistributionMatrix into
# a matrix.
class DistributionMatrix(param.Parameterized):
    """
    Maintains a matrix of Distributions (each of which is a dictionary
    of (feature value: activity) pairs).

    The matrix contains one Distribution for each unit in a
    rectangular matrix (given by the matrix_shape constructor
    argument).  The contents of each Distribution can be updated for a
    given bin value all at once by providing a matrix of new values to
    update().

    The results can then be accessed as a matrix of weighted averages
    (which can be used as a preference map) and/or a selectivity map
    (which measures the peakedness of each distribution).
    """

    def __init__(self, matrix_shape, axis_range=(0.0, 1.0), cyclic=False,
                 keep_peak=True):
        """Initialize the internal data structure: a matrix of Distribution
        objects."""
        self.axis_range = axis_range
        new_distribution = np.vectorize(
            lambda x: Distribution(axis_range, cyclic, keep_peak),
            doc="Return a Distribution instance for each element of x.")
        self.distribution_matrix = new_distribution(np.empty(matrix_shape))


    def update(self, new_values, bin):
        """Add a new matrix of histogram values for a given bin value."""
        ### JABHACKALERT!  The Distribution class should override +=,
        ### rather than + as used here, because this operation
        ### actually modifies the distribution_matrix, but that has
        ### not yet been done.  Alternatively, it could use a different
        ### function name altogether (e.g. update(x,y)).
        self.distribution_matrix + np.fromfunction(
            np.vectorize(lambda i, j: {bin: new_values[i, j]}),
            new_values.shape)

    def apply_DSF(self, dsf):
        """
        Apply the given dsf DistributionStatisticFn on each element of
        the distribution_matrix

        Return a dictionary of dictionaries, with the same structure
        of the called DistributionStatisticFn, but with matrices as
        values, instead of scalars
        """

        shape = self.distribution_matrix.shape
        result = {}

        # this is an extra call to the dsf() DistributionStatisticFn,
        # in order to retrieve
        # the dictionaries structure, and allocate the necessary matrices
        r0 = dsf(self.distribution_matrix[0, 0])
        for k, maps in r0.items():
            result[k] = {}
            for m in maps.keys():
                result[k][m] = np.zeros(shape, np.float64)

        for i in range(shape[0]):
            for j in range(shape[1]):
                response = dsf(self.distribution_matrix[i, j])
                for k, d in response.items():
                    for item, item_value in d.items():
                        result[k][item][i, j] = item_value

        return result


class FullMatrix(param.Parameterized):
    """
    Records the output of every unit in a sheet, for every combination
    of feature values.  Useful for collecting data for later analysis
    while presenting many input patterns.
    """

    def __init__(self, matrix_shape, features):
        self.matrix_shape = matrix_shape
        self.features = features
        self.dimensions = ()
        for f in features:
            self.dimensions = self.dimensions + (np.size(f.values),)
        self.full_matrix = np.empty(self.dimensions, np.object_)


    def update(self, new_values, feature_value_permutation):
        """Add a new matrix of histogram values for a given bin value."""
        index = ()
        for f in self.features:
            for ff, value in feature_value_permutation:
                if (ff == f.name):
                    index = index + (f.values.index(value),)
        self.full_matrix[index] = new_values



class MeasurementInterrupt(Exception):
    """
    Exception raised when a measurement is stopped before
    completion. Stores the number of executed presentations and
    total presentations.
    """
    def __init__(self, current, total):
        self.current = current
        self.total = total + 1



class FeatureResponses(PatternDrivenAnalysis):
    """
    Systematically vary input pattern feature values and collate the
    responses.

    A DistributionMatrix for each measurement source and feature is
    created.  The DistributionMatrix stores the distribution of
    activity values for that feature.  For instance, if the features
    to be tested are orientation and phase, we will create a
    DistributionMatrix for orientation and a DistributionMatrix for
    phase for each measurement source.  The orientation and phase of
    the input are then systematically varied (when measure_responses
    is called), and the responses of all units from a measurement
    source to each pattern are collected into the DistributionMatrix.

    The resulting data can then be used to plot feature maps and
    tuning curves, or for similar types of feature-based analyses.
    """

    cmd_overrides = param.Dict(default={}, doc="""
        Dictionary used to overwrite parameters on the pattern_response_fn.""")

    durations = param.List(default=[1.0], doc="""Times after presentation,
        when a measurement is taken.""")

    inputs = param.String(default=[], doc="""Names of the input supplied to
        the metadata_fns to filter out desired inputs.""")

    metadata_fns = param.HookList(default=[], instantiate=False, doc="""
        Interface functions for metadata. Should return a dictionary that at a
        minimum must contain the name and dimensions of the inputs and outputs
        for pattern presentation and response measurement.""")

    metafeature_fns = param.HookList(default=[], doc="""
        Metafeature functions can be used to coordinate lower level features
        across input devices or depending on a metafeature set on the function
        itself.""")

    measurement_prefix = param.String(default="", doc="""
        Prefix to add to the name under which results are stored.""")

    measurement_storage_hook = param.Callable(default=None, instantiate=True, doc="""
        Interface to store measurements after they have been completed.""")

    outputs = param.String(default=[], doc="""
        Names of the output source supplied to metadata_fns to filter out
        desired outputs.""")

    param_dict = param.Dict(default={}, doc="""
        Dictionary containing name value pairs of a feature, which is to be
        varied across measurements.""")

    pattern_generator = param.Callable(instantiate=True, default=None, doc="""
        Defines the input pattern to be presented.""")

    pattern_response_fn = param.Callable(default=None, instantiate=True, doc="""
        Presenter command responsible for presenting the input patterns provided
        to it and returning the response for the requested measurement sources.""")

    repetitions = param.Integer(default=1, bounds=(1, None), doc="""
        How many times each stimulus will be presented.

        Each stimulus is specified by a particular feature combination, and
        need only be presented once if the network has no other source of
        variability.  If results differ for each presentation of an identical
        stimulus (e.g. due to intrinsic noise), then this parameter can be
        increased so that results will be an average over the specified number
        of repetitions.""")

    store_fullmatrix = param.Boolean(default=False, doc="""
        Determines whether or not store the full matrix of feature
        responses as a class attribute.""")

    metadata = {}

    _fullmatrix = {}

    __abstract = True

    def _initialize_featureresponses(self, p):
        """
        Create an empty DistributionMatrix for each feature and each
        measurement source, in addition to activity buffers and if
        requested, the full matrix.
        """
        self._apply_cmd_overrides(p)
        for fn in p.metadata_fns:
            self.metadata = AttrDict(p.metadata, **fn(p.inputs, p.outputs))

        self._featureresponses = defaultdict(lambda: defaultdict(dict))
        self._activities = defaultdict(dict)

        self.measurement_labels = [label for label in product(self.metadata.outputs.keys(),
                                                              p.durations)]

        for label in self.measurement_labels:
            out_label, d = label
            output_metadata = self.metadata.outputs[out_label]
            self._activities[out_label][d] = np.zeros(output_metadata['shape'])
            for f in self.features:
                self._featureresponses[out_label][d][f.name] = \
                    DistributionMatrix(output_metadata['shape'],
                                       axis_range=f.range,
                                       cyclic=f.cyclic)
            if p.store_fullmatrix:
                self._fullmatrix[out_label][d] = FullMatrix(output_metadata['shape'],
                                                            self.features)



    def _measure_responses(self, p):
        """
        Generate feature permutations and present each in sequence.
        """

        # Run hooks before the analysis session
        for f in p.pre_analysis_session_hooks: f()

        features_to_permute = [f for f in self.features if f.compute_fn is None]
        self.features_to_compute = [f for f in self.features
                                    if f.compute_fn is not None]

        self.feature_names = [f.name for f in features_to_permute]
        values_lists = [f.values for f in features_to_permute]

        self.permutations = [permutation for permutation in product(*values_lists)]

        self.total_steps = len(self.permutations) * p.repetitions - 1
        for permutation_num, permutation in enumerate(self.permutations):
            try:
                self._present_permutation(p, permutation, permutation_num)
            except MeasurementInterrupt as MI:
                self.warning("Measurement was stopped after {0} out of {1} presentations. "
                             "Results may be incomplete.".format(MI.current, MI.total))
                break

        # Run hooks after the analysis session
        for f in p.post_analysis_session_hooks:
            f()


    def _present_permutation(self, p, permutation, permutation_num):
        """Present a pattern with the specified set of feature values."""
        output_names = self.metadata['outputs'].keys()
        for label in self.measurement_labels:
            out_label, d = label
            self._activities[out_label][d] *= 0

        # Calculate complete set of settings
        permuted_settings = zip(self.feature_names, permutation)
        complete_settings = permuted_settings +\
                            [(f.name, f.compute_fn(permuted_settings))
                             for f in self.features_to_compute]

        for i in xrange(0, p.repetitions):

            for f in p.pre_presentation_hooks: f()

            presentation_num = p.repetitions * permutation_num+i

            inputs = self._coordinate_inputs(p, dict(permuted_settings))

            responses = p.pattern_response_fn(inputs, output_names,
                                              presentation_num, self.total_steps,
                                              durations=p.durations)

            for f in p.post_presentation_hooks:
                f()

            for response_labels, response in responses.items():
                name, duration = response_labels
                self._activities[name][duration] += response

        for response_labels in responses.keys():
            name, duration = response_labels
            self._activities[name][duration] /= p.repetitions

        self._update(p, complete_settings)


    def _coordinate_inputs(self, p, feature_values):
        """
        Generates pattern generators for all the requested inputs, applies the
        correct feature values and iterates through the metafeature_fns,
        coordinating complex features.
        """
        input_names = self.metadata.inputs.keys()
        feature_values = dict(feature_values, **p.param_dict)

        for feature, value in feature_values.iteritems():
            setattr(p.pattern_generator, feature, value)

        if len(input_names) == 0:
            input_names = ['default']

        # Copy the given generator once for every input
        inputs = dict.fromkeys(input_names)
        for k in inputs.keys():
            inputs[k] = copy.deepcopy(p.pattern_generator)

        # Apply metafeature_fns
        for fn in p.metafeature_fns:
            fn(inputs, feature_values)

        return inputs


    def _update(self, p, current_values):
        """
        Update each DistributionMatrix with (activity,bin) and
        populate the full matrix, if enabled.
        """
        for label in self.measurement_labels:
            name, d = label
            act = self._activities[name][d]
            for feature, value in current_values:
                self._featureresponses[name][d][feature].update(act,value)
            if p.store_fullmatrix:
                self._fullmatrix[name][d].update(act, current_values)


    @bothmethod
    def set_cmd_overrides(self_or_cls, **kwargs):
        """
        Allows setting of cmd_overrides at the class and instance level.
        cmd_overrides are applied to the pattern_response_fn.
        """
        self_or_cls.cmd_overrides = dict(self_or_cls.cmd_overrides, **kwargs)


    def _apply_cmd_overrides(self, p):
        """
        Applies the cmd_overrides to the pattern_response_fn and
        the pattern_coordinator before launching a measurement.
        """
        for override, value in p.cmd_overrides.items():
            if override in p.pattern_response_fn.params():
                p.pattern_response_fn.set_param(override, value)



class FeatureMaps(FeatureResponses):
    """
    Measure and collect the responses to a set of features, for
    calculating feature maps.

    For each feature and each measurement source, the results are
    stored as a preference matrix and selectivity matrix in the
    sheet's sheet_views; these can then be plotted as preference or
    selectivity maps.
    """

    preference_fn = param.ClassSelector(DistributionStatisticFn,
        default=DSF_WeightedAverage(), doc="""
        Function for computing a scalar-valued preference, selectivity,
        etc. from the distribution of responses. Note that this default
        is overridden by specific functions for individual features, if
        specified in the Feature objects.""")

    selectivity_multiplier = param.Number(default=17.0, doc="""
        Scaling of the feature selectivity values, applied in all
        feature dimensions.  The multiplier sets the output scaling.
        The precise value is arbitrary, and set to match historical
        usage.""")


    def __call__(self, features, **params):
        """
        Present the given input patterns and collate the responses.

        Responses are statistics on the distributions of measure for
        every unit, extracted by functions that are subclasses of
        DistributionStatisticFn, and could be specified in each
        feature with the preference_fn parameter, otherwise the
        default in self.preference_fn is used.
        """
        p = ParamOverrides(self, params, allow_extra_keywords=True)
        self.features = features

        self._initialize_featureresponses(p)
        self._measure_responses(p)

        results = self._collate_results(p)

        if p.measurement_storage_hook:
            p.measurement_storage_hook(results)

        return results


    def _collate_results(self, p):
        results = defaultdict(dict)
        results['fullmatrix'] = self._fullmatrix if self.store_fullmatrix else None
        timestamp = self.metadata.timestamp

        for label in self.measurement_labels:
            name, duration = label
            output_metadata = self.metadata.outputs[name]
            feature_responses = self._featureresponses[name][duration]
            for feature in feature_responses:
                fp = filter(lambda f: f.name == feature, self.features)[0]
                fr = feature_responses[feature]
                ar = fr.distribution_matrix[0, 0].axis_range
                cyclic_range = ar if fp.cyclic else None
                pref_fn = fp.preference_fn if fp.preference_fn is not None\
                    else self.preference_fn
                if p.selectivity_multiplier is not None:
                    pref_fn.selectivity_scale = (pref_fn.selectivity_scale[0],
                                                 p.selectivity_multiplier)
                response = fr.apply_DSF(pref_fn)
                base_name = self.measurement_prefix + feature.capitalize()
                for k, maps in response.items():
                    for map_name, map_view in maps.items():
                        cr = None if map_name == 'selectivity' else cyclic_range
                        sv = SheetView(map_view, output_metadata['bounds'],
                                       cyclic_range=cr)
                        data = (duration, sv)
                        metadata = dict(dimension_labels=['Duration'],
                                        timestamp=timestamp,
                                        **output_metadata)
                        map_name = base_name + k + map_name.capitalize()
                        if map_name not in results[name]:
                            results[name][map_name] = SheetStack(data, **metadata)
                        else:
                            results[name][map_name][duration] = sv

        return results


class FeatureCurves(FeatureResponses):
    """
    Measures and collects the responses to a set of features, for calculating
    tuning and similar curves.

    These curves represent the response of a measurement source to patterns
    that are controlled by a set of features.  This class can collect data for
    multiple curves, each with the same x axis. The x axis represents the main
    feature value that is being varied, such as orientation.  Other feature
    values can also be varied, such as contrast, which will result in multiple
    curves (one per unique combination of other feature values).

    A particular set of patterns is constructed using a user-specified
    pattern_generator by adding the parameters determining the curve
    (curve_param_dict) to a static list of parameters (param_dict),
    and then varying the specified set of features. The input patterns will then
    be passed to the pattern_response_fn, which should return the measured
    responses for each of the requested sheets. Once the responses to all
    feature permutations has been accumulated, the measured curves are passed to
    the storage_fn and are finally returned.
    """

    x_axis = param.String(default=None, doc="""
        Parameter to use for the x axis of tuning curves.""")

    curve_params = param.String(default=None, doc="""
        Curve label, specifying the value along some feature dimension.""")

    label = param.String(default='', doc="""
        Units for labeling the curve_parameters in figure legends.
        The default is %, for use with contrast, but could be any
        units (or the empty string).""")


    def __call__(self, features, **params):
        p = ParamOverrides(self, params, allow_extra_keywords=True)
        self.features = features
        self._initialize_featureresponses(p)
        self._measure_responses(p)

        results = self._collate_results(p)

        if p.measurement_storage_hook:
            p.measurement_storage_hook(results)

        return results


    def _collate_results(self, p):
        results = {}
        results['fullmatrix'] = self._fullmatrix if self.store_fullmatrix else None

        time = self.metadata.timestamp
        curve_label = p.measurement_prefix + p.x_axis.capitalize()

        for label in self.measurement_labels:
            name, duration = label
            f_names, f_vals = zip(*p.curve_params.items())

            output_metadata = self.metadata.outputs[name]
            rows, cols = output_metadata['shape']
            metadata = dict(dimension_labels=[p.x_axis.capitalize()],
                            label=p.label, prefix=p.measurement_prefix,
                            curve_params=p.curve_params, **output_metadata)

            # Create top level NdMapping indexing over time and duration
            if name not in results:
                results[name] = NdMapping(dimension_labels=['Duration'],
                                          timestamp=time,
                                          curve_label=curve_label)

            # Create NdMapping for each feature name and populate it with an
            # entry of for the specified features
            results[name][duration] = NdMapping(dimension_labels=list(f_names))
            results[name][duration][f_vals] = SheetStack(**metadata)

            curve_responses = self._featureresponses[name][duration][p.x_axis].distribution_matrix
            r = results[name][duration][f_vals]
            # Populate the deepest NdMapping with measurements for each x value
            for x in curve_responses[0, 0]._data.iterkeys():
                y_axis_values = np.zeros(output_metadata['shape'], activity_dtype)
                for i in range(rows):
                    for j in range(cols):
                        y_axis_values[i, j] = curve_responses[i, j].get_value(x)
                sv = SheetView(y_axis_values, output_metadata['bounds'])
                r[x] = sv

        return results



class ReverseCorrelation(FeatureResponses):
    """
    Calculate the receptive fields for all neurons using reverse correlation.
    """

    continue_measurement = param.Boolean(default=True)

    def __call__(self, features, **params):
        p = ParamOverrides(self, params, allow_extra_keywords=True)
        self.features = features

        self._initialize_featureresponses(p)
        self._measure_responses(p)

        results = self._collate_results(p)

        if p.measurement_storage_hook:
            p.measurement_storage_hook(results)

        return results


    def _initialize_featureresponses(self, p):
        self._apply_cmd_overrides(p)
        self.metadata = p.metadata
        for fn in p.metadata_fns:
            self.metadata = AttrDict(self.metadata, **fn(p.inputs, p.outputs))

        # Create cross product of all sources and durations
        self.measurement_labels = [label for label in product(self.metadata.inputs.keys(),
                                                              self.metadata.outputs.keys(),
                                                              p.durations)]

        # Set up the featureresponses measurement dict
        self._featureresponses = defaultdict(lambda: defaultdict(dict))
        for labels in self.measurement_labels:
            in_label, out_label, duration = labels
            input_metadata = self.metadata.inputs[in_label]

            rows, cols = self.metadata.outputs[out_label]['shape']
            rc_array = np.array([[np.zeros(input_metadata['shape'], activity_dtype)
                                  for r in range(rows)] for c in range(cols)])

            self._featureresponses[in_label][out_label][duration] = rc_array


    def _present_permutation(self, p, permutation, permutation_num):
        """Present a pattern with the specified set of feature values."""

        # Calculate complete set of settings
        permuted_settings = zip(self.feature_names, permutation)

        # Run hooks before and after pattern presentation.
        for f in p.pre_presentation_hooks: f()

        inputs = self._coordinate_inputs(p, dict(permuted_settings))
        measurement_sources = self.metadata.outputs.keys() + self.metadata.inputs.keys()
        responses = p.pattern_response_fn(inputs, measurement_sources,
                                          permutation_num, self.total_steps,
                                          durations=p.durations)

        for f in p.post_presentation_hooks: f()

        self._update(p, responses)


    def _update(self, p, responses):
        """
        Updates featureresponses object with latest reverse correlation data.
        """
        for in_label in self.metadata.inputs:
            for out_label, output_metadata in self.metadata.outputs.items():
                for d in p.durations:
                    rows, cols = output_metadata['shape']
                    in_response = responses[(in_label, d)]
                    out_response = responses[(out_label, d)]
                    feature_responses = self._featureresponses[in_label][out_label][d]
                    for ii in range(rows):
                        for jj in range(cols):
                            delta_rf = out_response[ii, jj] * in_response
                            feature_responses[ii, jj] += delta_rf


    def _collate_results(self, p):
        """
        Collate responses into the results dictionary containing a
        ProjectionGrid for each measurement source.
        """
        results = defaultdict(dict)
        results['fullmatrix'] = self._fullmatrix if self.store_fullmatrix else None
        for labels in self.measurement_labels:
            in_label, out_label, duration = labels
            input_metadata = self.metadata.inputs[in_label]
            output_metadata = self.metadata.outputs[out_label]
            rows, cols = output_metadata['shape']
            timestamp = self.metadata['timestamp']
            view = ProjectionGrid(label=p.measurement_prefix + 'RFs',
                                  timestamp=timestamp,
                                  bounds=output_metadata['bounds'],
                                  shape=output_metadata['shape'])
            metadata = dict(measurement_src=output_metadata['src_name'],
                            dimension_labels=['Duration'], **input_metadata)
            rc_response = self._featureresponses[in_label][out_label][duration]
            for ii in range(rows):
                for jj in range(cols):
                    coord = view.matrixidx2coord(ii, jj)
                    sv = SheetView(rc_response[ii, jj], input_metadata['bounds'])
                    rf_metadata = dict(coord=coord, **metadata)
                    view[coord] = SheetStack((duration, sv), **rf_metadata)
            results[out_label][in_label] = view
        return results



class Feature(param.Parameterized):
    """
    Specifies several parameters required for generating a map of one input
    feature.
    """

    name = param.String(default="", doc="Name of the feature to test")

    cyclic = param.Boolean(default=False, doc="""
        Whether the range of this feature is cyclic (wraps around at the high
        end).""")

    compute_fn = param.Callable(default=None, doc="""
        If non-None, a function that when given a list of other parameter
        values, computes and returns the value for this feature.""")

    preference_fn = param.ClassSelector(DistributionStatisticFn,
                                        default=DSF_WeightedAverage(), doc="""
        Function that will be used to analyze the distributions of unit response
        to this feature.""")

    range = param.NumericTuple(default=(0, 0), doc="""
        Lower and upper values for a feature,used to build a list of values,
        together with the step parameter.""")

    steps = param.Integer(default=0, doc="""
        Number of steps, between lower and upper range value, to be presented.""")

    offset = param.Number(default=0.0, doc="""
        Offset to add to the values for this feature""")

    values = param.List(default=[], doc="""
        Explicit list of values for this feature, used in alternative to the
        range and step parameters""")


    def __init__(self, **params):
        """
        Users can provide either a range and a step size, or a list of
        values.  If a list of values is supplied, the range can be
        omitted unless the default of the min and max in the list of
        values is not appropriate.

        If non-None, the compute_fn should be a function that when
        given a list of other parameter values, computes and returns
        the value for this feature.

        If supplied, the offset is added to the given or computed
        values to allow the starting value to be specified.

        """

        super(Feature, self).__init__(**params)

        if len(self.values):
            self.values = self.values if self.offset == 0 \
                else [v + self.offset for v in self.values]
            if self.range == (0, 0):
                self.range = (min(self.values), max(self.values))
        else:
            if self.range == (0, 0):
                raise ValueError('The range or values must be specified.')
            low_bound, up_bound = self.range
            values = np.linspace(low_bound, up_bound, self.steps, not self.cyclic) + self.offset
            self.values = list(values % (up_bound - low_bound) if self.cyclic else values)

##############################################################################
###############################################################################
###############################################################################
#
# 20081017 JABNOTE: This implementation could be improved.
#
# It currently requires every subclass to implement the feature_list
# method, which constructs a list of features using various parameters
# to determine how many and which values each feature should have.  It
# would be good to replace the feature_list method with a Parameter or
# set of Parameters, since it is simply a special data structure, and
# this would make more detailed control feasible for users. For
# instance, instead of something like num_orientations being used to
# construct the orientation Feature, the user could specify the
# appropriate Feature directly, so that they could e.g. supply a
# specific list of orientations instead of being limited to a fixed
# spacing.
#
# However, when we implemented this, we ran into two problems:
#
# 1. It's difficult for users to modify an open-ended list of
#     Features.  E.g., if features is a List:
#
#      features=param.List(doc="List of Features to vary""",default=[
#          Feature(name="frequency",values=[2.4]),
#          Feature(name="orientation",range=(0.0,pi),step=pi/4,cyclic=True),
#          Feature(name="phase",range=(0.0,2*pi),step=2*pi/18,cyclic=True)])
#
#    then it it's easy to replace the entire list, but tough to
#    change just one Feature.  Of course, features could be a
#    dictionary, but that doesn't help, because when the user
#    actually calls the function, they want the arguments to
#    affect only that call, whereas looking up the item in a
#    dictionary would only make permanent changes easy, not
#    single-call changes.
#
#    Alternatively, one could make each feature into a separate
#    parameter, and then collect them using a naming convention like:
#
#     def feature_list(self,p):
#         fs=[]
#         for n,v in self.get_param_values():
#             if n in p: v=p[n]
#             if re.match('^[^_].*_feature$',n):
#                 fs+=[v]
#         return fs
#
#    But that's quite hacky, and doesn't solve problem 2.
#
# 2. Even if the users can somehow access each Feature, the same
#    problem occurs for the individual parts of each Feature.  E.g.
#    using the separate feature parameters above, Spatial Frequency
#    map measurement would require:
#
#      from topo.command.analysis import Feature
#      from math import pi
#      pre_plot_hooks=[measure_or_pref.instance(\
#         frequency_feature=Feature(name="frequency",values=frange(1.0,6.0,
# 0.2)), \
#         phase_feature=Feature(name="phase",range=(0.0,2*pi),step=2*pi/15,
# cyclic=True), \
#         orientation_feature=Feature(name="orientation",range=(0.0,pi),
# step=pi/4,cyclic=True)])
#
#    rather than the current, much more easily controllable implementation:
#
#      pre_plot_hooks=[measure_or_pref.instance(frequencies=frange(1.0,6.0,
# 0.2),\
#         num_phase=15,num_orientation=4)]
#
#    I.e., to change anything about a Feature, one has to supply an
#    entirely new Feature, because otherwise the original Feature
#    would be changed for all future calls.  Perhaps there's some way
#    around this by copying objects automatically at the right time,
#    but if so it's not obvious.  Meanwhile, the current
#    implementation is reasonably clean and easy to use, if not as
#    flexible as it could be.


__all__ = [
    "DistributionMatrix",
    "FullMatrix",
    "FeatureResponses",
    "ReverseCorrelation",
    "FeatureMaps",
    "FeatureCurves",
    "Feature",
]
