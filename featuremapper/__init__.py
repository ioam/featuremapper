"""
FeatureResponses and associated functions and classes.

These classes implement map and tuning curve measurement based
on measuring responses while varying features of an input pattern.
"""
import sys, os

import param
from param.version import Version
__version__ = Version(release=(0,2,0), fpath=__file__, commit="$Format:%h$")

import copy
from collections import defaultdict
from itertools import product

import numpy as np

from param.parameterized import ParamOverrides, bothmethod
from dataviews.ndmapping import AttrDict, NdMapping, Dimension
from dataviews.options import options, channels, StyleOpts, ChannelOpts
from dataviews.sheetviews import SheetView, SheetStack, CoordinateGrid
from dataviews.collector import AttrTree

from .distribution import Distribution, DistributionStatisticFn, DSF_WeightedAverage
from . import features
from .features import Feature # pyflakes:ignore (API import)

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
        self.full_matrix[index] = new_values.copy()



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

    static_features = param.Dict(default={}, doc="""
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

    store_responses = param.Boolean(default=False, doc="""
        Determines whether or not to return the full set of responses to the
        presented patterns.""")

    metadata = {}

    __abstract = True

    def _initialize_featureresponses(self, p):
        """
        Create an empty DistributionMatrix for each feature and each
        measurement source, in addition to activity buffers and if
        requested, the full matrix.
        """
        self._apply_cmd_overrides(p)
        self.metadata = AttrDict(p.metadata)
        for fn in p.metadata_fns:
            self.metadata.update(fn(p.inputs, p.outputs))

        # Features are split depending on whether a preference_fn is supplied
        # to collapse them
        self.outer = [f for f in self.features if f.preference_fn is None]
        self.inner = [f for f in self.features if f.preference_fn is not None]
        self.outer_names, self.outer_vals = [(), ()] if not len(self.outer)\
            else zip(*[(f.name, f.values) for f in self.outer])
        dimensions = [features.Duration] + list(self.outer)

        self.measurement_product = [mp for mp in product(self.metadata.outputs.keys(),
                                                         p.durations, *self.outer_vals)]

        ndmapping_fn = lambda: NdMapping(dimensions=dimensions)
        self._featureresponses = defaultdict(ndmapping_fn)
        self._activities = defaultdict(ndmapping_fn)
        if p.store_responses:
            response_dimensions = [features.Time]+dimensions+list(self.inner)
            response_stack_fn = lambda: SheetStack(dimensions=response_dimensions)
            self._responses = defaultdict(response_stack_fn)

        for label in self.measurement_product:
            out_label = label[0]
            output_metadata = self.metadata.outputs[out_label]
            f_vals = label[1:]

            self._activities[out_label][f_vals] = np.zeros(output_metadata['shape'])

            self._featureresponses[out_label][f_vals] = {}
            for f in self.inner:
                self._featureresponses[out_label][f_vals][f.name] = \
                    DistributionMatrix(output_metadata['shape'], axis_range=f.range,
                                       cyclic=f.cyclic)



    def _measure_responses(self, p):
        """
        Generate feature permutations and present each in sequence.
        """

        # Run hooks before the analysis session
        for f in p.pre_analysis_session_hooks: f()

        features_to_permute = [f for f in self.inner if f.compute_fn is None]
        self.features_to_compute = [f for f in self.inner if f.compute_fn is not None]

        self.feature_names, values_lists = zip(*[(f.name, f.values) for f in features_to_permute])

        self.permutations = [permutation for permutation in product(*values_lists)]

        # Permute outer or non-collapsed features
        self.outer_permutations = [permutation for permutation in product(*self.outer_vals)]
        if not self.outer_permutations: self.outer_permutations.append(())
        self.n_outer = len(self.outer_permutations)

        self.total_steps = len(self.permutations) * len(self.outer_permutations) * p.repetitions - 1
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
        for label in self.measurement_product:
            out_label = label[0]
            f_vals = label[1:]
            self._activities[out_label][f_vals] *= 0

        # Calculate complete set of settings
        permuted_settings = zip(self.feature_names, permutation)
        complete_settings = permuted_settings +\
                            [(f.name, f.compute_fn(permuted_settings))
                             for f in self.features_to_compute]

        for i, op in enumerate(self.outer_permutations):
            for j in range(0, p.repetitions):
                permutation = dict(permuted_settings)
                permutation.update(zip(self.outer_names, op))

                for f in p.pre_presentation_hooks: f()

                presentation_num = p.repetitions*((self.n_outer*permutation_num)+i) + j

                inputs = self._coordinate_inputs(p, permutation)

                responses = p.pattern_response_fn(inputs, output_names,
                                                  presentation_num, self.total_steps,
                                                  durations=p.durations)

                for f in p.post_presentation_hooks: f()

                for response_labels, response in responses.items():
                    name, duration = response_labels
                    self._activities[name][(duration,)+op] += response

            for response_labels in responses.keys():
                name, duration = response_labels
                self._activities[name][(duration,)+op] /= p.repetitions

        self._update(p, complete_settings)


    def _coordinate_inputs(self, p, feature_values):
        """
        Generates pattern generators for all the requested inputs, applies the
        correct feature values and iterates through the metafeature_fns,
        coordinating complex features.
        """
        input_names = self.metadata.inputs.keys()
        feature_values = dict(feature_values, **p.static_features)

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
        timestamp = self.metadata['timestamp']
        for mvals in self.measurement_product:
            name = mvals[0]
            bounds = self.metadata.outputs[name]['bounds']
            f_vals = mvals[1:]
            act = self._activities[name][f_vals]
            for feature, value in current_values:
                self._featureresponses[name][f_vals][feature].update(act, value)
            if p.store_responses:
                cn, cv = zip(*current_values)
                key = (timestamp,)+f_vals+cv
                self._responses[name][key] = SheetView(act.copy(), bounds,
                                                       label='Response')


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


    def _set_style(self, feature, map_type):
        fname = feature.name.capitalize()
        type_str = 'SheetView'
        style_str = options.normalize_key(fname + map_type.capitalize()) +'_' + type_str
        if style_str not in options.keys():
            cyclic = True if feature.cyclic and not map_type == 'selectivity' else False
            options[style_str] = StyleOpts(**(dict(cmap='hsv') if cyclic else dict()))


    def _collate_results(self, p):
        results = AttrTree()

        timestamp = self.metadata.timestamp

        # Generate dimension info dictionary from features
        dimensions = [features.Time, features.Duration] + self.outer
        pattern_dimensions = self.outer + self.inner
        pattern_dim_label = '_'.join(f.name.capitalize() for f in pattern_dimensions)

        for label in self.measurement_product:
            name = label[0] # Measurement source
            f_vals = label[1:] # Duration and outer feature values

            #Metadata
            inner_features = dict([(f.name, f) for f in self.inner])
            output_metadata = dict(self.metadata.outputs[name], inner_features=inner_features)
            stack_metadata = dict(dimensions=dimensions, **output_metadata)

            # Iterate over inner features
            fr = self._featureresponses[name][f_vals]
            for fname, fdist in fr.items():
                feature = fname.capitalize()
                base_name = self.measurement_prefix + feature

                # Get information from the feature
                fp = [f for f in self.features if f.name == fname][0]
                pref_fn = fp.preference_fn if fp.preference_fn is not None\
                    else self.preference_fn
                if p.selectivity_multiplier is not None:
                    pref_fn.selectivity_scale = (pref_fn.selectivity_scale[0],
                                                 p.selectivity_multiplier)

                # Get maps and iterate over them
                response = fdist.apply_DSF(pref_fn)
                for k, maps in response.items():
                    for map_name, map_view in maps.items():
                        # Set labels and metadata
                        map_index = base_name + k + map_name.capitalize()
                        map_label = ' '.join([base_name, map_name.capitalize()])
                        cyclic = (map_name != 'selectivity' and fp.cyclic)
                        value_dimension = Dimension(map_label, cyclic=cyclic, range=fp.range)
                        title = name + ' {label}'
                        self._set_style(fp, map_name)

                        # Create views and stacks
                        sv = SheetView(map_view, output_metadata['bounds'],
                                       label=' '.join([name, map_label]),
                                       metadata=AttrDict(timestamp=timestamp),
                                       value=value_dimension)
                        key = (timestamp,)+f_vals
                        if (map_label, name) not in results:
                            results.set_path((map_index, name), SheetStack((key, sv), **stack_metadata))
                        else:
                            results.path_items[(map_index, name)][key] = sv
                if p.store_responses:
                    info = (p.pattern_generator.__class__.__name__, pattern_dim_label, 'Response')
                    results.set_path(('%s_%s_%s' % info, name), self._responses[name])

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
        results = AttrTree()

        timestamp = self.metadata.timestamp
        axis_name = p.x_axis.capitalize()
        axis_feature = [f for f in self.features if f.name == p.x_axis][0]
        curve_label = ''.join([p.measurement_prefix, axis_name, 'Tuning'])

        dimensions = [features.Time, features.Duration] + [f for f in self.outer] + [axis_feature]
        pattern_dimensions = self.outer + self.inner
        pattern_dim_label = '_'.join(f.name.capitalize() for f in pattern_dimensions)

        for label in self.measurement_product:
            # Deconstruct label into source name and feature_values
            name = label[0]
            f_vals = label[1:]

            # Get data and metadata from the DistributionMatrix objects
            dist_matrix = self._featureresponses[name][f_vals][p.x_axis]
            curve_responses = dist_matrix.distribution_matrix

            output_metadata = self.metadata.outputs[name]
            rows, cols = output_metadata['shape']

            # Create top level NdMapping indexing over time, duration, the outer
            # feature dimensions and the x_axis dimension
            if name not in results:
                stack = SheetStack(dimensions=dimensions, timestamp=timestamp,
                                   **output_metadata)
                results.set_path((curve_label, name), stack)

            metadata = AttrDict(timestamp=timestamp, **output_metadata)

            # Populate the SheetStack with measurements for each x value
            for x in curve_responses[0, 0]._data.iterkeys():
                y_axis_values = np.zeros(output_metadata['shape'], activity_dtype)
                for i in range(rows):
                    for j in range(cols):
                        y_axis_values[i, j] = curve_responses[i, j].get_value(x)
                key = (timestamp,)+f_vals+(x,)
                sv = SheetView(y_axis_values, output_metadata['bounds'],
                               label=' '.join([name, curve_label]),
                               metadata=metadata.copy(), value=Dimension('Response'))
                results.path_items[(curve_label, name)][key] = sv
            if p.store_responses:
                info = (p.pattern_generator.__class__.__name__, pattern_dim_label, 'Response')
                results.set_path(('%s_%s_%s' % info, name), self._responses[name])
            
        return results



class ReverseCorrelation(FeatureResponses):
    """
    Calculate the receptive fields for all neurons using reverse correlation.
    """

    continue_measurement = param.Boolean(default=True)

    def __call__(self, features, **params):
        p = ParamOverrides(self, params, allow_extra_keywords=True)
        self.features = features
        self.n_permutation = 0

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
        self.measurement_product = [label for label in product(self.metadata.inputs.keys(),
                                                              self.metadata.outputs.keys(),
                                                              p.durations)]


        self.inner = [f for f in self.features if f.preference_fn is not None]

        self.outer = [f for f in self.features if f.preference_fn is None]
        self.outer_names, self.outer_vals = [(), ()] if not len(self.outer)\
            else zip(*[(f.name, f.values) for f in self.outer])

        if p.store_responses:
            response_dimensions = [features.Time, features.Duration]+self.outer+self.inner
            response_stack_fn = lambda: SheetStack(dimensions=response_dimensions)
            self._responses = defaultdict(response_stack_fn)

        # Set up the featureresponses measurement dict
        self._featureresponses = defaultdict(lambda: defaultdict(dict))
        for labels in self.measurement_product:
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
        self.n_permutation += 1


    def _update(self, p, responses):
        """
        Updates featureresponses object with latest reverse correlation data.
        """
        timestamp = self.metadata['timestamp']
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
                    if p.store_responses:
                        key = (timestamp, d, self.n_permutation)
                        bounds = output_metadata['bounds']
                        self._responses[out_label][key] = SheetView(out_response.copy(), bounds,
                                                                    label='Response')


    def _collate_results(self, p):
        """
        Collate responses into the results dictionary containing a
        ProjectionGrid for each measurement source.
        """
        results = AttrTree()

        timestamp = self.metadata.timestamp
        dimensions = [features.Time(), features.Duration()]
        pattern_dimensions = self.outer + self.inner
        pattern_dim_label = '_'.join(f.name.capitalize() for f in pattern_dimensions)

        for labels in self.measurement_product:
            in_label, out_label, duration = labels
            input_metadata = self.metadata.inputs[in_label]
            output_metadata = self.metadata.outputs[out_label]
            rows, cols = output_metadata['shape']
            time_key = (timestamp, duration)
            title = ' '.join([p.measurement_prefix, out_label, '{label}'])
            view = CoordinateGrid(output_metadata['bounds'], output_metadata['shape'],
                                  title=title, label='RFs')
            metadata = dict(dimensions=dimensions, **input_metadata)
            rc_response = self._featureresponses[in_label][out_label][duration]

            for ii in range(rows):
                for jj in range(cols):
                    coord = view.matrixidx2sheet(ii, jj)
                    sv = SheetView(rc_response[ii, jj], input_metadata['bounds'],
                                   metadata=AttrDict(timestamp=timestamp),
                                   title=title, label='Receptive Field')
                    view[coord] = SheetStack((time_key, sv), **metadata)
            label = '%s_Reverse_Correlation' % out_label
            results.set_path((label, in_label), view)
            if p.store_responses:
                info = (p.pattern_generator.__class__.__name__, pattern_dim_label, 'Response')
                results.set_path(('%s_%s_%s' % info, out_label), self._responses[out_label])
        return results

#Default styles
options.Preference_SheetView = StyleOpts(cmap='jet')
options.Selectivity_SheetView = StyleOpts(cmap='gray')
options.Activity_SheetView = StyleOpts(cmap='gray')
options.Response_SheetView = StyleOpts(cmap='gray')

# Default channel definitions
channels.Preference_Selectivity = ChannelOpts('HCS', 'Preference * Selectivity', flipSC=True)


__all__ = [
    "DistributionMatrix",
    "FullMatrix",
    "FeatureResponses",
    "ReverseCorrelation",
    "FeatureMaps",
    "FeatureCurves",
    "Feature",
]
