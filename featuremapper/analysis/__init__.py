import numpy as np

import param
import colorsys

from holoviews import Element, Layout, NdOverlay, Overlay, Collator
from holoviews import RGB, Image, ItemTable, ElementOperation, Dimension

from holoviews.core.traversal import unique_dimkeys
from holoviews.operation.normalization import raster_normalization

from featuremapper.distribution import Distribution, DSF_WeightedAverage, \
    DSF_MaxValue

hsv_to_rgb = np.vectorize(colorsys.hsv_to_rgb)



class TreeOperation(param.ParameterizedFunction):
    """
    A TreeOperation is the most general Operation type; it accepts any
    HoloViews datastructure and outputs a Layout containing one or
    more elements.
    """

    group = param.String(default='Operation', doc="""
       The group string used to identify the output of the
       Operation. By default this should match the operation name.""")


    @classmethod
    def search(cls, element, pattern):
        """
        Helper method that returns a list of elements that match the
        given path pattern of form {type}.{group}.{label}.

        The input may be a Layout, an Overlay type or a single
        Element.
        """
        if isinstance(element, Layout):
            return [el for cell in element for el in cls.search(cell, pattern)]
        if isinstance(element, (NdOverlay, Overlay)):
            return [el for el in element if el.matches(pattern)]
        elif isinstance(element, Element):
            return [element] if element.matches(pattern) else []


    @classmethod
    def get_overlay_label(cls, overlay, default_label=''):
        """
        Returns a label if all the elements of an overlay agree on a
        consistent label, otherwise returns the default label.
        """
        if all(el.label==overlay.get(0).label for el in overlay):
            return overlay.get(0).label
        else:
            return default_label


    @classmethod
    def get_overlay_bounds(cls, overlay):
        """
        Returns the extents if all the elements of an overlay agree on
        a consistent extents, otherwise raises an exception.
        """
        if all(el.bounds==overlay.get(0).bounds for el in overlay):
            return overlay.get(0).bounds
        else:
            raise ValueError("Extents across the overlay are inconsistent")


    def process_element(self, element, key, **params):
        """
        The process_element method allows a single element to be
        operated on given an externally supplied key.
        """
        self.p = param.ParamOverrides(self, params)
        maps = self._process(element, key)
        return reduce(lambda x,y: x + y, maps)


    def __call__(self, src, **params):
        self.p = param.ParamOverrides(self, params)
        dims, keys = unique_dimkeys(src)
        if isinstance(src, Layout) and not src.uniform:
            raise Exception("TreeOperation can only process uniform Layouts")

        if not dims:
            return self.process_element(src, None)
        else:
            dim_names = [d.name for d in dims]
            values = {}
            for key in keys:
                selection = src.select(**dict(zip(dim_names, key)))
                if not isinstance(selection, Layout):
                    selection = Layout.from_values([selection])
                processed = self._process(selection, key)
                if isinstance(processed, list):
                    processed = Layout.from_values(processed)
                values[key] = processed
        return Collator(values, kdims=dims)()



    def _process(self, tree, key=None):
        """
        Process a single input Layout, returning a list of
        elements to be merged with the output Layout.
        """
        raise NotImplementedError



class cyclic_difference(ElementOperation):
    """
    The cyclic difference between any two cyclic Image quantities
    normalized such that maximum possible cyclic difference is 0.5.

    Although this operation may be applied to any Image data defined
    over some cyclic quantity, in practice it is rarely used outside
    some fairly special uses; namely the computation of cyclic
    similarity (e.g. the similarity of two orientation of direction
    maps).

    The cyclic similarity is the normalized inverse of the cyclic
    difference:

         cyclic_similarity = 1 - (2.0 * cyclic_difference)

    This may be computed in conjunction with the transform operator
    over the input overlay 'input':

         transform(cyclic_difference(input), operator=lambda x: 1-2*x)

    This quantity varies between 0.0 and 1.0 where 0.0 corresponds to
    the maximum possible difference (i.e the difference between two
    anti-correlated quanitities). A cyclic similarity of 0.5 indicates
    two uncorrelated inputs and a value of 1.0 indicates perfectly
    matching cyclic values.

    Often a more useful measure is a rescaled cyclic similarity such
    that -1.0 indicates anti-correlation, 0.0 indicates
    zero-correlation and 1.0 indicates perfect correlation:

         nominal_cyclic_similarity = (2 * (cyclic_similarity - 0.5))

    Or alternatively:

         nominal_cyclic_similarity = 1 - (4.0 * cyclic_difference)

    This may be expressed as the following operation on the input:

         transform(cyclic_difference(input), operator=lambda x: 1-4*x)
    """

    value = param.String(default='CyclicDifference', doc="""
        The value assigned to the result after computing the cyclic
        difference.""")

    @classmethod
    def difference(cls, arr1, arr2):
        """
        Computes the cyclic difference between two arrays assumed to
        be normalized in the range 0.0-1.0.

        The minimum possible cyclic difference between two such
        quantities is 0.0 and the maximum possible difference is 0.5.
        """
        difference = abs(arr1 - arr2)  # Cyclic difference is symmetric
        greaterHalf = (difference >= 0.5)
        difference[greaterHalf] = 1.0 - difference[greaterHalf]
        return difference


    def _process(self, overlay, key=None):

        if len(overlay) != 2:
            raise Exception("The similarity index may only be computed"
                            "using overlays of Image Views.")

        mat1, mat2 = overlay.get(0), overlay.get(1)
        val_dims = [mat1.value_dimensions, mat2.value_dimensions]

        if tuple(len(el) for el in val_dims) != (1,1):
            raise Exception("Both input Matrices must have single value dimension.")
        if False in [val_dims[0][0].cyclic, val_dims[1][0].cyclic]:
            raise Exception("Both input Matrices must be defined as cyclic.")

        normfn = raster_normalization.instance()
        overlay = normfn.process_element(overlay, key)

        return Image(self.difference(overlay.get(0).data, overlay.get(1).data),
                     bounds=self.get_overlay_bounds(overlay),
                     group=self.p.value)


class center_cyclic(ElementOperation):
    """
    Centers the values in a 1D Element on the index returned
    by the supplied function (by default the argmax). Element
    must have only one key dimension, which must be cyclic
    and declare the range. If set to be relative absolute
    cyclic values are discarded and object will be indexed
    by the cyclic range symmetric around the center.
    """

    function = param.Callable(default=np.argmax, doc="""
        Function to compute the index of the center value.""")

    dimension = param.ClassSelector(class_=(int, str, Dimension),
                                    default=1, doc="""
        The dimension to apply the function on to compute the center index.""")

    relative = param.Boolean(default=True, doc="""
        Whether the x-values should be relative or absolute.""")

    def _process(self, element, key=None):
        dim = element.get_dimension(0)
        if len(element.kdims) != 1:
            raise Exception("Can only center elements with one key dimension.")
        if not dim.cyclic:
            raise Exception("Cannot center non cyclic Element.")
        elif dim.range[0] is None or dim.range[1] is None:
            raise Exception("Must define cyclic range to center Element.")

        # Compute required shift and x-values
        length = len(element)
        center = element.dimension_values(self.dimension).argmax()
        shift = int(length/2) - center
        cyclic_range = dim.range[1] - dim.range[0]
        half_range = cyclic_range/2.
        xvals = element.dimension_values(0)
        offset = 0 if self.p.relative else xvals[center]
        new_range = (offset-half_range, offset+half_range)

        # Roll the remaining dimensions
        dim = dim(range=new_range)
        data = {dim.name: np.linspace(dim.range[0], dim.range[1], length)}
        for d in element.dimensions()[1:]:
            rolled = np.roll(element.dimension_values(d)[:-1], shift)
            data[d.name] = np.concatenate([rolled, [rolled[0]]])
        return element.clone(data, kdims=[dim])



class toHCS(ElementOperation):
    """
    Hue-Confidence-Strength plot.

    Accepts an overlay containing either 2 or 3 layers. The first two
    layers are hue and confidence and the third layer (if available)
    is the strength channel.
    """

    output_type = RGB

    S_multiplier = param.Number(default=1.0, bounds=(0.0,None), doc="""
        Post-normalization multiplier for the strength value.

        Note that if the result is outside the bounds 0.0-1.0, it will
        be clipped. """)

    C_multiplier = param.Number(default=1.0, bounds=(0.0,None), doc="""
        Post-normalization multiplier for the confidence value.

        Note that if the result is outside the bounds 0.0-1.0, it will
        be clipped.""")

    flipSC = param.Boolean(default=False, doc="""
        Whether to flip the strength and confidence channels""")

    group = param.String(default='HCS', doc="""
        The group string for the output (an RGB element).""")

    def _process(self, overlay, key=None):

        normfn = raster_normalization.instance()
        if self.p.input_ranges:
            overlay = normfn.process_element(overlay, key, *self.p.input_ranges)
        else:
            overlay = normfn.process_element(overlay, key)

        hue, confidence = overlay.get(0), overlay.get(1)
        strength_data = overlay.get(2).data if (len(overlay) == 3) else np.ones(hue.shape)

        hue_data = hue.data
        hue_range = hue.value_dimensions[0].range
        if (not hue.value_dimensions[0].cyclic) or (None in hue_range):
             raise Exception("The input hue channel must be declared cyclic with a defined range.")
        if hue.shape != confidence.shape:
            raise Exception("Cannot combine input Matrices with different shapes.")

        (h,s,v)= (hue_data,
                  (confidence.data * self.p.C_multiplier).clip(0.0, 1.0),
                  (strength_data * self.p.S_multiplier).clip(0.0, 1.0))

        if self.p.flipSC:
            (h,s,v) = (h,v,s.clip(0,1.0))

        return RGB(np.dstack(hsv_to_rgb(h,s,v)),
                   bounds = self.get_overlay_bounds(overlay),
                   label =  self.get_overlay_label(overlay),
                   group =  self.p.group)



class decode_feature(ElementOperation):
    """
    Estimate the value of a feature from the current activity pattern
    on a sheet and a preference map of the sheet. The activity and
    preference should be supplied as an Overlay or a ViewMap of Overlay
    objects.

    If weighted_average is False, the feature value returned is the
    value of the preference_map at the maximally active location.

    If weighted_average is True, the feature value is estimated by
    weighting the preference_map by the current activity level, and
    averaging the result across all units in the sheet.

    For instance, if preference_map is OrientationPreference (a cyclic
    quantity), then the result will be the vector average of the
    activated orientations.  For an orientation map this value should
    be an estimate of the orientation present on the input.

    Record the V1 activity in response to an oriented Gaussian and the
    orientation preference as follows and overlay them::

       act = measure_response(pattern_generator=imagen.Gaussian(),
                              durations=[0.1, 0.5, 1.0])['GaussianResponse']['V1']
       pref = measure_or_pref()['OrientationPreference']['V1'].last
       ActivityPreferenceStack = act * pref

       decode_feature(ActivityPreferenceStack)
    """

    reference_value = param.Number(default=None, allow_None=True, doc="""
        Allows specifying a reference value, to compute the error of the
        decoded value.""")

    weighted_average = param.Boolean(default=True, doc="""
        Decode as vector average if True or by maximal responding unit.""")

    def _process(self, overlay, key=None):
        preference = None; selectivity = None; activity = None;
        for sv in overlay:
            if sv.label.endswith('Preference'):
                preference = sv
                fname = sv.label[:-len('Preference')]
            if sv.label.endswith('Selectivity'):
                selectivity = sv
            if sv.label.endswith('Response') or sv.label.endswith('Activity'):
                activity = sv

        if preference is None or activity is None:
            raise ValueError("DecodeFeature requires overlay with response/"
                             "activity and preference as input.")
        if selectivity is None:
            selectivity = Image(np.ones(preference.data.shape),
                                 preference.bounds)

        cr = preference.cyclic_range
        cyclic = False if cr is None else True
        range = (0, 1.0 if cr is None else cr)

        d = Distribution(range, cyclic)
        for (p, a, s) in zip(preference.data.ravel(), activity.data.ravel(), selectivity.data.ravel()):
            d.add({p: a*s})

        decoded_label = "Decoded " + fname.strip()

        res = DSF_WeightedAverage()(d) if self.p.weighted_average else DSF_MaxValue()(d)
        decoded_value = res['']['preference']
        ret = {decoded_label: decoded_value}

        if self.p.reference_value is not None:
            difference = abs(decoded_value - self.p.reference_value)
            difference = cr - difference if difference >= 0.5*cr else difference
            if cyclic: difference = difference % cr
            ret.update({"Decoding Error": difference})

        return [ItemTable(ret)]

