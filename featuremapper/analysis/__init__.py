import numpy as np

import param

from holoviews.core import ViewOperation
from holoviews.views import SheetMatrix, ItemTable

from featuremapper.distribution import Distribution, DSF_WeightedAverage, \
    DSF_MaxValue


class decode_feature(ViewOperation):
    """
    Estimate the value of a feature from the current activity pattern
    on a sheet and a preference map of the sheet. The activity and
    preference should be supplied as an Overlay or a HoloMap of Overlay
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
            selectivity = SheetMatrix(np.ones(preference.data.shape),
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

