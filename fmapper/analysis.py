import param

from distribution import Distribution, DSF_WeightedAverage, DSF_MaxValue

def decode_feature(sheet, preference_map = "OrientationPreference", axis_bounds=(0.0,1.0), cyclic=True, weighted_average=True, cropfn=lambda(x):x):
    """
    Estimate the value of a feature from the current activity pattern on a sheet.

    The specified preference_map should be measured before this
    function is called.

    If weighted_average is False, the feature value returned is the
    value of the preference_map at the maximally active location.

    If weighted_average is True, the feature value is estimated by
    weighting the preference_map by the current activity level, and
    averaging the result across all units in the sheet.  The
    axis_bounds specify the allowable range of the feature values in
    the preference_map.  If cyclic is true, a vector average is used;
    otherwise an arithmetic weighted average is used.

    For instance, if preference_map is OrientationPreference (a cyclic
    quantity), then the result will be the vector average of the
    activated orientations.  For an orientation map this value should
    be an estimate of the orientation present on the input.

    If desired, a cropfn can be supplied that will narrow the analysis
    down to a specific region of the array; this function will be
    applied to the preference_map and to the activity array before
    decoding.  Examples:

    Decode whole area:

       decode_feature(topo.sim["V1"])

    Decode left half only:

       r,c = topo.sim["V1"].activity.shape
       lefthalf  = lambda(x): x[:,0:c/2]
       righthalf = lambda(x): x[:,c/2:]

       decode_feature(topo.sim["V1"], cropfn=lefthalf)

    """

    d = Distribution(axis_bounds, cyclic)

    if not (preference_map in sheet.views.maps):
        param.Parameterized.warning(preference_map + " should be measured before calling decode_feature.")
    else:
        v = sheet.views.maps[preference_map]
        for (p,a) in zip(cropfn(v.view()[0]).ravel(),
                         cropfn(sheet.activity).ravel()): d.add({p:a})

    res = DSF_WeightedAverage()(d) if weighted_average else DSF_MaxValue()(d)
    return res['']['preference']
