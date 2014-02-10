import numpy as np

import param

from distribution import DistributionStatisticFn, DSF_WeightedAverage

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

    preference_fn = param.ClassSelector(DistributionStatisticFn, allow_None=True,
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

    unit = param.String(default="", doc="Unit string associated with the Feature")

    values = param.List(default=[], doc="""
        Explicit list of values for this feature, used in alternative to the
        range and step parameters""")

    _init = False # Allows creation of default Features in the file


    def _init_values(self):
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
        if self._init:
            self._init_values()
            

    def __call__(self, **params):
        settings = dict(self.get_param_values(onlychanged=True), **params)
        return self.__class__(**settings)

# Cyclic features
Cyclic              = Feature(cyclic=True, unit="rad")
Hue                 = Cyclic(name="hue", range=(0.0, 1.0))

FullCycle           = Cyclic(range=(0, 2*np.pi))
Angle               = FullCycle(name="angle")
Direction           = FullCycle(name="direction")
Phase               = FullCycle(name="phase")
PhaseDisparity      = FullCycle(name="phasedisparity")

HalfCycle           = Cyclic(range=(0, np.pi))
Orientation         = HalfCycle(name="orientation")
OrientationSurround = HalfCycle(name="orientationsurround")

# Non-cyclic features
Frequency    = Feature(name="frequency", unit="cycles per unit distance")
Presentation = Feature(name="presentation")
Size         = Feature(name="size", unit="Diameter")
Scale        = Feature(name="scale")
X            = Feature(name="x")
Y            = Feature(name="y")

# Complex features
Contrast         = Feature(name="contrast", range=(0, 100), unit="%")
ContrastSurround = Contrast(name="contrastsurround", preference_fn=None)
Ocular           = Feature(name="ocular")
Speed            = Feature(name="speed")

Feature._init = True # All Features created externally have to supply range or values
