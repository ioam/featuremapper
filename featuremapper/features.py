import numpy as np

import param

from holoviews import Dimension

from .distribution import DistributionStatisticFn, DSF_WeightedAverage



class Feature(Dimension):
    """
    Specifies several parameters required for generating a map of one input
    feature.
    """

    compute_fn = param.Callable(default=None, doc="""
        If non-None, a function that when given a list of other parameter
        values, computes and returns the value for this feature.""")

    preference_fn = param.ClassSelector(DistributionStatisticFn, allow_None=True,
                                        default=DSF_WeightedAverage(), doc="""
        Function that will be used to analyze the distributions of unit response
        to this feature.""")

    steps = param.Integer(default=0, doc="""
        Number of steps, between lower and upper range value, to be presented.""")

    offset = param.Number(default=0.0, doc="""
        Offset to add to the values for this feature""")

    _init = False # Allows creation of default Features in the file

    definitions = {}

    def _init_values(self):
        if len(self.values):
            self.values = self.values if self.offset == 0 \
                else [v + self.offset for v in self.values]
            if self.range in [(0, 0), (None, None)]:
                self.range = (min(self.values), max(self.values))
        else:
            if self.range == (0, 0):
                raise ValueError('The range or values must be specified.')
            low_bound, up_bound = self.range
            values = np.linspace(low_bound, up_bound, self.steps, not self.cyclic) + self.offset
            self.values = list(values % (up_bound - low_bound) if self.cyclic else values)

        
    def __init__(self, name, **params):
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
        super(Feature, self).__init__(name, **params)
        if self._init:
            self._init_values()

        Feature.definitions[self.name] = self


def degree_formatter(x, pos=None):
    return str(int(np.rad2deg(x)))


# Basic features
Float   = Feature("float", type=float)
Integer = Feature("float", type=int)


# Cyclic features
Cyclic              = Float("cyclic", cyclic=True, unit="rad")
Hue                 = Cyclic("hue", range=(0.0, 1.0))

FullCycle           = Cyclic("full cycle",
                             value_format=degree_formatter,
                             range=(0, 2*np.pi),
                             unit="$^{\circ}$")
Angle               = FullCycle("Angle")
Direction           = FullCycle("Direction")
Phase               = FullCycle("Phase")
PhaseDisparity      = FullCycle("PhaseDisparity")

HalfCycle           = Cyclic("half cycle", range=(0, np.pi),
                             value_format=degree_formatter,
                             unit="$^{\circ}$")
Orientation         = HalfCycle("Orientation")
OrientationSurround = HalfCycle("OrientationSurround",
                                range=(-np.pi/2., np.pi/2.))

# Non-cyclic features
Frequency    = Float("Frequency", unit="cycles per unit distance")
Presentation = Integer("Presentation")
Size         = Float("Size", unit="Diameter")
Scale        = Float("Scale")
X            = Float("X")
Y            = Float("Y")

# Complex features
Contrast         = Float("Contrast", range=(0, 100), unit="%")
ContrastSurround = Contrast("ContrastSurround", preference_fn=None)
Ocular           = Integer("Ocular")
Speed            = Float("Speed")

# Time features
Time     = Dimension("Time", type=param.Dynamic.time_fn.time_type)
Duration = Time("Duration")

Feature._init = True # All Features created externally have to supply range or values
