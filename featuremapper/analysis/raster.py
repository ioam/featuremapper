"""
Useful ElementOperations over Raster elements.
"""

import numpy as np

import param
from holoviews import CompositeOverlay, BoundingBox, Dimension
from holoviews import Image, Curve, VectorField
from holoviews.core import ElementOperation
from holoviews.operation.normalization import raster_normalization


class fft_power(ElementOperation):
    """
    Given a Image element, compute the power of the 2D Fast Fourier
    Transform (FFT).
    """

    output_type = Curve

    max_power = param.Number(default=1.0, doc="""
    The maximum power value of the output power spectrum.""")

    group = param.String(default='FFT Power', doc="""
    The group assigned to the output power spectrum.""")


    def _process(self, matrix, key=None):
        normfn = raster_normalization.instance()
        if self.p.input_ranges:
            matrix = normfn.process_element(matrix, key, *self.p.input_ranges)
        else:
            matrix = normfn.process_element(matrix, key)

        fft_spectrum = abs(np.fft.fftshift(np.fft.fft2(matrix.data - 0.5, s=None, axes=(-2, -1))))
        fft_spectrum = 1 - fft_spectrum # Inverted spectrum by convention
        zero_min_spectrum = fft_spectrum - fft_spectrum.min()
        spectrum_range = fft_spectrum.max() - fft_spectrum.min()
        spectrum = (self.p.max_power * zero_min_spectrum) / spectrum_range

        l, b, r, t = matrix.bounds.lbrt()
        density = matrix.xdensity
        bounds = BoundingBox(radius=(density/2)/(r-l))

        return Image(spectrum, bounds=bounds, label=matrix.label, group=self.p.group)



class vectorfield(ElementOperation):
    """
    Given a Image with a single channel, convert it to a VectorField
    object at a given spatial sampling interval. The values in the
    Image are assumed to correspond to the vector angle in radians
    and the value is assumed to be cyclic.

    If supplied with an Overlay, the second sheetview in the overlay
    will be interpreted as the third vector dimension.
    """

    output_type = VectorField

    rows = param.Integer(default=10, doc="""
       The number of rows in the vector field.""")

    cols = param.Integer(default=10, doc="""
       The number of columns in the vector field.""")

    group = param.String(default='Vectors', doc="""
       The group assigned to the output vector field.""")


    def _process(self, view, key=None):

        if isinstance(view, CompositeOverlay) and len(view) >= 2:
            radians, lengths = view[0], view[1]
        else:
            radians, lengths = view, None

        cyclic_dim = radians.vdims[0]
        if not cyclic_dim.cyclic:
            raise Exception("First input Image must be declared cyclic")

        l, b, r, t = radians.bounds.lbrt()
        X, Y = np.meshgrid(np.linspace(l, r, self.p.cols+2)[1:-1],
                           np.linspace(b, t, self.p.rows+2)[1:-1])

        vector_data = []
        for x, y in zip(X.flat, Y.flat):
            components = (x,y, radians[x,y])
            if lengths is not None:
                components += (lengths[x,y],)

            vector_data.append(components)

        vdims = [Dimension('Angle', cyclic=True, range=cyclic_dim.range)]
        if lengths is not None:
            vdims.append(Dimension('Magnitude'))
        return VectorField(np.array(vector_data), label=radians.label, group=self.p.group,
                           vdims=vdims)

