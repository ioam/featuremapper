import sys, cmath, math
import numpy as np
from matplotlib import pyplot as plt

import param
from imagen.analysis import ViewOperation
from dataviews import Contours, Points

from dataviews.options import options, StyleOpts

__author__ = "Jean-Luc Stevens"


class WarningCounter(object):
    """
    A simple class to count 'divide by zero' and 'invalid value'
    exceptions to allow a suitable warning message to be generated.
    """
    def __init__(self):
        self.div_by_zeros = 0
        self.invalid_values = 0


    def __call__(self, errtype, flag):
        if errtype == "divide by zero":
            self.div_by_zeros += 1
        elif errtype == "invalid value":
            self.invalid_values += 1


    def warn(self):
        total_events = self.div_by_zeros + self.invalid_values
        if total_events == 0: return
        info = (total_events, self.div_by_zeros, self.invalid_values)
        self.div_by_zeros = 0;
        self.invalid_values = 0
        message = ("Warning: There were %d invalid intersection events:"
                   "\n\tNumpy 'divide by zero' events: %d"
                   "\n\tNumpy 'invalid value' events: %d\n")
        sys.stderr.write(message % info)



class PinwheelAnalysis(ViewOperation):
    """
    Given a SheetView or SheetStack of a cyclic feature preference,
    compute the position of all pinwheel singularities in the
    map. Optionally includes the contours for the real and imaginary
    components of the preference map used to determine the pinwheel
    locations.

    Returns the original SheetView input overlayed with a Points
    object containing the computed pinwheel locations and (optionally)
    Contours overlays including the real and imaginary contour lines
    respectively.
    """

    # TODO: Optional computation of pinwheel polarities.

    include_contours = param.Boolean(default=True, doc="""
      Whether or not to include the computed contours for the real and
      imaginary components of the map.""")


    def _process(self, view):
        [sheetview] = self.get_views(view, 'Preference')
        bounds = sheetview.bounds
        polar_map = self.polar_preference(sheetview.N.data)
        contour_info = self.polarmap_contours(polar_map, bounds)
        (re_contours, im_contours, intersections) = contour_info

        pinwheels = self.identify_pinwheels(*(re_contours, im_contours, intersections))
        pinwheels = Points(np.array(pinwheels), bounds,
                                label = sheetview.label+' Pinwheel')

        if self.p.include_contours:
            re_lines = Contours(re_contours, bounds,
                                label = sheetview.label+' Real')
            im_lines = Contours(im_contours, bounds,
                                  label = sheetview.label+' Imaginary')
            return [sheetview * re_lines * im_lines * pinwheels]
        else:
            return [sheetview * pinwheels]

    def polar_preference(self, pref):
        """
        Turns hue representation to polar representation.
        Hue representation uses values expected in the range 0-1.0
        """
        polarfn = lambda x: cmath.rect(1.0, x * 2 * math.pi)
        polar_vecfn = np.vectorize(polarfn)
        return polar_vecfn(pref)


    def normalize_polar_channel(self, polar_channel):
        """
        This functions normalizes an OR map (polar_channel) taking into
        account the region of interest (ROI). The ROI is specified by
        values set to 99. Note that this functionality is implemented to
        reproduce the experimental approach and has not been tested (not
        required for Topographica simulations)
        """

        def grad(r):
            (r_x, r_y) = np.gradient(r)
            (r_xx, r_xy) = np.gradient(r_x);
            (r_yx, r_yy) = np.gradient(r_y);
            return r_xx ** 2 + r_yy ** 2 + 2 * r_xy ** 2

        # Set ROI to 0 to ignore values of -99.
        roi = np.ones(polar_channel.shape)
        # In Matlab: roi(find(z==-99))=0
        roi[roi == -99] = 0

        fst_grad = grad(roi)
        snd_grad = grad(fst_grad)

        # Find non-zero elements in second grad and sets to unity
        snd_grad[snd_grad != 0] = 1
        # These elements now mask out ROI region (set to zero)
        roi[snd_grad == 1] = 0

        # Find the unmasked coordinates
        ind = (polar_channel != 99)
        # The complex abs of unmasked
        normalisation = np.mean(np.abs(polar_channel))
        # Only normalize with unmasked
        return polar_channel / normalisation


    def polarmap_contours(self, polarmap, bounds):
        """
        Identifies the real and imaginary contours in a polar map.
        Returns the real and imaginary contours as 2D vertex arrays
        together with the pairs of contours known to intersect. The
        coordinate system used is specified by the supplied bounds.

        Contour plotting requires origin='upper' for consistency with
        image coordinate system.
        """
        l,b,r,t = bounds.lbrt()
        # Convert to polar and normalise
        normalized_polar = self.normalize_polar_channel(polarmap)
        figure_handle = plt.figure()
        # Real component
        re_contours_plot = plt.contour(normalized_polar.real, 0, origin='upper',
                                       extent=[l,r,b,t])
        re_path_collections = re_contours_plot.collections[0]
        re_contour_paths = re_path_collections.get_paths()
        # Imaginary component
        im_contours_plot = plt.contour(normalized_polar.imag, 0, origin='upper',
                                       extent=[l,r,b,t])

        im_path_collections = im_contours_plot.collections[0]
        im_contour_paths = im_path_collections.get_paths()
        plt.close(figure_handle)

        intersections = [(re_ind, im_ind)
                         for (re_ind, re_path) in enumerate(re_contour_paths)
                         for (im_ind, im_path) in enumerate(im_contour_paths)
                         if im_path.intersects_path(re_path)]

        # Contour vertices  0.5 pixel inset. Eg. (0,0)-(48,48)=>(0.5, 0.5)-(47.5,  47.5)
        # Returned values will not therefore reach limits of 0.0 and 1.0
        re_contours = [self.remove_path_duplicates(re_path.vertices) for re_path in re_contour_paths]
        im_contours = [self.remove_path_duplicates(im_path.vertices) for im_path in im_contour_paths]
        return (re_contours, im_contours, intersections)


    def remove_path_duplicates(self, vertices):
        "Removes successive duplicates along a path of vertices."
        zero_diff_bools = np.all(np.diff(vertices, axis=0) == 0, axis=1)
        duplicate_indices, = np.nonzero(zero_diff_bools)
        return np.delete(vertices, duplicate_indices, axis=0)


    def find_intersections(self, contour1, contour2):
        """
        Vectorized code to find intersections between contours. All
        successive duplicate vertices along the input contours must be
        removed to help avoid division-by-zero errors.

        There are cases were no intersection exists (eg. parallel lines)
        where division by zero and invalid value exceptions occur. These
        exceptions should be caught as warnings: these edge cases are
        unavoidable with this algorithm and do not indicate that the
        output is erroneous.
        """
        # Elementwise min selection
        amin = lambda x1, x2: np.where(x1 < x2, x1, x2)
        # Elementwise max selection
        amax = lambda x1, x2: np.where(x1 > x2, x1, x2)
        # dstacks, checks True depthwise
        aall = lambda abools: np.dstack(abools).all(axis=2)
        # Uses delta (using np.diff) to find successive slopes along path
        slope = lambda line: (lambda d: d[:, 1] / d[:, 0])(np.diff(line, axis=0))
        # Meshgrids between both paths (x and y). One element sliced off end/beginning
        x11, x21 = np.meshgrid(contour1[:-1, 0], contour2[:-1, 0])
        x12, x22 = np.meshgrid(contour1[1:, 0], contour2[1:, 0])
        y11, y21 = np.meshgrid(contour1[:-1, 1], contour2[:-1, 1])
        y12, y22 = np.meshgrid(contour1[1:, 1], contour2[1:, 1])
        # Meshgrid of all slopes for both paths
        m1, m2 = np.meshgrid(slope(contour1), slope(contour2))
        m2inv = 1 / m2 # m1inv was not used.
        yi = (m1 * (x21 - x11 - m2inv * y21) + y11) / (1 - m1 * m2inv)
        xi = (yi - y21) * m2inv + x21 # (xi, yi) is intersection candidate
        # Bounding box type conditions for intersection candidates
        xconds = (amin(x11, x12) < xi, xi <= amax(x11, x12),
                  amin(x21, x22) < xi, xi <= amax(x21, x22) )
        yconds = (amin(y11, y12) < yi, yi <= amax(y11, y12),
                  amin(y21, y22) < yi, yi <= amax(y21, y22) )
        return xi[aall(xconds)], yi[aall(yconds)]


    def identify_pinwheels(self, re_contours, im_contours, intersections):
        """
        Locates the pinwheels from the intersection of the real and
        imaginary contours of of polar OR map.
        """
        warning_counter = WarningCounter()
        pinwheels = []
        np.seterrcall(warning_counter)
        for (re_ind, im_ind) in intersections:
            re_contour = re_contours[re_ind]
            im_contour = im_contours[im_ind]
            np.seterr(divide='call', invalid='call')
            x, y = self.find_intersections(re_contour, im_contour)
            np.seterr(divide='raise', invalid='raise')
            pinwheels += zip(x, y)

        warning_counter.warn()
        return pinwheels



options.Pinwheel_Points =    StyleOpts(color= 'w', marker= 'o', edgecolor= 'k')
options.Imaginary_Contours = StyleOpts(color= 'k')
options.Real_Contours =      StyleOpts(color= 'w')
