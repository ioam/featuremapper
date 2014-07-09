"""
Color conversion utilities.
The two main end-objects are:
  * ColorSpace:
       Create a ColorSpace object and then use its convert(from, to, what) method to perform color conversion between colorspaces.

  * FeatureColorConverter:
       Defines color spaces to be used in images, receptors (simulated retinas) and analysis (generally, HSV for Hue Analysis).
       An object is created by default, and its color spaces can be updated. It is still possible to create custom objects, anyway.
       Customization of the object can be done as, eg,

         featuremapper.color_conversion.set_image_colorspace("XYZ") # format of the dataset used XYZ, LMS
         featuremapper.color_conversion.set_receptor_responses("RGB") # possible values are RGB and LMS.
         featuremapper.color_conversion.set_analysis_space("HSV") # HSV, LCH

"""

from math import pi

import numpy

import param

import copy

import colorsys

# SPG: warning, this depends on Topographica !
#from topo.misc.inlinec import inline


## Old CEB colorfns.py file contents


## return Ma where M is 3x3 transformation matrix, for each pixel
def _threeDdot_dumb(M,a):
    result = numpy.empty(a.shape,dtype=a.dtype)

    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            A = numpy.array([a[i,j,0],a[i,j,1],a[i,j,2]]).reshape((3,1))
            L = numpy.dot(M,A)
            result[i,j,0] = L[0]
            result[i,j,1] = L[1]
            result[i,j,2] = L[2]

    return result

def _threeDdot_faster(M,a):
    swapped = a.swapaxes(0,2)
    shape = swapped.shape
    result = numpy.dot(M,swapped.reshape((3,-1)))
    result.shape = shape
    b = result.swapaxes(2,0)
    # need to do asarray to ensure dtype?
    return numpy.asarray(b,dtype=a.dtype)

# CB: probably could make a faster version if do aM instead,
# e.g. something like (never tested):
#def _threeDdot(M,a):
#    shape = a.shape
#    result = np.dot(a.reshape((-1,3)),M)
#    result.shape = shape
#    return result

threeDdot = _threeDdot_faster

def _abc_to_def_array(ABC,fn):
    shape = ABC[:,:,0].shape
    dtype = ABC.dtype
    
    DEF = numpy.zeros(ABC.shape,dtype=dtype)

    for i in range(shape[0]):
        for j in range(shape[1]):
            DEF[i,j,0],DEF[i,j,1],DEF[i,j,2]=fn(ABC[i,j,0],ABC[i,j,1],ABC[i,j,2])

    return DEF
    

def _rgb_to_hsv_array(RGB):
    """
    Equivalent to colorsys.rgb_to_hsv, except expects array like :,:,3
    """
    return _abc_to_def_array(RGB,colorsys.rgb_to_hsv)


def _hsv_to_rgb_array(HSV):
    """
    Equivalent to colorsys.hsv_to_rgb, except expects array like :,:,3
    """
    return _abc_to_def_array(HSV,colorsys.hsv_to_rgb)




rgb_to_hsv = _rgb_to_hsv_array #_opt
hsv_to_rgb = _hsv_to_rgb_array #_opt


KAP = 24389/27.0
EPS = 216/24389.0


def xyz_to_lab(XYZ,wp):

    X,Y,Z = numpy.dsplit(XYZ,3)
    xn,yn,zn = X/wp[0], Y/wp[1], Z/wp[2]

    def f(t):
        t = t.copy() # probably unnecessary! 
        t_eps = t>EPS
        t_not_eps = t<=EPS
        t[t_eps] = numpy.power(t[t_eps], 1.0/3)
        t[t_not_eps] = (KAP*t[t_not_eps]+16.0)/116.
        return t
            
    fx,fy,fz = f(xn), f(yn), f(zn)
    L = 116*fy - 16
    a = 500*(fx - fy)
    b = 200*(fy - fz)

    return numpy.dstack((L,a,b))


def lab_to_xyz(LAB,wp):

    L,a,b = numpy.dsplit(LAB,3)
    fy = (L+16)/116.0
    fz = fy - b / 200.0
    fx = a/500.0 + fy

    def finv(y):
        y =copy.copy(y) # CEBALERT: why copy?
        eps3 = EPS**3 
        return numpy.where(y > eps3,
                           numpy.power(y,3),
                           (116*y-16)/KAP)

    xr, yr, zr = finv(fx), finv(fy), finv(fz)
    return numpy.dstack((xr*wp[0],yr*wp[1],zr*wp[2]))


def lch_to_lab(LCH):
    L,C,H = numpy.dsplit(LCH,3)
    return numpy.dstack( (L,C*numpy.cos(H),C*numpy.sin(H)) )

def lab_to_lch(LAB):
    L,A,B = numpy.dsplit(LAB,3)
    range_ = 2*pi
    x = numpy.arctan2(B,A)
    return numpy.dstack( (L, numpy.hypot(A,B), fmod(x + 2*range_*(1-floor(x/(2*range_))), range_) ) )
    #return numpy.dstack( (L, numpy.hypot(A,B), wrap(0,2*pi,numpy.arctan2(B,A))) )




def xyz_to_lch(XYZ,whitepoint):
    return lab_to_lch(xyz_to_lab(XYZ,whitepoint))

def lch_to_xyz(LCH,whitepoint):
    return lab_to_xyz(lch_to_lab(LCH),whitepoint)

## End of old colorfns.py file contents





# started from 
# http://projects.scipy.org/scipy/browser/trunk/Lib/sandbox/image/color.py?rev=1698

whitepoints = {'CIE A': ['Normal incandescent', 0.4476, 0.4074],
               'CIE B': ['Direct sunlight', 0.3457, 0.3585],
               'CIE C': ['Average sunlight', 0.3101, 0.3162],
               'CIE E': ['Normalized reference', 1.0/3, 1.0/3],
               'D50' : ['Bright tungsten', 0.3457, 0.3585],
               'D55' : ['Cloudy daylight', 0.3324, 0.3474],
               'D65' : ['Daylight', 0.312713, 0.329016],
               'D75' : ['?', 0.299, 0.3149],
               'D93' : ['low-quality old CRT', 0.2848, 0.2932]
               }

def triwhite(chrwhite):
    x,y = chrwhite
    X = float(x) / y
    Y = 1.0
    Z = (1-x-y)/y
    return X,Y,Z

for key in whitepoints.keys():
    whitepoints[key].append(triwhite(whitepoints[key][1:]))


transforms = {}


# CEBALERT: add reference
transforms = {}
transforms['D65'] = sD65 = {}

sD65['rgb_from_xyz'] = numpy.array([[3.2410,-1.5374,-0.4986],
                                       [-0.9692,1.8760,0.0416],
                                       [0.0556,-0.204,1.0570]])
sD65['xyz_from_rgb'] = numpy.array([[ 0.41238088,  0.35757284,  0.1804523 ],
                                       [ 0.21261986,  0.71513879,  0.07214994],
                                       [ 0.0193435 ,  0.11921217,  0.95050657]])


# Guth (1980) - SP; L, M, and S normalized to one)
sD65['lms_from_xyz'] = numpy.array([[0.2435, 0.8524, -0.0516],
                                        [-0.3954, 1.1642, 0.0837],
                                        [0, 0, 0.6225]])

sD65['xyz_from_lms'] = numpy.array([[  1.87616336e+00,  -1.37368291e+00,   3.40220544e-01],
                                        [  6.37205799e-01,   3.92411765e-01,   5.61517442e-05],
                                        [  0.00000000e+00,   0.00000000e+00,   1.60642570e+00]])



### Make LCH like other spaces (0,1)

Lmax = 100.0
Cmax = 360.0 # ? CEBALERT: A,B typically -127 to 128 (wikipedia...), so 360 or so max for C?
Hmax = 2*pi

def xyz_to_lch01(XYZ, whitepoint):
    L,C,H = numpy.dsplit(xyz_to_lch(XYZ,whitepoint),3)
    L/=Lmax
    C/=Cmax
    H/=Hmax
    return numpy.dstack((L,C,H))
    
def lch01_to_xyz(LCH, whitepoint):
    L,C,H = numpy.dsplit(LCH,3)
    L*=Lmax
    C*=Cmax
    H*=Hmax
    return lch_to_xyz(numpy.dstack((L,C,H)),whitepoint)

###


# This started off general but ended up being useful only
# for the specific transforms I wanted to do.
class ColorSpace(param.Parameterized):
    """
    Low-level color conversion. The 'convert' method handle color conversion to and from (and through) XYZ,
    and supports RGB, LCH, LMS and HSV.
    """

    whitepoint = param.String(default='D65')

    transforms = param.Dict(default=transforms)

    input_limits = param.NumericTuple((0.0,1.0))

    output_limits = param.NumericTuple((0.0,1.0))

    output_clip = param.ObjectSelector(default='silent',
                                       objects=['silent','warn','error','none'])

    dtype = param.Parameter(default=numpy.float32)

    def convert(self, from_, to, what):
        """
        Convert "what" from "from_" colorpace to "to" colorspace. Eg, convert("rgb", "hsv", X)
        """
        if(from_.lower()==to.lower()):
            return what


        # Check if there exist an optimized function that performs from_to_to conversion
        direct_conversion = '%s_to_%s'%(from_.lower(),to.lower())
        if( hasattr(self, direct_conversion ) ):
            fn = getattr(self, direct_conversion)
            return fn(what)

        from_to_xyz = getattr(self, '%s_to_xyz'%(from_.lower()) )
        xyz_to_to = getattr(self, 'xyz_to_%s'%(to.lower()) )

        return xyz_to_to( from_to_xyz(what) )


    def _triwp(self):
        return whitepoints[self.whitepoint][3]

    def _get_shape(self,a):        
        if hasattr(a,'shape') and a.ndim>0: # i.e. really an array, I hope
            return a.shape
        else:
            # also support e.g. tuples
            try:
                length = len(a)
                return (length,)
            except TypeError:            
                return None

    def _put_shape(self,a,shape):
        if shape is None:
            return self.dtype(a)
        else:
            a.shape = shape
            return a
    
    def _prepare_input(self,a,min_,max_):
        in_shape = self._get_shape(a)
        a = numpy.array(a,copy=False,ndmin=3,dtype=self.dtype)
        if a.min()<min_ or a.max()>max_:
            raise ValueError('Input out of limits')
        return a, in_shape
        
    def _clip(self,a,min_limit,max_limit,action='silent'):
        if action=='none':
            return
        
        if action=='error':
            if a.min()<min_limit or a.max()>max_limit:
                raise ValueError('(%s,%s) outside limits (%s,%s)'%(a.min(),a.max(),min_limit,max_limit))
        elif action=='warn':
            if a.min()<min_limit or a.max()>max_limit:                
                self.warning('(%s,%s) outside limits (%s,%s)'%(a.min(),a.max(),min_limit,max_limit))

        a.clip(min_limit,max_limit,out=a)
                    
    def _threeDdot(self,M,a):
        # b = Ma        
        a, in_shape = self._prepare_input(a,*self.input_limits)
        b = threeDdot(M,a)
        self._clip(b,*self.output_limits,action=self.output_clip)
        self._put_shape(b,in_shape)
        return b

    def _ABC_to_DEF_by_fn(self,ABC,fn,*fnargs):
        ABC, in_shape = self._prepare_input(ABC,*self.input_limits)
        DEF = fn(ABC,*fnargs)
        self._clip(DEF,*self.output_limits,action=self.output_clip)
        self._put_shape(DEF, in_shape)
        return DEF

    # CEBALERT: I meant to wrap these to use paramoverrides (e.g. to
    # allow rgb_to_hsv(RGB,output_clip='error') ) but never got round
    # to it.
    # CEBALERT: could cut down boilerplate by generating.

    ##  TO XYZ:     RGB, LCH, LMS, HSV(passing through RGB)
    def rgb_to_xyz(self,RGB):
        return self._threeDdot(
            self.transforms[self.whitepoint]['xyz_from_rgb'], RGB)

    def lch_to_xyz(self,LCH):
        return self._ABC_to_DEF_by_fn(LCH,lch01_to_xyz,self._triwp())

    def lms_to_xyz(self,LMS):
        return self._threeDdot(
            self.transforms[self.whitepoint]['xyz_from_lms'], LMS)

    def hsv_to_xyz(self,HSV):
        return self.rgb_to_xyz(self.hsv_to_rgb(HSV))



    ##  XYZ TO:     RGB, LCH, LMS, HSV(passing through RGB)
        
    def xyz_to_rgb(self,XYZ):
        return self._threeDdot(
            self.transforms[self.whitepoint]['rgb_from_xyz'], XYZ)

    def xyz_to_lch(self, XYZ):
        return self._ABC_to_DEF_by_fn(XYZ,xyz_to_lch01,self._triwp())

    def xyz_to_lms(self,XYZ):
        return self._threeDdot(
            self.transforms[self.whitepoint]['lms_from_xyz'], XYZ)

    def xyz_to_hsv(self, XYZ):
        return self.rgb_to_hsv( self.xyz_to_rgb(XYZ) ) 



    ## Optimized
    @staticmethod
    def _gamma_rgb(RGB):
        return 12.92*RGB*(RGB<=0.0031308) + ((1+0.055)*RGB**(1/2.4) - 0.055) * (RGB>0.0031308)

    @staticmethod
    def _ungamma_rgb(RGB):
        return RGB/12.92*(RGB<=0.04045) + (((RGB+0.055)/1.055)**2.4) * (RGB>0.04045)

    # linear rgb to hsv
    def rgb_to_hsv(self,RGB):
        gammaRGB = self._gamma_rgb(RGB)
        return self._ABC_to_DEF_by_fn(gammaRGB,rgb_to_hsv)

    # hsv to linear rgb
    def hsv_to_rgb(self,HSV):
        gammaRGB = self._ABC_to_DEF_by_fn(HSV,hsv_to_rgb)
        return self._ungamma_rgb(gammaRGB)

    ### for display
    def hsv_to_gammargb(self,HSV):
        # hsv is already specifying gamma corrected rgb
        return self._ABC_to_DEF_by_fn(HSV,hsv_to_rgb)

    def lch_to_gammargb(self,LCH):
        return self._gamma_rgb(self.lch_to_rgb(LCH))

    def lms_to_lch(self,LCH):
        lch_to_xyz






def _swaplch(LCH):
    # brain not working
    try:
        L,C,H = numpy.dsplit(LCH,3)
        return numpy.dstack((H,C,L))
    except:
        L,C,H = LCH
        return H,C,L




## SPG: this is the only object that needs to be accessed from the outside.
class FeatureColorConverter(param.Parameterized):
    """
    Color conversion class designed to support color space transformations along a pipeline common
    in color vision modelling : image -> receptors (retinas) -> [higher stages] -> analysis
    """

    # CEBALERT: should be class selector
    colorspace = param.Parameter(default=ColorSpace())

    image_space = param.ObjectSelector(
        default='XYZ', 
        objects=['XYZ', 'LMS'],
        doc="""
        Color space in which images are encoded.
        """) # CEBALERT: possibly add sRGB?

    receptor_space = param.ObjectSelector(
        default='RGB',
        objects=['RGB','LMS'],
        doc="""
        Color space to which images are transformed to provide input to later stages of processing.
        """)

    analysis_space = param.ObjectSelector(
        default='HSV',
        objects=['HSV','LCH'],
        doc="""
        Color space in which analysis is performed.
        """)

    # CEBALERT: should be classselector
    display_sat = param.Number(default=1.0)
    display_val = param.Number(default=1.0)

    swap_polar_HSVorder = {
        'HSV': lambda HSV: HSV,
        'LCH': _swaplch }
    

    def set_receptor_space(self, receptor_colorspace):
        """
        Set the receptor space.
        """
        self.receptor_space = receptor_colorspace

    def set_image_space(self, dataset_colorspace):
        """
        Set the color space of input images.
        """
        self.image_space = dataset_colorspace

    def set_analysis_space(self, analysis_colorspace):
        """
        Set the analysis space to be used by the system.
        """
        self.analysis_space = analysis_colorspace


    def image2receptors(self,i):
        """
        Transform input images onto the specified receptor color space.

        """
        return self.colorspace.convert(self.image_space, self.receptor_space, i)

    def receptors2analysis(self,r):
        """
        Transform receptor space inputs to the analysis color space.
        """
        a = self.colorspace.convert(self.receptor_space, self.analysis_space, r)
        return self.swap_polar_HSVorder[self.analysis_space](a)

    def analysis2receptors(self,a):
        """
        Convert back from the analysis color space to the receptor's.
        """
        a = self.swap_polar_HSVorder[self.analysis_space](a)        
        return self.colorspace.convert(self.analysis_space, self.receptor_space, a)

    def analysis2display(self,a):
        """
        Utility conversion function that transforms data from the analysis color space to the display space (currently hard-set to RGB) for visualization.
        """
        a = self.swap_polar_HSVorder[self.analysis_space](a)
        return self.colorspace.convert(self.analysis_space.lower(), 'gammargb', a)
    
    def jitter_hue(self,a,amount):
        a[:,:,0] += amount
        a[:,:,0] %= 1.0

    def multiply_sat(self,a,factor):
        a[:,:,1] *= factor




# Initialize FeatureMapper.color_conversion object! (FeatureColorConverter)
default_receptor_colorspace='RGB'

default_dataset_colorspace = 'XYZ'
default_analysis_colorspace = 'HSV'

color_conversion = FeatureColorConverter(
    analysis_space = default_analysis_colorspace,
    image_space = default_dataset_colorspace)

color_conversion.set_receptor_space(default_receptor_colorspace)




__all__ = [
    "ColorSpace",
    "FeatureColorConverter"
]


