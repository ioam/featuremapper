"""
Metaparameter functions allow for the presentation of complex patterns that are
coordinated across different inputs (controlled via the input name) or
controlled by a higher-level parameter.

Current examples are transformations of contrast between the stimulus and the
background based on a contrast parameter, which can be set to implement
Michelson or Weber contrast or the coordinated presentation of input patterns
to each eye, which assigns the correct input pattern based on whether an input
source contains 'left' or 'right'.
"""

import numpy as np
from colorsys import hsv_to_rgb

import param
from imagen import Translator, Sweeper

class contrast2centersurroundscale(param.ParameterizedFunction):
    """
    Allows controlling the contrast in orientation contrast patterns, where the
    contrast of the central stimulus has to be set independently from the
    surround annulus and the background. Can be controlled with the
    contrast_parameter, which can be set to one of three values:
    'michelson_contrast', 'weber_contrast' or simply 'scale'.
    """

    contrast_parameter = param.String(default='weber_contrast')

    def __call__(self, inputs, features):
        if "contrastcenter" in features:
            if self.contrast_parameter == 'michelson_contrast':
                for g in inputs.itervalues():
                    g.offsetcenter = 0.5
                    g.scalecenter = 2*g.offsetcenter * g.contrastcenter/100.0

            elif self.contrast_parameter == 'weber_contrast':
                # Weber_contrast is currently only well defined for
                # the special case where the background offset is equal
                # to the target offset in the pattern type
                # SineGrating(mask_shape=Disk())
                for g in inputs.itervalues():
                    g.offsetcenter = 0.5   #In this case this is the offset of
                    # both the background and the sine grating
                    g.scalecenter = 2*g.offsetcenter * g.contrastcenter/100.0

            elif self.contrast_parameter == 'scale':
                for g in inputs.itervalues():
                    g.offsetcenter = 0.0
                    g.scalecenter = g.contrastcenter

        if "contrastsurround" in features:
            if self.contrast_parameter == 'michelson_contrast':
                for g in inputs.itervalues():
                    g.offsetsurround = 0.5
                    g.scalesurround = 2*g.offsetsurround * g.contrastsurround/100.0

            elif self.contrast_parameter == 'weber_contrast':
                # Weber_contrast is currently only well defined for
                # the special case where the background offset is equal
                # to the target offset in the pattern type
                # SineGrating(mask_shape=Disk())
                for g in inputs.itervalues():
                    g.offsetsurround = 0.5   #In this case this is the offset
                    # of both the background and the sine grating
                    g.scalesurround = 2*g.offsetsurround * g.contrastsurround/100.0

            elif self.contrast_parameter == 'scale':
                for g in inputs.itervalues():
                    g.offsetsurround = 0.0
                    g.scalesurround = g.contrastsurround



class contrast2scale(param.ParameterizedFunction):
    """
    Coordinates complex contrast values in single and compound patterns.
    To change the contrast behavior change the contrast_parameter to one of
    three values: 'michelson_contrast', 'weber_contrast' or simply 'scale'.
    """

    contrast_parameter = param.String(default='michelson_contrast')

    def __call__(self, inputs, features):

        if "contrast" in features:
            if self.contrast_parameter == 'michelson_contrast':
                for g in inputs.itervalues():
                    g.offset = 0.5
                    g.scale = 2*g.offset * g.contrast/100.0

            elif self.contrast_parameter == 'weber_contrast':
                # Weber_contrast is currently only well defined for
                # the special case where the background offset is equal
                # to the target offset in the pattern type
                # SineGrating(mask_shape=Disk())
                for g in inputs.itervalues():
                    g.offset = 0.5   #In this case this is the offset of both
                    # the background and the sine grating
                    g.scale = 2*g.offset * g.contrast/100.0

            elif self.contrast_parameter == 'scale':
                for g in inputs.itervalues():
                    g.offset = 0.0
                    g.scale = g.contrast



class direction2translation(param.ParameterizedFunction):
    """
    Coordinates the presentation of moving patterns. Currently
    supports an old and new motion model.
    """

    def __call__(self, inputs, features):
        if 'direction' in features:
            import __main__ as main

            if '_new_motion_model' in main.__dict__ and main.__dict__[
                '_new_motion_model']:
            #### new motion model ####

                for name in inputs:
                    inputs[name] = Translator(generator=inputs[name],
                                              direction=features['direction'],
                                              speed=features['speed'],
                                              reset_period=features['duration'])
            else:
            #### old motion model ####
                orientation = features['direction'] + np.pi/2

                for name in inputs.keys():
                    speed = features['speed']
                    try:
                        step = int(name[-1])
                    except:
                        if not hasattr(self, 'direction_warned'):
                            self.warning('Assuming step is zero; no input lag'
                                         ' number specified at the end of the'
                                         ' input sheet name.')
                            self.direction_warned = True
                        step = 0
                    speed = features['speed']
                    inputs[name] = Sweeper(generator=inputs[name], step=step,
                                           speed=speed)
                    setattr(inputs[name], 'orientation', orientation)



class phasedisparity2leftrightphase(param.ParameterizedFunction):
    """
    Coordinates phase disparity between two eyes, by looking for
    the keywords Left and Right in the input names.
    """

    def __call__(self, inputs, features):
        if "contrast" in features:
            temp_phase1 = features['phase'] - features['phasedisparity']/2.0
            temp_phase2 = features['phase'] + features['phasedisparity']/2.0
            for name in inputs.keys():
                if (name.count('Right')):
                    inputs[name].phase = temp_phase1 % 2*np.pi
                elif (name.count('Left')):
                    inputs[name].phase = temp_phase2 % 2*np.pi
                else:
                    if not hasattr(self, 'disparity_warned'):
                        self.warning('Unable to measure disparity preference, '
                                     'because disparity is defined only when '
                                     'there are inputs for Right and Left '
                                     'retinas.')
                        self.disparity_warned = True


class hue2rgbscale(param.ParameterizedFunction):
    """
    Coordinates hue between inputs with Red, Green or Blue in their
    name.
    """

    def __call__(self, inputs, features):
        if 'hue' in features:

            for name in inputs.keys():
                r, g, b = hsv_to_rgb(features['hue'], 1.0, 1.0)
                if (name.count('Red')):
                    inputs[name].scale = r
                elif (name.count('Green')):
                    inputs[name].scale = g
                elif (name.count('Blue')):
                    inputs[name].scale = b
                else:
                    # Note: this warning will be skipped if any retina
                    # has 'Red', 'Green', or 'Blue' in its
                    # name. E.g. if there is only a 'RedRetina', the
                    # warning will be skipped and a hue map will be
                    # measured without error.
                    if not hasattr(self, 'hue_warned'):
                        self.warning('Unable to measure hue preference,'
                                     ' because hue is defined only when'
                                     ' there are different input sheets'
                                     ' with names with Red, Green or Blue'
                                     ' substrings.')
                        self.hue_warned = True



class ocular2leftrightscale(param.ParameterizedFunction):
    """
    Coordinates patterns between two eyes, by looking for the
    keywords Left and Right in the input names.
    """

    def __call__(self, inputs, features):
        if "ocular" in features:
            for name in inputs.keys():
                if (name.count('Right')):
                    inputs[name].scale = 2 * features['ocular']
                elif (name.count('Left')):
                    inputs[name].scale = 2.0 - 2*features['ocular']
                else:
                    self.warning('Skipping input region %s; Ocularity is defined '
                                 'only for Left and Right retinas.' % name)

__all__ = ['contrast2centersurroundscale',
           'contrast2scale',
           'hue2rgbscale',
           'phasedisparity2leftrightphase',
           'direction2translation',
           'ocular2leftrightscale']
