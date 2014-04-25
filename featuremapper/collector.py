import sys, time, math

import numpy as np
try:
    from IPython.core.display import clear_output
except:
    clear_output = None
    from nose.plugins.skip import SkipTest
    raise SkipTest("IPython extension requires IPython >= 0.12")

from collections import OrderedDict

import param
from dataviews import Stack, View, SheetView, SheetStack, CoordinateGrid
from dataviews.boundingregion import BoundingBox

import features as f

class ProgressBar(param.Parameterized):
    """
    A simple text progress bar suitable for the IPython notebook.
    """

    width = param.Integer(default=70, doc="""
        The width of the progress bar in multiples of 'char'.""")

    fill_char = param.String(default='#', doc="""
        The character used to fill the progress bar.""")

    def __init__(self, **kwargs):
        super(ProgressBar,self).__init__(**kwargs)

    def update(self, percentage):
        " Update the progress bar to the given percentage value "
        if clear_output: clear_output()
        percent_per_char = 100.0 / self.width
        char_count = int(math.floor(percentage/percent_per_char) if percentage<100.0 else self.width)
        blank_count = self.width - char_count
        sys.stdout.write('\r' + "[%s%s] %0.1f%%" % (self.fill_char * char_count,
                             ' '*len(self.fill_char)*blank_count,
                             percentage))
        sys.stdout.flush()
        time.sleep(0.0001)



class ViewContainer(param.Parameterized):

    class group(object):
        """
        Container class for convenient attribute access.
        """

        def __init__(self, parent, source):
            self.__dict__['parent'] = parent
            self.__dict__['source'] = source
            self.__dict__['items'] = {}

        def __repr__(self):
            return ", ".join(self.labels)

        @property
        def labels(self):
            return self.items.keys()

        def add(self, label, stack):
            if label not in self.items:
                self.items[label] = stack
                self.__dict__[label] = stack
            else:
                self.items[label].update(stack)


        def update(self, other):
            if isinstance(other, ViewContainer.group):
                for label, stack in other.items.items():
                    self.add(label, stack)
            else:
                raise NotImplementedError('Groups can currently only be updated with another Groups.')

        def __getitem__(self, label):
            return self.items[label]

        def __setattr__(self, label, value):
            """
            Allows attribute specifications to be defined on the parent class,
            i.e. Collector.
            """
            if isinstance(self.parent, Collector):
                self.parent.measurements[(self.source, label)] = value


    def __init__(self, **kwargs):
        self._keys = []
        self.groups = {}

        super(ViewContainer, self).__init__(**kwargs)


    def __getitem__(self, key):
        return self.groups[key]


    def __contains__(self, key):
        if isinstance(key, tuple):
            return key in self.keys()
        else:
            return key in self.groups

    @property
    def sources(self):
        return set([src for src, l in self._keys])

    def labels(self, source):
        return self.groups[source].labels

    def _add_group(self, source):
        if source not in self.sources:
            group = self.group(self, source)
            setattr(self, source, group)
            self.groups[source] = group

    def _declare_entry(self, source, label, stack):
        self._add_group(source)
        group = self.groups[source]
        if label in group.items:
            raise Exception('Label already in group, why are we here!?!?!?!')
        group.add(label, stack)
        self._keys.append((source, label))


    def _access_entry(self, src, label):
        return self.groups[src][label]


    def add(self, src, label, view, key=None):
        if (src, label) in self._keys:
            stack = self._access_entry(src, label)
            if isinstance(view, View):
                stack[key] = view
            elif isinstance(view, (Stack, CoordinateGrid)):
                stack.update(view)
            else:
                raise TypeError('Collector currently only supports adding'
                                 'Views or Stacks')
        elif isinstance(view, (Stack, CoordinateGrid)):
            self._declare_entry(src, label, view)
        else:
            raise TypeError('Must initialize new entry with a Stack.')


    def update(self, other):
        if isinstance(other, ViewContainer):
            for src, grp in other.groups.items():
                self._add_group(src)
                self[src].update(grp)
        else:
            raise NotImplementedError('ViewContainers can currently only be'
                                      ' updated with another ViewContainer.')


    def keys(self):
        return self._keys

    def __repr__(self):
        repr_str = ''
        for src, group in self.groups.items():
            repr_str += '%s: \n' % src
            repr_str += '   %r \n' % group

        return repr_str



class Collector(ViewContainer):

    run_hook = param.Callable()

    progress_bar = param.Parameter(default=ProgressBar, doc="""
       If not None this should be a progress bar with an empty constructor
       and an update method (percentage complete).""")

    time_fn = param.Parameter(default=None)

    def __init__(self, **kwargs):
        self.durations = []
        self.measurements = OrderedDict()
        self._n = 0
        super(Collector, self).__init__(**kwargs)


    def run(self, durations, cycles=1):
        try:
            self.durations = list(durations) * cycles
        except:
            self.durations = [durations] * cycles
        return self


    def __enter__(self):
        return self

    def __exit__(self, exc, *args):
        self._progress_bar = self.progress_bar() if self.progress_bar else None
        self.advance(self.durations)


    def advance(self, durations):
        """
        Advances simulation time and launches scheduled measurements.
        """
        total_duration = sum(durations)
        completed = 0.
        for i, duration in enumerate(durations):
            self.run_hook(duration)
            if self._progress_bar:
                self._progress_bar.update(completed/total_duration*100)
            for key, measurement in self.measurements.items():
                self._process_measurement(key, measurement, self.time_fn())
            completed += duration


    def _process_measurement(self, key, measurement, time):
        """
        Processes scheduled measurement and updates the internal container
        with the results.
        """
        type_error = False
        if isinstance(key, int):
            times = measurement['times']
            if not times or (time in times):
                container = measurement['cmd'](**measurement['kwargs'])
                self.update(container)
        elif isinstance(key, tuple):
            src, label = key
            if isinstance(measurement, np.ndarray):
                sv = SheetView(measurement.copy(), bounds=BoundingBox(), label=label)
                stack = SheetStack(((time,), sv), dimensions=[f.Time])
            elif callable(measurement):
                measurement = measurement()
                if callable(measurement) or isinstance(measurement, np.ndarray):
                    self._process_measurement(key, measurement, time)
                    return None
                if isinstance(measurement, SheetView):
                    stack = SheetStack(((time,), measurement),
                                       dimensions=[f.Time])
                elif isinstance(measurement, (Stack, CoordinateGrid)):
                    stack = measurement
                else:
                    type_error = True
            else:
                type_error = True
            if type_error:
                print "Collector cannot process %s %s measurement " \
                  "result of type %s" % (src, label, str(type(measurement)))
            else:
                self.add(src, label, stack)


    def __getattr__(self, source):
        if source in self.__dict__:
            return self.__dict__[source]

        groups = self.__dict__['groups']
        if source in groups:
            group = groups[source]
        else:
            assert source[0].isupper()
            group = ViewContainer.group(self, source)
            setattr(self, source, group)
            groups[source] = group
        return group


    def collect(self, measurement, times=[], **kwargs):
        """
        Schedules collection of measurement Containers returned by
        a MeasurementCommand.
        """
        self.measurements[self._n] = dict(cmd=measurement, kwargs=kwargs,
                                          times=times)
        self._n += 1
