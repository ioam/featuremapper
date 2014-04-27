import sys, time, math, uuid

import numpy as np
try:
    from IPython.core.display import clear_output
except:
    clear_output = None
    from nose.plugins.skip import SkipTest
    raise SkipTest("IPython extension requires IPython >= 0.12")

from collections import OrderedDict

import param
from dataviews import Stack, View, SheetView, SheetStack, CoordinateGrid, GridLayout
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

    def __call__(self, percentage):
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




class Container(object):
    """
    Container class for convenient attribute access used by
    ViewContainer. Containers have single level attribute access of
    Views by a label identifier and may be merged together using the
    update method.
    """

    def __init__(self, parent, source):
        self.__dict__['parent'] = parent
        self.__dict__['source'] = source
        self.__dict__['items'] = {}


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
        if isinstance(other, Container):
            for label, stack in other.items.items():
                self.add(label, stack)
        else:
            raise NotImplementedError('Containers may only be updated by other containers.')


    def __getitem__(self, label):
        return self.items[label]


    def __str__(self):
        if len(self.labels) == 0:
            return "Empty %s" % self.__class__.__name__
        return ", ".join(self.labels)


    def __contains__(self, name):
        return name in self.__dict__['items']




class ViewContainer(param.Parameterized):
    """
    A ViewContainer is a collection of Containers organised by a
    source identifier. This allows two level attribute access, first
    by source and then by label.

    ViewContainers are designed to provide convenient access to large
    collections of dataview objects.
    """

    container_class = Container

    def __init__(self, **kwargs):
        self._keys = []
        self.containers = {}
        super(ViewContainer, self).__init__(**kwargs)


    @property
    def sources(self):
        return set([src for src, l in self.keys()])


    def labels(self, source):
        return self.containers[source].labels


    def __getitem__(self, key):
        return self.containers[key]


    def __contains__(self, key):
        if isinstance(key, tuple):
            return key in self.keys()
        else:
            return key in self.containers


    def _create_container(self, source):
        """
        Create a new container with the given source label.
        """
        if source not in self.sources:
            container = self.container_class(self, source)
            setattr(self, source, container)
            self.containers[source] = container


    def _add_stack(self, source, label, stack):
        """
        Add the supplied stack to the container indexed by source with
        the given identifier label.
        """
        self._create_container(source)
        self.containers[source].add(label, stack)
        self._keys.append((source, label))


    def add(self, src, label, view, key=None):
        """
        Add the supplied view or stack to the ViewContainer indexed by
        source and label.
        """
        if (src, label) in self._keys:
            stack = self.containers[src][label]
            if isinstance(view, View):
                stack[key] = view
            elif isinstance(view, (Stack, CoordinateGrid, GridLayout)):
                stack.update(view)
            else:
                raise TypeError('ViewContainer currently only supports adding'
                                 'Views or Stacks')
        elif isinstance(view, (Stack, CoordinateGrid, GridLayout)):
            self._add_stack(src, label, view)
        else:
            raise TypeError('Must initialize new entry with a Stack.')


    def update(self, other):
        """
        Update the current ViewContainer with the contents of another
        ViewContainer.
        """
        if isinstance(other, ViewContainer):
            for src, grp in other.containers.items():
                self._create_container(src)
                self[src].update(grp)
        else:
            raise NotImplementedError('ViewContainers can currently only be'
                                      ' updated with another ViewContainer.')


    def keys(self):
        return self._keys


    def __str__(self):
        repr_str = ''
        for src, container in self.containers.items():
            repr_str += '%s: \n' % src
            repr_str += '   %r \n' % container

        return repr_str




class Reference(object):
    """
    A Reference object is a pointer to a view held by a Collector. A
    Reference allows data to be referenced when scheduling
    measurements before the data itself exists. For instance, this is
    useful to define the data to be input into an analysis
    ViewOperation.

    References compose in the same ways as GridLayouts to allow
    complex accessed to be defined ahead of time. In particular, the *
    operator and general indexing is supported.
    """

    def __init__(self, refs):
        self.refs = refs
        self.slices = dict.fromkeys(refs)


    def resolve(self, collector):
        composite_view = None
        for (src, label) in self.refs:
            slc = self.slices.get((src, label), None)
            if (src not in collector) or (label not in collector[src]):
                raise Exception("Data not available to resolve %s.%s" % (src,label))
            view = collector[src][label]
            view = view if slc is None else view[slc]
            if composite_view is None:
                composite_view = view
            else:
                composite_view = composite_view * view
        return composite_view

    def __getitem__(self, index):
        if len(self.refs) > 1:
            raise NotImplementedError
        src, label = self.refs[0]
        self.slices[(src, label)] = index
        return self


    def __mul__(self, other):
        return Reference(self.refs+other.refs)

    def __repr__(self):
        return "Reference(%r)" %  self.refs



class Collection(Container):
    """
    A Collection is a container that supports the creation of
    references to data not immediately accessible. This functionality
    is used by Collector to schedule analyses.
    """
    def __init__(self, parent, source):
        self.__dict__['references'] = []
        super(Collection, self).__init__(parent, source)


    def add(self, label, stack):
        super(Collection, self).add(label, stack)
        if label in self.__dict__['references']:
            self.__dict__['references'].remove(label)


    def __dir__(self):
        default_dir = dir(type(self)) + list(self.__dict__)
        return sorted(set(default_dir + self.__dict__['references']))


    def __setattr__(self, label, value):
        """
        Allows definitions to be added to the parent Collector via
        attribute name.
        """
        if isinstance(self.parent, Collector):
            self.parent.measurements[(self.source, label)] = value


    def __getattr__(self, label):
        if label in self.__dict__:
            return self.__dict__[label]
        return Reference([(self.source, label)])




class Collector(ViewContainer):
    """
    A Collector accumulates the output of arrays, measurements,
    analyses or miscellaneous View objects over time.
    """

    duration_hook = param.Callable(time.sleep, doc="""
       A callable that is given a duration value that is executed
       between blocks of measurements.""")

    progress_hook = param.Callable(default=ProgressBar, doc="""
       If set this should be an object with an empty constructor for
       initialization and an update method which is given with the
       completion percentage.""")

    time_fn = param.Callable(default=None, doc="""
        A callable that returns the time where the time may be the
        simulation time or wall-clock time. The time values are
        recorded by the Stack keys.""")


    container_class = Collection

    def __init__(self, **kwargs):
        self.durations = []
        self.measurements = OrderedDict()
        super(Collector, self).__init__(**kwargs)

        self._declare_collection('Analysis')


    def analyze(self, reference, analysis_fn, **kwargs):
        """
        A deferred analysis that applies the given analysis function
        with the supplied keywords.
        """
        return lambda : analysis_fn(reference.resolve(self), **kwargs)


    def collect(self, measurement, times=[], **kwargs):
        """
        Schedules collection of measurement output containers returned
        by a MeasurementCommands.
        """
        self.measurements[uuid.uuid4().hex] = dict(cmd=measurement,
                                                   kwargs=kwargs,
                                                   times=times)

    def clear(self):
        """"
        Clear all scheduled definitions
        """
        self.measurements = OrderedDict()


    def run(self, durations, cycles=1):
        try:
            self.durations = list(durations) * cycles
        except:
            self.durations = [durations] * cycles
        return self


    def __enter__(self):
        return self


    def __exit__(self, exc, *args):
        self(self.durations)
        self.durations = []
        self.clear()


    def __call__(self, durations, cycles=1):
        """
        Repeatedly advance time between measurement blocks and launch
        the scheduled measurements until the full set of durations is
        completed.
        """
        try:
            durations = list(durations) * cycles
        except:
            durations = [durations] * cycles

        self._progress_hook = self.progress_hook() if self.progress_hook else None
        total_duration = sum(durations)
        completed = 0.
        for i, duration in enumerate(durations):
            self.duration_hook(duration)
            completed += duration
            if self._progress_hook:
                self._progress_hook(completed/total_duration*100)
            for key, measurement in self.measurements.items():
                self._process_measurement(key, measurement, self.time_fn())


    def _process_measurement(self, key, measurement, time):
        """
        Processes scheduled measurement and updates the internal
        collections with the results.
        """
        type_error = False
        if isinstance(key, str):
            times = measurement['times']
            if not times or (time in times):
                container = measurement['cmd'](**measurement['kwargs'])
                self.update(container)
        elif isinstance(key, tuple):
            src, label = key
            if isinstance(measurement, np.ndarray):
                sv = SheetView(measurement.copy(), bounds=BoundingBox(), label=label)
                stack = SheetStack(((time,), sv), dimensions=[f.Time])
            elif callable(measurement) and not isinstance(measurement, GridLayout):
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
            elif isinstance(measurement, GridLayout):
                stack = measurement
            else:
                type_error = True
            if type_error:
                self.warning("Collector cannot process %s %s measurement " \
                  "result of type %s" % (src, label, str(type(measurement))))
            else:
                self.add(src, label, stack)


    def __getitem__(self, key):
        return self.__dict__['containers'][key]


    def _declare_collection(self, source):
        collections = self.__dict__['containers']
        assert source[0].isupper()
        collection = Collection(self, source)
        setattr(self, source, collection)
        collections[source] = collection
        return collection



    def __getattr__(self, source):
        """
        Access the collection with the given source label.
        """
        collections = self.__dict__['containers']
        if source in collections:
            collection = collections[source]
        else:
            collection = self._declare_collection(source)

        return collection


