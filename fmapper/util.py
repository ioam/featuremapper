import numpy as np

def wrap(lower, upper, x):
    """
    Circularly alias the numeric value x into the range [lower,upper).

    Valid for cyclic quantities like orientations or hues.
    """
    #I have no idea how I came up with this algorithm; it should be simplified.
    #
    # Note that Python's % operator works on floats and arrays;
    # usually one can simply use that instead.  E.g. to wrap array or
    # scalar x into 0,2*pi, just use "x % (2*pi)".
    range_=upper-lower
    return lower + np.fmod(x-lower + 2*range_*(1-np.floor(x/(2*range_))), range_)


"""
Return the cross-product of a variable number of lists (e.g. of a list of lists).

Use to obtain permutations, e.g.
l1=[a,b]
l2=[c,d]
cross_product([l1,l2]) =
[[a,c], [a,d], [b,c], [b,d]]


From:
http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/159975
"""
# Need to re-write so someone other than Python knows what might happen when this runs
cross_product=lambda ss,row=[],level=0: len(ss)>1 \
   and reduce(lambda x,y:x+y,[cross_product(ss[1:],row+[i],level+1) for i in ss[0]]) \
   or [row+[i] for i in ss[0]]


# JABALERT: Should frange be replaced with numpy.arange or numpy.linspace?
def frange(start, end=None, inc=1.0, inclusive=False):
    """
    A range function that accepts float increments.

    Otherwise, works just as the inbuilt range() function.  If
    inclusive is False, as in the default, the range is exclusive (not
    including the end value), as in the inbuilt range(). If inclusive
    is true, the range may include the end value.

    'All theoretic restrictions apply, but in practice this is
    more useful than in theory.'

    From: http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/66472
    """
    if end == None:
        end = start + 0.0
        start = 0.0

    # Increments of zero would lead to an infinite loop, which can happen if
    # this is mistakenly called with a integer-based rational expression like 1/2.
    assert ((inc>0 and start<=end) or (inc<0 and start>=end))

    L = []
    while 1:
        next = start + len(L) * inc
        if inclusive:
          if inc > 0 and next > end: break
          elif inc < 0 and next < end: break
        else:
          if inc > 0 and next >= end: break
          elif inc < 0 and next <= end: break
        L.append(next)

    return L