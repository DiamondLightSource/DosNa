#!/usr/bin/env python
"""
Helper functions to manage data and metadata
"""


def slices2shape(slices):
    result = []
    for slice_ in slices:
        result.append(slice_.stop - slice_.start)
    return tuple(result)
