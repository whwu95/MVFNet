"""tools"""
import functools

# See:
# https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects


def rsetattr(obj, attr, val):
    """set attribute on nested subobjects/chained properties."""

    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    """get attribute on nested subobjects/chained properties."""

    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split('.'))


def rhasattr(obj, attr, *args):
    """judge if an attribute is on nested subobjects/chained properties."""

    def _hasattr(obj, attr):
        if hasattr(obj, attr):
            return getattr(obj, attr)
        else:
            return None

    return functools.reduce(_hasattr, [obj] + attr.split('.')) is not None


def get_layer_name(layer_params):
    """get layer name when loading checkpoint."""
    layer_name = list()

    for layer_param in layer_params:
        pos = layer_param.rfind('.')
        layer = layer_param[:pos]
        if layer not in layer_name:
            layer_name.append(layer)

    return layer_name
