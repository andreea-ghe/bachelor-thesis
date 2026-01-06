from easydict import EasyDict as edict


def static_vars(**kwargs):
    """
    Decorator to add static variables to a function.
    """
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate

@static_vars(indent_cnt=0)
def print_edict(input_dict: edict):
    """
    Pretty print an EasyDict object.
    """
    for key, value in input_dict.items():
        if type(value) is edict or type(value) is dict:
            print("{}{}:".format(" " * 2 * print_edict.indent_cnt, key))
            # go recursive
            print_edict.indent_cnt += 1
            print_edict(value)
            print_edict.indent_cnt -= 1
        else:
            print("{}{}: {}".format(" " * 2 * print_edict.indent_cnt, key, value))