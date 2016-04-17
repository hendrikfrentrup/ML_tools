​import numpy as np

def func1(new_value):
    array_to_mod = np.zeros(100)
    for i in range(0,100):
        array_to_mod[i] = new_value
    final_array = array_to_mod[0:len(range(0,100))]
    return final_array

def func2(new_value):
    final_array = np.array([])
    for i in range(0,100):
        final_array = np.append(final_array, new_value)
    return(final_array)

    array_to_mod = np.zeros(100)
def stack(old, new):
    out = np.ones(len(old)+len(new))
    out[0:len(old)] = old
    out[len(old):] = new
    return out
