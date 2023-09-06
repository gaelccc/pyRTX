import os, sys, inspect
from multiprocessing import Pool
from pathos.multiprocessing import ProcessingPool
from functools import partial, wraps

"""
A set of utilities focused on increasing the computational efficiency of numerical calculations.
"""


def get_ncores(lst, **kwargs):
    """
    Returns the number of cores for parallel computations.
    If not specified, the number of cores is set to the number available on your machine.
    """
      
    if ('n_cores' in kwargs.keys()) and (isinstance(kwargs['n_cores'], int)) and (os.cpu_count() >= kwargs['n_cores'] >= 1): 
        n_cores = kwargs['n_cores']
    else: n_cores = os.cpu_count() 
    
    n_cores = min(len(lst), n_cores) if len(lst) else 1
    
    return n_cores



def is_method(func, cls_inst):
    """
    Checks if the wrapped object is a method of a class.
    """
    return (not inspect.isclass(cls_inst)) and (hasattr(cls_inst, func.__name__)) and ('self' in inspect.getargspec(func).args)



def get_unwrapped(*args, **kwargs):
    """
    Returns the unwrapped version of a class method.
    """
    cls_inst = args[0]
    cls_attr = kwargs['method']
    return getattr(cls_inst, cls_attr).__wrapped__(*args)



def parallel(func):
    """
    Decorator for parallel computations.
    The decorated object can be a class method or a regular function.
    It applies the function to every item of an iterable performing the computations in parallel. 
    The output will be an iterator which contains the return value of every function call.
    
    The decorated object can accept multiple arguments, but the last positional argument 
    must be an item of the input iterable (e.g. an item of a list / numpy array / range object ...).
    
    If the keyword argument 'n_cores' is specified, the decorated object will run on n_cores processes. 
    Otherwise, the number of cores is set to the number available on your machine.
    
    Basic usage:
    
    - Define the input iterable:
    
    iterable = [2,5,6,7,3,4,1] 
    
    - Define the function that will be applied to every item of the input iterable:
    
    @parallel
    def target_func( some_inputs, item ): 
        ... perform tasks 
        return result
    
    - Apply the function to every item of the iterable using 5 cores in parallel:
    
    results = target_func( some_inputs, iterable, n_cores = 5 )   

    NOTE: In its definition, the decorated object accepts the item of the iterable as last positional argument.
    However, in the function call, you must pass directly the iterable (the list of integers in this example) 
    in place of the item. Every other positional argument before the iterable must be the same.
    In the function call it is possible to also pass the number of cores, but only as a keyword argument.
    
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        """
        Wrapper for the target function.
        It creates a multiprocessing pool to run the target function in parallel.
        
        """

        # Check if the last positional input is an iterable.
        if iter(args[-1]): lst = args[-1]
        
        # Get the number of cores to use.
        n_cores = get_ncores(lst, **kwargs)
        
        # Case 0: sequential computations (n_cores = 1).
        if get_ncores(lst, **kwargs) == 1: results = [func(*args[:-1], it) for it in lst]
        
        # Case 1: parallel computations for a class method.
        elif is_method(func, args[0]):

            # Run the multiprocessing pool.
            task = partial(get_unwrapped, *args[:-1], method = func.__name__)
            with Pool(n_cores) as p: results = p.map(task, lst)

        # Case 2: parallel computations for a regular function.
        else: 

            # Run the multiprocessing pool from 'pathos' library (it does not use pickle).
            task = partial(func, *args[:-1])
            with ProcessingPool(n_cores) as p: results = p.map(task, lst)
            
        return results

    return wrapper

