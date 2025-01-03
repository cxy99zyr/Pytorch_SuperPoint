def get_dataset(name):
    """Get a dataset class by name.
    
    Args:
        name: The name of the dataset module.
        
    Returns:
        The dataset class.
    """
    mod = __import__('superpoint.datasets.{}'.format(name), fromlist=[''])
    return getattr(mod, _module_to_class(name))


def _module_to_class(name):
    """Convert a module name to a class name.
    
    Args:
        name: The module name (e.g. 'synthetic_shapes').
        
    Returns:
        The class name (e.g. 'SyntheticShapes').
    """
    return ''.join(n.capitalize() for n in name.split('_')) 