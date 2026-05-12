# Initialize weights
from torch.nn import init

def weights_init(m, func=init.kaiming_uniform_, debug_print=False):
    """Performs weight initialization for a layer.
    Parameters
    ----------
    m   :  The layer which weights should be initialized.
    func:  The sampling function from torch.nn.init class
           to use to initialize weights.
           Some other examples: init.xavier_normal_, 
           xavier_uniform_, kaiming_normal_
    Returns
    -------
    m:  Weight initialized layer.
    """ 
    if hasattr(m, 'weight') and 'BatchNorm' not in m.__class__.__name__:
        func(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0.001)
    # for layers like nn.BatchNorm3d, just reset the parameters
    elif hasattr(m, 'reset_parameters') and callable(m.reset_parameters): 
        m.reset_parameters()
    # else skip the layer and print out its name
    elif debug_print and len(list(m.children()))==0:
        print(f"weights_init:: skipping layer {m}")