import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def to_numpy(*torch_arrs):
    """Converts the given torch Tensors to numpy.ndarray for metric calculations.

    Parameters
    ----------
    *torch_arrs : one or multiple torch.tensors
        Either a CPU or GPU tensor.

    Returns
    -------
    *numpy_arrs : numpy.ndarray
        Numpy.ndarray of the input tensor data.
    """
    out = []
    for torch_arr in torch_arrs:
        if isinstance(torch_arr, torch.Tensor):
            if torch_arr.is_cuda: 
                out.append(torch_arr.cpu().detach().numpy())
            else:
                out.append(torch_arr.detach().numpy())
        else:
            out.append(torch_arr)
    return out


def dataset_length(data_loader):
    """Return the full length of the dataset from the DataLoader alone.

    Calling len(data_loader) only shows the number of mini-batches.
    Requires data to be located at.

    Parameters
    ----------
    data_loader : torch.utils.data.DataLoader
        The data_loader for the data.

    Returns
    -------
    int
        The total length of the `data_loader`.

    Raises
    ------
    KeyError
        If each entry in the `data_loader` is a dictionary, key for the label is expected to be 'label'.

    """
    sample = next(iter(data_loader))
    batch_size = None

    if isinstance(sample, dict):
        try:
            if isinstance(sample["label"], torch.Tensor):
                batch_size = sample["label"].shape[0]
            else:
                # in case of sequence of inputs use first input
                batch_size = sample["label"][0].shape[0]
        except:
            KeyError("Expects key to be 'label'.")
    else:
        if isinstance(sample[1], torch.Tensor):
            batch_size = sample[1].shape[0]
        else:
            # in case of sequence of inputs use first input
            batch_size = sample[1][0].shape[0]
    return len(data_loader) * batch_size


def count_parameters(model):
    """Returns the number of adjustable parameters of the input.

    Parameters
    ----------
    model
        The model.

    Returns
    -------
    int
        The number of adjustable parameters of `model`.

    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class WatchGrads:
    
    def __init__(self, model_named_params):    
        """Plots the gradients flowing through different layers in the net during training.
        Can be used for checking for possible gradient vanishing / exploding problems.

        Parameters
        ----------
        model_named_params :
            A list of tuples returned by model.named_parameters() from pytorch models. 
            First entry is the name, second the value."""
        self.ave_grads = []
        self.max_grads = []
        self.layers = []
        for n, p in model_named_params:
            if p.requires_grad and "bias" not in n:
                self.layers.append(n.replace(".weight", ""))
        self.N = len(self.layers)
        
        
    def store(self, model_named_params):
        ave_grads=[]
        max_grads=[]
        for n, p in model_named_params:
            if p.requires_grad and "bias" not in n:
                ave_grads.append(p.grad.abs().mean().cpu())
                max_grads.append(p.grad.abs().max().cpu())
        self.ave_grads.append(ave_grads)
        self.max_grads.append(max_grads)
    
    
    def plot(self, save_fig_path=''):
        f, ax = plt.subplots(figsize=(
            round(self.N**(1/2))+1,4+self.N//15), 
                             constrained_layout=True) 
        
        for max_grads, ave_grads in zip(self.max_grads, self.ave_grads):
            plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
            plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
        
        plt.hlines(0, 0, self.N+1, lw=2, color="k")
        plt.xticks(range(0, self.N, 1), self.layers, rotation="vertical")
        plt.xlim(left=0, right=self.N)
        plt.ylim(bottom=-0.001, top=2*np.array(self.ave_grads).mean()) 
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Average gradients flowing through network layers")
        plt.grid(True)
        plt.legend([Line2D([0], [0], color="c", lw=4),
                    Line2D([0], [0], color="b", lw=4),
                    Line2D([0], [0], color="k", lw=4)], 
                   ['max-gradient', 'mean-gradient', 'zero-gradient'],
                  loc='upper left')
       
        if save_fig_path:
            plt.savefig(save_fig_path+f"grad_flow.jpg")
        plt.show()
    
    

def is_bad_grad(grad_output):
    """Checks if gradient is too big

    Parameters
    ----------
    grad_output
        The gradient you got back during back-propagation.

    Returns
    -------
    bool
        True if gradient is bad, False otherwise.

    """
    grad_output = grad_output.data
    return grad_output.ne(grad_output).any() or grad_output.gt(1e6).any()
