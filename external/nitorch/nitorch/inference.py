import torch
from torch import nn
from nitorch.utils import to_numpy

def predict(
        all_outputs,
        all_labels,
        task_type,
        criterion,
        **kwargs
):
    """Predict according to loss and prediction type.

    Parameters
    ----------
    all_outputs
        All outputs of a forward process of a model.
    all_labels
        All labels of the corresponding inputs to the outputs.
    task_type
        "classif_binary", "classif", "regression", or "other".
    criterion
        Criterion, e.g. "loss"-function. Could for example be "nn.BCEWithLogitsLoss".
    kwargs
        Variable arguments.

    Returns
    -------
    all_preds
        All predictions.
    all_labels
        All labels.

    Raises
    ------
    NotImplementedError
        If `task_type` invalid.

    """
    all_outputs = all_outputs.squeeze()
    if (task_type=="classif_binary") and (all_outputs.ndim==1):
        all_preds, all_labels = classif_binary_inference(
                                    all_outputs, all_labels, 
                                    criterion=criterion, 
                                    **kwargs)
        
    elif (task_type=="classif") or (task_type=="classif_binary" and all_outputs.ndim==2):
        # users might also use binary classification with 2 class outputs instead
        # which should be handled same as multi class classification infernece
        all_preds, all_labels = classif_inference(
                                    all_outputs, all_labels, 
                                    criterion=criterion, 
                                    **kwargs)
        
    elif task_type in ["regression", "other"]:
        all_preds = all_outputs
    else:
        raise NotImplementedError(f"task_type={task_type} not supported currently in nitorch. Only ['classif_binary', 'classif', 'regression', or 'other'] supported..")
    
    
    return to_numpy(all_preds, all_labels)


def classif_inference(
        all_outputs,
        all_labels,
        criterion,
        **kwargs
):    
    print('[D]',all_outputs.shape, all_labels.shape, )

    # get the class with the highest logit as the predicted class
    all_preds = torch.argmax(all_outputs, 1)   
    # if labels are one-hot vectors, convert them to class variables for metric calculations
    if all_labels.ndim>1:
        all_labels = torch.argmax(all_labels, 1)
    print('[D]2',all_outputs.shape, all_labels.shape, )
    return all_preds, all_labels


def classif_binary_inference(
        all_outputs,
        all_labels,
        criterion,
        **kwargs):
    # convert the outputs from logits to probability
    if isinstance(criterion, nn.BCEWithLogitsLoss):
        all_outputs = torch.sigmoid(all_outputs)

    class_threshold = 0.5
    if "class_threshold" in kwargs and kwargs["class_threshold"] is not None:
        class_threshold = kwargs["class_threshold"]

    all_preds = (all_outputs >= class_threshold)
    return all_preds, all_labels

