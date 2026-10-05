def weight_sd(model):
    """Standard deviation of all elements of the 10 x hidden classifier weight."""
    return model.classifier_weight().detach().std().item()
