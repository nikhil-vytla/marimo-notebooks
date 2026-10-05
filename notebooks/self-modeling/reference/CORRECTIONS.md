# Important corrections and context

1. The hyperparameters in Table 1 list an incorrect batch size.  The correct batch size is 512.
2. The MNIST data is preprocessed via linear rescaling of the training data to [-1, 1], centering
   data at zero.
3. Note: target activations should be detached from the computation graph before computing the self-modeling loss
