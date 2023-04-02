# Notes

## Ideas

- Use a profiler
  - PyTorch profiler
  - Nsight systems
- Track loss
  - Using some tool?
  - Weights and biases?
- Optimise hyperparameters
- Try more difficult problems
- Batch norm
  - [Disable bias for convolutions directly followed by batch norm](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#disable-bias-for-convolutions-directly-followed-by-a-batch-norm)
- Track optimizer information like update norm, gradient norm, norm of momentum term, angle between them etc.
- Randomise inputs for regularisation

## Convolutional autoencoders

- Don't use MaxUnpool2d, use Upsample.
  - MaxUnpool2D requires indices, which didn't work too well as they gave too much information away, rather than storing information in the latent space
