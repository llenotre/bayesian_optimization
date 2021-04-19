# Bayesian Optimization

The Bayesian Optimization algorithm is a method used in a lot of domains, including machine
leaning for finding hyperparameters, which allows to compute an approximation of a black-box
function which is expensive to evaluate.

The implementation is based off the following paper: [](https://export.arxiv.org/pdf/1807.02811)

The Gaussian kernel is used as the interpolation kernel, and the Expected Improvement function as
the acquisition function.
