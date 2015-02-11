# NeuralNets.jl
An open-ended implentation of deep neural networks (AKA multi-layer perceptrons) and deep time-delay neural networks in Julia.

Some features:
* Flexible network topology with any combination of activation function/layer number.
* Support for a number of common node activation functions in addition to support for arbitrary activation functions with the use of automatic differentiation.
* A broad range of training algorithms to chose from.
* DropOut and Poisson-like noise regularization.
* Time Delay Neural Networks, with arbitrary time delays at each layer.



## Usage
Multi-layer perceptrons are instantiated by using the `MLP(layer_sizes,act)` constructor  to describe the network topology and initialisation procedure as follows:
* `layer_sizes::Vector{Int}` is a vector whose first element is the number of input nodes, and the last element is the number of output nodes, intermediary elements are the numbers of hidden nodes per layer
* `act::Vector{Function}` is the vector of activation functions corresponding to each layer

For example, `MLP([4,8,8,2], [relu,logis,ident])` returns a 3-layer network with 4 input nodes, 2 output nodes, and two hidden layers comprised of 8 nodes each. The first hidden layer uses a `relu` activation function, the second uses `logis`. The output nodes lack any activation function and so we specify them with the `ident` 'function'—but this could just as easily be another `logis` to ensure good convergence behaviour on a 1-of-k target vector like you might use with a classification problem.

Time-delay multi-layer perceptrons are instantiated by using the `TDMLP(layer_sizes, delays, act)` constructor  to describe the network topology and initialisation procedure as follows:
* `layer_sizes::Vector{Int}` is a vector whose first element is the number of input nodes, and the last element is the number of output nodes, intermediary elements are the numbers of hidden nodes per layer
* `delays::Vector{Int}` is a vector of the number of time steps used at each layer. When set to all ones, it functions identially to mlp. Any values greater than 1 indicate that past values of the previous layer will be used as input to the next layer.
* `act::Vector{Function}` is the vector of activation functions corresponding to each layer

For example, `TDMLP([4,10,10,1], [1 12 1], [relu,relu,relu])` returns a 3-layer network with 4 input nodes, 1 output node, and two hidden layers comprised of 10 nodes each. The first and second hidden layer uses a `relu` activation function, as does the output node. The weights in the first hidden layer only act on the input at the current time step, while the second hidden layer has weights for the output of the first hidden layer at the current time step and the 11 previous time steps.

Once your neural network is initialised (and trained), predictions are made with the `prop(mlp,x)` command, where `x` is a column vector of the node inputs. Of course `prop()` is also defined on arrays, so inputting a k by n array of data points returns a j by n array of predictions, where k is the number of input nodes, and j is the number of output nodes.

### Activation Functions
There is 'native' support for the following activation functions. If you define an arbitrary activation function its derivative is calculated automatically using the [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) package. The natively supported activation derivatives are a bit over twice as fast to evaluate compared with derivatives calculated using [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl).
* `ident` the identify function, f(x) = x.
* `logis` the logistic sigmoid, f(x) = 1 ./(1 .+ exp(-x)).
* `softmaxact` softmax activation function for multiclass classification, f(x) = exp(x) ./sum(exp(x)).
* `logissafe` the logistic sigmoid with a 'safe' derivative which doesn't collapse when evaluating large values of x.
* `srelu` soft rectified linear units , f(x) = log(1 .+ exp(x)).
* `relu` rectified linear units , f(x) = max(0,x).
* `nrelu` Poisson-like noisy rectified linear units , f(x) = max(0,x) + sqrt(max(0,x))*randn(size(x)).
* `donrelu` Poisson-like noisy rectified linear units, f(x) = max(0,x) + sqrt(max(0,x))*randn(size(x)), with 50% drop out.
* `tanhact` hyperbolic tangent activation f(x) = tanh(x).

### Training Methods
Once the MLP type is constructed we train it using one of several provided training functions.

* `gdmtrain(nn, x, t)`: This is a natively-implemented gradient descent training algorithm with Nesterov momentum. Optional parameters include:
    * `batch_size` (default: n): Randomly selected subset of `x` to use when training extremely large data sets. Use this feature for 'stochastic' gradient descent.
    * `maxiter` (default: 1000): Number of iterations before giving up.
    * `tol` (default: 1e-5): Convergence threshold.
    * `learning_rate` (default: .3): Learning rate of gradient descent. While larger values may converge faster, using values that are too large may result in lack of convergence (you can typically see this happening with weights going to infinity and getting lots of NaNs). It's suggested to start from a small value and increase if it improves learning.
    * `momentum_rate` (default: .6): Amount of momentum to apply. Try 0 for no momentum.
    * `eval` (default: 10): The network is evaluated for convergence every `eval` iterations. A smaller number gives slightly better convergence but each iteration takes a slightly longer time.
    * `verbose` (default: true): Whether or not to print out information on the training state of the network.
    * 
* `rmsproptrain(nn, x, t)`: This is a natively-implemented RMSProp training algorithm with Nesterov momentum. Optional parameters include:
    * `batch_size` (default: n): Randomly selected subset of `x` to use when training extremely large data sets. Use this feature for 'stochastic' gradient descent.
    * `maxiter` (default: 1000): Number of iterations before giving up.
    * `tol` (default: 1e-5): Convergence threshold.
    * `learning_rate` (default: .3): Learning rate of gradient descent. While larger values may converge faster, using values that are too large may result in lack of convergence (you can typically see this happening with weights going to infinity and getting lots of NaNs). It's suggested to start from a small value and increase if it improves learning.
    * `momentum_rate` (default: .6): Amount of momentum to apply. Try 0 for no momentum.
    * `eval` (default: 10): The network is evaluated for convergence every `eval` iterations. A smaller number gives slightly better convergence but each iteration takes a slightly longer time.
    * `verbose` (default: true): Whether or not to print out information on the training state of the network.

