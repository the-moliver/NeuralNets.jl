module NeuralNets

using Optim
using ArrayViews

import Optim:levenberg_marquardt

# functions
export train, gdmtrain, adatrain, prop, rmsproptrain
export logis, logisd, logissafe, logissafed, relu, relud, srelu, srelud, nrelu, nrelud, ident, identd, tanhd

# types
export MLP, NNLayer

# multi-layer perceptrons
include("activations.jl")
include("losses.jl")
include("mlp.jl")

# training
include("backprop.jl")
include("lmtrain.jl")
include("gradientdescent.jl")
include("train.jl")

end
