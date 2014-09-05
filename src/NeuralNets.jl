module NeuralNets

using Optim
using ArrayViews

import Optim:levenberg_marquardt

# functions
export train, gdmtrain, adatrain, prop, rmsproptrain
export logis, logisd, logissafe, logissafed, relu, relud, srelu, srelud, nrelu, nrelud, donrelu, donrelud, ident, identd, tanhact, tanhactd, expact, expactd
export squared_loss, squared_lossd, linear_loss, linear_lossd, hinge_loss, hinge_lossd, log_loss, log_lossd, quartic_loss, quartic_lossd, poisson_loss, poisson_lossd
 
# types
export MLP, NNLayer

# multi-layer perceptrons
include("activations.jl")
include("losses.jl")
include("mlp.jl")
include("tdmlp.jl")

# training
include("backprop.jl")
include("lmtrain.jl")
include("gradientdescent.jl")
include("train.jl")

end
