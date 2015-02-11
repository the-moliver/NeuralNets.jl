module NeuralNets

using Optim
using ArrayViews

import Optim:levenberg_marquardt

# functions
export train, gdmtrain, adatrain, prop, rmsproptrain, backprop, errprop, mprop, finite_diff, finite_diff_m, mini_batch!, mini_batch_init, deltas_init, view, reshape_view, flatten_net, unflatten_net!, nansum
export logis, logisd, logissafe, logissafed, relu, relud, srelu, srelud, nrelu, nrelud, ident, identd, tanhact, tanhactd, expact, expactd, softmaxact, sreluk, srelukd, doprelu, doprelud, dopnrelu, dopnrelud
export squared_loss, squared_lossd, linear_loss, linear_lossd, hinge_loss, hinge_lossd, log_loss, log_lossd, quartic_loss, quartic_lossd, poisson_loss, poisson_lossd, xent_loss
# dictionaries
export derivs, cannonical
# types
export MLP, NNLayer, TDMLP, TDNNLayer, MLNN, Deltas, deltaLayer

# multi-layer perceptrons
include("losses.jl")
include("activations.jl")
include("mlp.jl")
include("tdmlp.jl")

# training
include("backprop.jl")
include("gradientdescent.jl")

end
