using NeuralNets

# xor training data
x = [0.0 1.0 0.0 1.0
    0.0 0.0 1.0 1.0]

t = [0.0 1.0 1.0 0.0]

# network topology
layer_sizes = [2, 3, 3, 1]
act   = [nrelu,  nrelu,  logissafe]



# initialize net
mlp = MLP(layer_sizes, act)

# # train without a validation set
# mlp1 = train(mlp, x, [], t, [], train_method=:levenberg_marquardt)
# @show prop(mlp1, x)

mlp = MLP(layer_sizes, act, datatype=Float64)
mlp2 = gdmtrain(mlp, x, t)
@show prop(mlp2, x)

# mlp = MLP(layer_sizes, act, datatype=Float64)
# mlp3 = adatrain(mlp, x, t, loss=log_loss)
# @show prop(mlp3, x)

mlp = MLP(layer_sizes, act, datatype=Float64)
mlp4 = rmsproptrain(mlp, x, t, learning_rate=.1,loss=log_loss, maxiter=1000)
@show prop(mlp4, x)

