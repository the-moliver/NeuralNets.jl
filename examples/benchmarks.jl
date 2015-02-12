using DataFrames 
using MLBase
using StatsBase
using NeuralNets

cd("C:\\Users\\micha_000\\Documents\\GitHub\\NeuralNets.jl")
bicycle = readtable("./datasets/BicycleDemand/bicycle_demand.csv") # load test data

function dataprep(data)
    year    = int(map(x->x[1:4],data[:datetime]))
    month   = int(map(x->x[6:7],data[:datetime]))
    day     = int(map(x->x[9:10],data[:datetime]))
    hour    = int(map(x->x[12:13],data[:datetime]))
    X = hcat(year,month,hour,array(data[[:season,:temp,:holiday,:workingday,:weather,:atemp,:humidity,:windspeed]]))'
    T = array(data[[:count]])'

    X = convert(Array{Float64,2},X)
    T = convert(Array{Float64,2},T)
    return X,T
end

X,T = dataprep(bicycle)

X = X[:,1:end-1000] # reduce the size of the data set by a factor of 100
T = T[:,1:end-1000]

# feature standardization
mX = mean(X,2)
X = X .- mX
sX = std(X,2)
X = X ./ sX

mT = mean(T)
T = T - mT
sT = std(T)
T = T ./ sT

ind = size(X,1)
outd = size(T,1)

layer_sizes = [ind, 10,10,10, outd]
act   = [donrelu,nrelu,nrelu, ident]

X = convert(Array{Float32},X);
T = convert(Array{Float32},T);

layer_lags = [1, 1, 1, 1]



tdmlp = TDMLP(layer_sizes, layer_lags, act);
tdmlp = rmsproptrain(tdmlp, X, T, learning_rate=.00001, learning_rate_factor=.999, maxiter=1, batch_size = 100);
tdmlp = rmsproptrain(tdmlp, X, T, learning_rate=.00001, learning_rate_factor=.999, maxiter=500, batch_size = 100);


mlp = MLP(layer_sizes, act);
mlp = rmsproptrain(mlp, X, T, learning_rate=.00001, learning_rate_factor=.999, maxiter=1, batch_size = 100);
mlp = rmsproptrain(mlp, X, T, learning_rate=.00001, learning_rate_factor=.999, maxiter=500, batch_size = 100);

O = prop(mlp,X)
@show mean((O .- T).^2)

for ii=1:3
tdmlp.net[1].a = relu
tdmlp.net[1].ad = relud
end
tdmlp = rmsproptrain(tdmlp, X, T, learning_rate=.00001, learning_rate_factor=.999, maxiter=100, batch_size = 100)


mlp = MLP(randn, layer_sizes, act)
#params = TrainingParams(100, 1e-5, 2e-6, .7, :levenberg_marquardt)

O = prop(mlp,X)
@show mean((O .- T).^2)

println("Training...")
#mlp = train(mlp, params, X, T, verbose=false)
#mlp1 = train(mlp, X, [], T, [], train_method=:levenberg_marquardt)

O = prop(mlp1,X)
@show mean((O .- T).^2)

#mlp = MLP(randn, layer_sizes, act)
#mlp2 = gdmtrain(mlp, X, T, learning_rate=.00005, maxiter=10000)

O = prop(mlp2,X)
@show mean((O .- T).^2)

#mlp = MLP(randn, layer_sizes, act)
#mlp3 = adatrain(mlp, X, T, maxiter=10000)

O = prop(mlp3,X)
@show mean((O .- T).^2)

mlp = MLP(randn, layer_sizes, act);
mlp4 = rmsproptrain(mlp, X, T, learning_rate=.01, learning_rate_factor=.999, maxiter=1000, batch_size = 100)


for ii=1:3
mlp4.net[1].a = relu
mlp4.net[1].ad = relud
end

mlp4 = rmsproptrain(mlp4, X, T, learning_rate=.01, learning_rate_factor=.9, maxiter=10, loss=quartic_loss)

O = prop(mlp4,X)
@show mean((O .- T).^2)

X,T = dataprep(bicycle)

X2 = X[:,end-999:end] # reduce the size of the data set by a factor of 100
T2 = T[:,end-999:end]

X2 = X2 .- mX
X2 = X2 ./ sX

T2 = T2 - mT
T2 = T2 ./ sT

O2 = prop(tdmlp,X2)

@show mean((O2 .- T2).^2)

layer_lags = [1, 5, 1, 1]
tdmlp = TDMLP(randn, layer_sizes, layer_lags, act);
tdmlp = rmsproptrain(tdmlp, X, T, learning_rate=.01, learning_rate_factor=.999, maxiter=1000, batch_size = 100)
