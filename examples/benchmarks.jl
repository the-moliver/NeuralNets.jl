using DataFrames 
using MLBase
using StatsBase
using NeuralNets

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

function untransform(t::Standardize, x::Array)
    (x .+ t.mean .* t.scale) ./ t.scale
end

X,T = dataprep(bicycle)

X = X[:,1:100:end] # reduce the size of the data set by a factor of 100
T = T[:,1:100:end]

# feature standardization
mX = mean(X,2)
X = X .- mX
sX = std(X,2)
X = X ./ sX

T = T - mean(T)
T = T ./ std(T)

ind = size(X,1)
outd = size(T,1)

layer_sizes = [ind, 6, outd]
act   = [relu,  ident]

# not working 100%, it's a difficult set to get to converge in a sensible time period

mlp = MLP(rand, layer_sizes, act)
#params = TrainingParams(100, 1e-5, 2e-6, .7, :levenberg_marquardt)

O = prop(mlp,X)
@show mean((O .- T).^2)

println("Training...")
#mlp = train(mlp, params, X, T, verbose=false)
mlp1 = train(mlp, X, [], T, [], train_method=:levenberg_marquardt)

mlp = MLP(rand, layer_sizes, act)
mlp2 = gdmtrain(mlp, X, T, learning_rate=.0005, maxiter=20000)

mlp = MLP(rand, layer_sizes, act)
mlp3 = adatrain(mlp, X, T, maxiter=10000)

mlp = MLP(rand, layer_sizes, act)
mlp4 = rmproptrain(mlp, X, T, learning_rate=.01)

O = prop(mlp3,X)
@show mean((O .- T).^2)
