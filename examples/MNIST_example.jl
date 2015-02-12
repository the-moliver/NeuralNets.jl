using NeuralNets
using MNIST

# load training MNIST data
trainstim, trainresp = traindata()

# convert to 1-hot format
trainval = zeros(10, length(trainresp))
for ii=1:length(trainresp)
	trainval[trainresp[ii]+1,ii]=1;
end

# feature standardization
mX = mean(trainstim,2)
trainstim = trainstim .- mX
sX = std(trainstim,2)
sX[sX.==0.]=1.
trainstim = trainstim ./ sX

ind = size(trainstim,1)
outd = size(trainval,1)


# create parameterized dropout noisy rectified linear activation function with p=0.8 (probability of retaining unit)
do8nrelu(x) = dopnrelu(x,.8)
do8nrelud(x,idx) = dopnrelud(x,idx,.8)
merge!(derivs, [do8nrelu   => do8nrelud])

# specify layer sizes
layer_sizes = [ind, 50, 50, 20, outd]

# create vector of activations at each layer
act   = [do8nrelu,nrelu,nrelu, softmaxact];

trainstim = convert(Array{Float32}, trainstim)
trainval = convert(Array{Float32}, trainval)

# initialize multi-layer perceptron
mlp = MLP(layer_sizes, act);

# fit using rmsprop
mlp = rmsproptrain(mlp, trainstim, trainval, learning_rate=.001, learning_rate_factor=.96, maxiter=20, batch_size = 100, loss=xent_loss, maxnorm=1.)


# load test data
teststim, testresp = testdata()

# convert to 1-hot format
testval = zeros(10, length(testresp))
for ii=1:length(testresp)
	testval[testresp[ii]+1,ii]=1;
end

# apply standardization to test data
teststim = teststim .- mX
teststim = teststim ./ sX

# calculate prediction
pred = prop(mlp, teststim);
pred= sum(diagm([0:9])*round(pred),1)


perf = 100*(1-sum(abs(testresp - pred') .> 0)./length(pred));
println("Prediction Performance: $perf% correct")