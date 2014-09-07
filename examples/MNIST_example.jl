using NeuralNets
using MNIST

trainstim, trainresp = traindata()

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

layer_sizes = [ind, 100,100,100, outd]
act   = [donrelu,nrelu,nrelu, softmaxactsafe];


function smallweights(x)
	randn(x)/1000
end

mlp = MLP(smallweights, layer_sizes, act);

mlp = rmsproptrain(mlp, trainstim, trainval, learning_rate=.0001, learning_rate_factor=.999, maxiter=100, batch_size = 100, loss=xent_loss)

O = prop(mlp, trainstim);