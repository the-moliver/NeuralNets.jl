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


layer_sizes = [ind, 500, 50, 200, outd]
act   = [nrelu,nrelu,nrelu, softmaxact];

trainstim = convert(Array{Float32}, trainstim)
trainval = convert(Array{Float32}, trainval)


mlp = MLP(layer_sizes, act);
mlp = rmsproptrain(mlp, trainstim, trainval, learning_rate=.001, learning_rate_factor=.96, maxiter=200, batch_size = 100, loss=xent_loss, maxnorm=1.)

# for ii=1:3
# mlp.net[ii].a = relu
# mlp.net[ii].ad = relud
# end

# mlp = rmsproptrain(mlp, trainstim, trainval, learning_rate=.001, learning_rate_factor=.9, maxiter=100, batch_size = 100, loss=xent_loss, maxnorm=1.)


O = prop(mlp, trainstim);

teststim, testresp = testdata()

testval = zeros(10, length(testresp))
for ii=1:length(testresp)
	testval[testresp[ii]+1,ii]=1;
end

teststim = teststim .- mX
teststim = teststim ./ sX
pred = prop(mlp, teststim);

pred= sum(diagm([0:9])*round(pred),1)

1-sum(abs(testresp - pred') .> 0)./length(pred)