using NeuralNets
using MAT


# load data
data = matread("examples/v1cattest2.mat")

stim = float32(data["stim"]);
resp = float32(data["resp"]);
trainidx = vec(int(data["trainidx"]));
validx = vec(int(data["validx"]));

# split into training and validation data
trainstim = stim[trainidx,:];
trainresp = resp[trainidx,:];
valstim = stim[validx,:];
valresp = resp[validx,:];

# put into correction orientation
trainstim=trainstim';
trainresp=trainresp';
valstim=valstim';
valresp=valresp';

# find input and output dimensionality
ind = size(trainstim,1)
outd = size(trainresp,1)

# specify layer sizes
layer_sizes = [ind, 20,20, outd];

# create parameterized soft-rectification output nonlinearity, k=10
srelu10(x) = sreluk(x,10)
srelu10d(x,idx) = srelukd(x,idx,10)
merge!(derivs, [srelu10   => srelu10d])


# create parameterized dropout noisy rectified linear activation function with p=0.8 (probability of retaining unit)
do8nrelu(x) = dopnrelu(x,.8)
do8nrelud(x,idx) = dopnrelud(x,idx,.8)
merge!(derivs, [do8nrelu   => do8nrelud])

# create vector of activations at each layer
act   = [do8nrelu,nrelu,srelu10];

# create vector of time delays to consider in each layer
delays = [1, 12, 1];

# initialize time-delay multi-layer perceptron, without output gain = .1
tdmlp = TDMLP(layer_sizes, delays, act, gain=.1);

# train network for 200 epochs
tdmlp = rmsproptrain(tdmlp, trainstim, trainresp, learning_rate=.0001, learning_rate_factor=.99, maxiter=200, batch_size = 100, loss=poisson_loss, maxnorm=3., minadapt=1., xval=valstim, tval=valresp);

# calculate prediction performance
pred = prop(tdmlp, valstim);
perf = cor(pred[12:end], valresp[12:end]);

println("Prediction Performance r = $perf")