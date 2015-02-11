using NeuralNets
using MAT
using ProfileView

data = matread("/auto/k1/moliver/v1cattest2.mat")

stim = float32(data["stim"]);
resp = float32(data["resp"]);
trainidx = vec(int(data["trainidx"]));
validx = vec(int(data["validx"]));

trainstim = stim[trainidx,:];
trainresp = resp[trainidx,:];
valstim = stim[validx,:];
valresp = resp[validx,:];

trainstim=trainstim';
trainresp=trainresp';
valstim=valstim';
valresp=valresp';
ind = size(trainstim,1)
outd = size(trainresp,1)

layer_sizes = [ind, 20,20, outd]

srelu10(x) = sreluk(x,10)
srelu10d(x,idx) = srelukd(x,idx,10)
merge!(derivs, [srelu10   => srelu10d])

do8nrelu(x) = dopnrelu(x,.8)
do8nrelud(x,idx) = dopnrelud(x,idx,.8)
merge!(derivs, [do8nrelu   => do8nrelud])


act   = [do8nrelu,nrelu,srelu10];

# trainresp[isnan(trainresp)]=0.
# valresp[isnan(valresp)]=0.

tdmlp = TDMLP(layer_sizes, [1, 12, 1], act, gain=.1);
# tdmlp = TDMLP(layer_sizes, [1, 12, 1], act, gain=1., datatype=Float64);

tdmlp = rmsproptrain(tdmlp, trainstim, trainresp, learning_rate=.0001, learning_rate_factor=.999, maxiter=1, batch_size = 400, loss=poisson_loss, maxnorm=3., minadapt=1., xval=valstim, tval=valresp);
tdmlp = TDMLP(layer_sizes, [1, 12, 1], act, gain=.1);
tdmlp = rmsproptrain(tdmlp, trainstim, trainresp, learning_rate=.0001, learning_rate_factor=.999, maxiter=200, batch_size = 400, loss=poisson_loss, maxnorm=3., minadapt=1., xval=valstim, tval=valresp);

tdmlp.net = tdmlp.net + 0.;
tdmlp.net[1].b[:]= 1.;
tdmlp.net[2].b[:]= 1.;
tdmlp.net[3].b[:]= 1.;

fitpoints = [101:200];


x_batch, t_batch = mini_batch_init(trainstim,trainresp,fitpoints, tdmlp);
mini_batch!(trainstim,trainresp,x_batch,t_batch,fitpoints, tdmlp);

g = finite_diff_m(tdmlp, x_batch, t_batch, poisson_loss);

D = deltas_init(tdmlp, 100);
∇,δ = backprop(tdmlp.net,x_batch,t_batch,poisson_lossd,D.deltas,ones(1,100), tdmlp.gain);
g2 = float64(vec(flatten_net(∇)));

y = prop(tdmlp, trainstim);

y2 = mprop(tdmlp.net, x_batch, tdmlp.gain);

[vec(y[fitpoints]) vec(y2)]

@time tdmlp = rmsproptrain(tdmlp, trainstim, trainresp, learning_rate=.0001, learning_rate_factor=.999, maxiter=1, batch_size = 400, loss=poisson_loss, maxnorm=3., minadapt=1., xval=valstim, tval=valresp);


tdmlp = rmsproptrain(tdmlp, trainstim, trainresp, learning_rate=.0001, learning_rate_factor=.999, maxiter=100, batch_size = 400, loss=poisson_loss, maxnorm=3., minadapt=1., xval=valstim, tval=valresp);


@time tdmlp = rmsproptrain(tdmlp, trainstim, trainresp, learning_rate=.0001, learning_rate_factor=.999, maxiter=1, batch_size = 400, loss=poisson_loss, maxnorm=1., minadapt=1., xval=valstim, tval=valresp);

@profile tdmlp = rmsproptrain(tdmlp, trainstim, trainresp, learning_rate=.0001, learning_rate_factor=.999, maxiter=1, batch_size = 400, loss=poisson_loss, maxnorm=1., minadapt=1., xval=valstim, tval=valresp);

tdmlp = TDMLP(layer_sizes, [1, 12, 1], act, datatype=Float32);

tdmlp = rmsproptrain(tdmlp, trainstim, trainresp, learning_rate=.0001, learning_rate_factor=.999, maxiter=500, batch_size = 400, loss=poisson_loss, maxnorm=1., minadapt=1., xval=valstim, tval=valresp)

tdmlp = rmsproptrain(tdmlp, trainstim, trainresp, learning_rate=.0001, learning_rate_factor=.992, maxiter=500, batch_size = 400, loss=poisson_loss, maxnorm=1., minadapt=1., xval=valstim, tval=valresp)

fitpoints = sample_epoch(200000, 400)
x_batch,t_batch = mini_batch(trainstim,trainresp, fitpoints[:,i], mlp)

∇,δ = backprop(tdmlp.net,x_batch,t_batch,poisson_lossd)
