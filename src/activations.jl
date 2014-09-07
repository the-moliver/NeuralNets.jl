# collection of commonly-used activation functions
function logis(x)
	1 ./(1 .+ exp(-x)), NaN
end
logisd(x,idx) = exp(x) ./ ((1 .+ exp(x)).^2)

function logissafe(x) 
	logis(x), NaN
end
logissafed(x,idx) = logisd(min(x,400.0))

function softmaxact(x)
	ex = exp(x)
	ex ./ sum(ex,1), NaN
end

function softmaxactsafe(x) 
	softmaxact(min(x,400.0))
end

function srelu(x) 
	log(1 .+ exp(x)), NaN
end

srelud(x,idx) = 1 ./(1 .+ exp(-x))


function relu(x) 
	max(0.,x), NaN
end

relud(x,idx) = (x .> 0) + 0.

function nrelu(x) 
	a = max(0.,x)
	a += sqrt(a).*randn(size(a))
	a = max(0.,a)
	a, NaN
end

nrelud(x,idx) = (x .> 0) + 0.

function donrelu(x) 
	a = max(0.,x)
	a += sqrt(a).*randn(size(a))
	a = max(0.,a)
	idx = randperm(size(a,1))
	a[idx[1:(.5*length(idx))],:,:] = 0.
	a[idx[(.5*length(idx)+1):end],:,:] .*= 2.0
	a, idx
end

function donrelud(x,idx) 
	a = (x .> 0) + 0.
	a[idx[1:(.5*length(idx))],:,:] = 0.
	a[idx[(.5*length(idx)+1):end],:,:] .*= 2.0
	a
end

function ident(x) 
	x, NaN
end
identd(x,idx) = 1

function tanhact(x)
	tanh(x), NaN
end
tanhactd(x,idx) = sech(x).^2

function expact(x)
	exp(x), NaN
end
expactd(x,idx) = exp(x)

# dictionary of commonly-used activation derivatives
derivs = Dict{Function, Function}([
                                   relu      => relud,
                                   donrelu   => donrelud,
                                   srelu     => srelud,
                                   nrelu     => nrelud, 
                                   ident     => identd, 
                                   tanhact   => tanhactd,
                                   expact    => expactd
                                   ])


# dictionary of cannonical activation/loss function pairs
cannonical = Dict{Function, Function}([
                                   logis     => log_loss, 
                                   logissafe => log_loss,
                                   ident     => squared_loss,
                                   expact    => poisson_loss,
                                   softmaxact     => xent_loss,
                                   softmaxactsafe => xent_loss
                                   ])

# automatic differentiateion with ForwardDiff.jl
# due to limitations of ForwardDiff.jl, this function
# will only produce derivatives with Float64 methods
function autodiff(activ::Function)
    f(x) = activ(x[1])
    forwarddiff_derivative(x::Float64) = forwarddiff_gradient(f,Float64)([x])[1]
    return forwarddiff_derivative
end