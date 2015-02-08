## A collection of commonly-used activation functions
## To enable dropout within the existing framework, the second return value of each 
## activation function is either an index of dropped units or NaN

## Logistic function & derivative
function logis(x)
	1. ./(1 .+ exp(-x)), NaN
end

logisd(x,idx) = exp(x) ./ ((1. .+ exp(x)).^2.)

## Logistic function & derivative is numerical safety
function logissafe(x)
	logis(x)
end
logissafed(x,idx) = logisd(min(x,400.0),idx)

## Soft-max function, mainly used in output layer so the derivative is not needed with cross-entropy loss
function softmaxact(x)
	ex = exp(x.-maximum(x,1))
	a = ex ./ sum(ex,1)
	a, NaN
end

## Identity function and derivative
function ident(x)
	x, NaN
end
identd(x,idx) = 1.

## Hyperbolic tangent function and derivative
function tanhact(x)
	tanh(x), NaN
end
tanhactd(x,idx) = sech(x).^2.

## Exponential function and derivative
function expact(x)
	exp(x), NaN
end
expactd(x,idx) = exp(x)

## Soft rectifier function & derivative
function srelu(x)
	a = similar(x)
	for ii=1:prod(size(x))
		a[ii] = log(1. + exp(x[ii]))
	end
	a, NaN
end

function srelud(x,idx)
	a = similar(x)
	for ii=1:prod(size(x))
		a[ii] = 1 /(1. + exp(-x[ii]))
	end
	a
end

## A parameterized soft rectifier function & derivative which 
## can be used to make a custom activation function as below.
## The parameter k controls the softness of the rectification.
##
## srelu10(x) = sreluk(x,10)
## srelu10d(x,idx) = srelukd(x,idx,10)
## merge!(derivs, [srelu10   => srelu10d])
function sreluk(x,k)
	a = similar(x)
	for ii=1:prod(size(x))
		a[ii] = log(1. + exp(k.*x[ii]))./k
	end
	a, NaN
end

function srelukd(x,idx,k)
	a = similar(x)
	for ii=1:prod(size(x))
		a[ii] = 1 /(1. + exp(-k.*x[ii]))
	end
	a
end


## Rectified linear activation function and derivative
function relu(x)
	a = similar(x)
	for ii=1:prod(size(x))
		a[ii] = x[ii] > 0. ? x[ii] : 0.
	end
	a,NaN
end

function relud(x,idx)
	a = similar(x)
	for ii=1:prod(size(x))
		a[ii] = x[ii] > 0. ? 1.0 : 0.0
	end
	a
end

## Noisy rectified linear activation function and derivative, with Poisson-like noise
function nrelu(x)
	a = similar(x)
	for ii=1:prod(size(x))
		a[ii] = x[ii] > 0. ? max(0. , x[ii] + sqrt(x[ii]).*randn()) : 0.
	end
	a,NaN
end

nrelud(x,idx) = relud(x,idx)

## Rectified linear activation function and derivative, with parameterized
## dropout which can be used to make a custom activation function as below.
## The parameter p controls the amount of units retained in each minibatch.
##
## do8relu(x) = doprelu(x,.8)
## do8relud(x,idx) = doprelud(x,idx,.8)
## merge!(derivs, [do8relu   => do8relud])

function doprelu(x,p)
	a = similar(x)
	for ii=1:prod(size(x))
		a[ii] = x[ii] > 0. ? x[ii] : 0.
	end
	idx = randperm(size(a,1))
  	li = length(idx)
  	p1 = round((1 - p) .* li)
	a[idx[1:p1],:,:] = 0.
	a[idx[(p1+1):end],:,:] .*= li./(li-p1)
	a, idx
end

function doprelud(x,idx,p)
	a = similar(x)
  	li = length(idx)
  	p1 = round((1 - p) .* li)
	for ii=1:prod(size(x))
		a[ii] = x[ii] > 0. ? 1.0 : 0.0
	end
	a[idx[1:p1],:,:] = 0.
	a[idx[(p1+1):end],:,:] .*= li./(li-p1)
	a
end

## Noisy rectified linear activation function and derivative, with Poisson-like noise and 
## parameterized dropout which can be used to make a custom activation function as below.
## The parameter p controls the amount of units retained in each minibatch.
##
## do8nrelu(x) = dopnrelu(x,.8)
## do8nrelud(x,idx) = dopnrelud(x,idx,.8)
## merge!(derivs, [do8nrelu   => do8nrelud])

function dopnrelu(x,p)
	a = similar(x)
	for ii=1:prod(size(x))
		a[ii] = x[ii] > 0. ? max(0. , x[ii] + sqrt(x[ii]).*randn()) : 0.
	end
	idx = randperm(size(a,1))
  	li = length(idx)
  	p1 = round((1 - p) .* li)
	a[idx[1:p1],:,:] = 0.
	a[idx[(p1+1):end],:,:] .*= li./(li-p1)
	a, idx
end

function dopnrelud(x,idx,p)
	a = similar(x)
  	li = length(idx)
  	p1 = round((1 - p) .* li)
	for ii=1:prod(size(x))
		a[ii] = x[ii] > 0. ? 1.0 : 0.0
	end
	a[idx[1:p1],:,:] = 0.
	a[idx[(p1+1):end],:,:] .*= li./(li-p1)
	a
end



# dictionary of commonly-used activation derivatives
derivs = Dict{Function, Function}([
                                   logis 	 => logisd,
                                   logissafe => logissafed,
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
                                   softmaxact     => xent_loss
                                   ])

# automatic differentiateion with ForwardDiff.jl
# due to limitations of ForwardDiff.jl, this function
# will only produce derivatives with Float64 methods
function autodiff(activ::Function)
    f(x) = activ(x[1])
    forwarddiff_derivative(x::Float64, idx) = forwarddiff_gradient(f,Float64)([x])[1]
    return forwarddiff_derivative
end
