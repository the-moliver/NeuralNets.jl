# collection of commonly-used activation functions
logis(x) = 1 ./(1 .+ exp(-x))
logisd(x) = exp(x) ./ ((1 .+ exp(x)).^2)

logissafe(x) = logis(x)
logissafed(x) = logisd(min(x,400.0))

srelu(x) = log(1 .+ exp(x))
srelud(x) = 1 ./(1 .+ exp(-x))

relu(x) = max(0.,x)
relud(x) = (x .> 0) + 0.

function nrelu(x) 
	a = max(0.,x)
	a += sqrt(a).*randn(size(a))
	a = max(0.,a)
end

nrelud(x) = (x .> 0) + 0.

function donrelu(x) 
	a = max(0.,x)
	a += sqrt(a).*randn(size(a))
	a = max(0.,a)
	idx = randperm(size(a,1))
	a[idx[1:(.5*length(idx))],:] = 0.
	a[idx[(.5*length(idx)+1):end],:] .*= 2.0
	a
end

donrelud(x) = (x .> 0) .* 2.

ident(x) = x
identd(x) = 1

tanhd(x) = sech(x).^2
expd(x) = exp(x)

# dictionary of commonly-used activation derivatives
derivs = Dict{Function, Function}([
                                   logis     => logisd, 
                                   logissafe => logissafed,
                                   relu      => relud,
                                   donrelu   => donrelud,
                                   srelu     => srelud,
                                   nrelu     => nrelud, 
                                   ident     => identd, 
                                   tanh      => tanhd,
                                   exp      => expd
                                   ])

# automatic differentiateion with ForwardDiff.jl
# due to limitations of ForwardDiff.jl, this function
# will only produce derivatives with Float64 methods
function autodiff(activ::Function)
    f(x) = activ(x[1])
    forwarddiff_derivative(x::Float64) = forwarddiff_gradient(f,Float64)([x])[1]
    return forwarddiff_derivative
end