# collection of commonly-used loss functions and their derivatives

# y = predicted value, t = expected value

squared_loss(y, t) = 0.5 * nansum((y .- t).^2.) # L(y, t) = .5 || y - t ||^2 = .5 (y - t)^2
squared_lossd(y, t) = y .- t # d/dx L(y, t) = (y - t)


xent_loss(y, t) = -nansum(t.*log(max(y, eps(typeof(y[1])))))  ## fix to prevent log(0)

quartic_loss(y, t) = 0.25 * nansum(abs(y .- t).^4)
quartic_lossd(y, t) = (y .- t).^3

function poisson_loss(y, t)
	y[y.==0] = eps(typeof(y[1]))
	nansum(y .- t.*log(y))
end

function poisson_lossd(y, t)
	y[y.==0] = eps(typeof(y[1]))
	d = 1 .- (t ./ y)
end

linear_loss(y, t) = nansum(abs(y .- t)) # L(y, t) = || y .- t ||_1 = sum(abs(y .- t))
linear_lossd(y, t) = sign(y .- t)

hinge_loss(y, t, m=1.0) = nansum(max(0, m - (y .* t))) # max(0, m - y t)
hinge_lossd(y, t, m=1.0) = - t .* (m > t .* y) # if (m > y t) then - t, else 0

log_loss(y, t) = nansum(- ((t .* log(y)) + ((1 - t) .* log(1 - y)))) # - t * log(y) - (1 - t) * log(1 - y)
log_lossd(y, t) = (t .- y) ./ ((y .- 1) .* y) # (t - y) / ((y - 1) * y)

# dictionary of commonly-used loss derivatives
lossderivs = Dict{Function, Function}([
                                   squared_loss      => squared_lossd,
                                   linear_loss       => linear_lossd,
                                   poisson_loss      => poisson_lossd,
                                   quartic_loss      => quartic_lossd,
                                   hinge_loss        => hinge_lossd,
                                   log_loss          => log_lossd
                                   ])

function nansum(A::Array)
	s = zero(eltype(A))
	for val in A
		if !isnan(val)
			s += val
		end
	end
	return s
end
