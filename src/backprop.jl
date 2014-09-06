function prop(net, x)
	if length(net) == 0 # First layer
		x
	else # Intermediate layers
		net[end].a(net[end] * prop(net[1:end-1], x))[1]
	end
end

function prop(net, x, delays::Int)
	if length(net) == 0 # Input layer
		x
	elseif length(net) == 1 # First hidden layer, create 3d data to pass to rest of net

		z = zeros(size(net[1].w,1), size(x,2));
		for ti=1:size(net[1].w,3)
			z += view(net[1].w,:,:,ti)*[zeros(size(x,1), ti-1) x[:,1:end-ti+1]]
		end
		z .+= (net[1].b + 0.)

		z2 = zeros(size(z,1), size(z,2), delays+1);
		for ii=0:delays
			z2[:,:,ii+1] = [zeros(size(z,1), ii) z[:,1:end-ii]]
		end
		z2

	else                    # Intermediate layers
		net[end].a(net[end] * prop(net[1:end-1], x, delays))[1]
	end
end

function prop(mlp::MLP,x)
	if mlp.trained
		acts = Function[]
		for l in mlp.net
			push!(acts,l.a)
			if l.a == nrelu || l.a == donrelu
				l.a = relu
			end
		end
	end
	a = prop(mlp.net,x)
	if mlp.trained
		for l in mlp.net
			l.a = shift!(acts)
		end
	end
	a
end

function prop(tdmlp::TDMLP,x)
	if tdmlp.trained
		acts = Function[]
		for l in tdmlp.net
			push!(acts,l.a)
			if l.a == nrelu || l.a == donrelu
				l.a = relu
			end
		end
	end
	a = prop(tdmlp.net,x,tdmlp.delays)
	if tdmlp.trained
		for l in tdmlp.net
			l.a = shift!(acts)
		end
	end
	a
end

# add some 'missing' functionality to ArrayViews
function setindex!{T}(dst::ContiguousView, src::Array{T}, idx::UnitRange)
	offs = dst.offset
	dst.arr[offs+idx.start:offs+idx.stop] = src
end

# backpropagation;
# with memory for gradients pre-allocated.
# (gradients returned in stor)
function backprop!{T}(net::Vector{T}, stor::Vector{T}, x, t)
	if length(net) == 0 # Final layer
		# Error is simply difference with target
		r = x .- t
		r[find(isnan(r))]=0
		r
	else # Intermediate layers
		# current layer
		l = net[1]

		# forward activation
		h = l * x
		y,idx = l.a(h)

		# compute error recursively
		δ = l.ad(h,idx) .* backprop!(net[2:end], stor[2:end], y, t)

		# calculate weight and bias gradients
		stor[1].w[:] = vec(δ*x')
		stor[1].b[:] = sum(δ,2)

		# propagate error
		l.w' * δ
	end
end

# backprop(net,x,t) returns array of gradients and error for net 
# todo: make gradient unshift! section more generic
function backprop{T}(net::Vector{T}, x, t; lossd = squared_lossd)
    if length(net) == 0   	# Final layer
        δ  = lossd(x,t)     	# Error (δ) is simply difference with target
        grad = T[]        	# Initialize weight gradient array
    else                	# Intermediate layers
        l = net[1]
        h = l * x           # Not a typo!
        y,idx = l.a(h)
        grad,δ = backprop(net[2:end], y, t)
        δ = l.ad(h,idx) .* δ
        if any(isnan(δ))
        	print(δ)
        	error("Nans are starting")
	    end
        unshift!(grad,typeof(l)(δ*x',vec(sum(δ,2)),exp,exp))  # Weight gradient
        δ = l.w' * δ
    end
    return grad,δ
end