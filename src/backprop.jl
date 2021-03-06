## high level forward prop functions
function prop(mlp::MLP,x)
    # if trained use normal relu rather than noisy version
	if mlp.trained
		acts = Function[]
		for l in mlp.net
			push!(acts,l.a)
			if string(Base.function_name(l.a))[end-3:end]=="relu"
				l.a = relu
			end
		end
	end
	a = prop(mlp.net,x,mlp.gain)
    # replace activation functions with originals
	if mlp.trained
		for l in mlp.net
			l.a = shift!(acts)
		end
	end
	a
end

function prop(tdmlp::TDMLP,x)
    # if trained use normal relu rather than noisy version
	if tdmlp.trained
		acts = Function[]
		for l in tdmlp.net
			push!(acts,l.a)
			if string(Base.function_name(l.a))[end-3:end]=="relu"
				l.a = relu
			end
		end
	end
	a = prop(tdmlp.net,x,tdmlp.delays,tdmlp.gain)
    a[:,1:tdmlp.delays,:]=NaN
    # replace activation functions with originals
	if tdmlp.trained
		for l in tdmlp.net
			l.a = shift!(acts)
		end
	end
	a[:,:,1]
end

## low level prop functions recursively forware propagate input

## prop function used with mlp, with output gain
function prop(net, x, gain::FloatingPoint)
  net[end].a(gain.* (net[end] * prop(net[1:end-1], x)))[1]
end

## prop function used with mlp
function prop(net, x)
    if length(net) == 0 # First layer
        x
    else # Intermediate layers
        net[end].a(net[end] * prop(net[1:end-1], x))[1]
    end
end

## prop function used with tdmlp, with output gain
function prop(net, x, delays::Int, gain::FloatingPoint)
  net[end].a(gain.* (net[end] * prop(net[1:end-1], x, delays)))[1]
end

## prop function used with tdmlp
function prop(net, x, delays::Int)
  if length(net) == 1 # First hidden layer, create 3d data to pass to rest of net

        z = zeros(eltype(x), size(net[1].w,1), size(x,2));
        for ti=1:size(net[1].w,3)
            z += view(net[1].w,:,:,ti)*[zeros(eltype(x), size(x,1), ti-1) x[:,1:end-ti+1]]
        end
        z .+= (net[1].b + 0.)

        z2 = zeros(eltype(x), size(z,1), size(z,2), delays+1);
        for ii=0:delays
            z2[:,:,ii+1] = [zeros(eltype(x), size(z,1), ii) z[:,1:end-ii]]
        end

        net[end].a(z2)[1]

    else                    # Intermediate layers
        net[end].a(net[end] * prop(net[1:end-1], x, delays))[1]
    end
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
function backprop{T}(net::Vector{T}, x, t, lossd::Function, deltas, weights, gain)  ## Backprop for non-cannonical activation/loss function pairs
    if length(net) == 0   	# Final layer
        δ  = lossd(x,t)    	# Error (δ) is simply difference with target
        δ[isnan(δ)] = 0.
        grad = T[]        	# Initialize weight gradient array
    elseif length(net) == 1                	# Last hidden layer
    	l = net[1]
        h = gain.*(l * x)
        y,idx = l.a(h)
        grad,δ = backprop(net[2:end], y, t, lossd, deltas[2:end], weights, gain)
        δ = l.ad(h,idx) .* δ
        δ = gain.*(weights.* δ)
        unshift!(grad,typeof(l)(δ*x',vec(sum(sum(δ,2),3)),exp,exp))  # Weight gradient
        δ = errprop!(l.w, δ, deltas[1])
    else
        l = net[1]
        h = l * x
        y,idx = l.a(h)
        grad,δ = backprop(net[2:end], y, t, lossd, deltas[2:end], weights, gain)
        δ = l.ad(h,idx) .* δ
        unshift!(grad,typeof(l)(δ*x',vec(sum(sum(δ,2),3)),exp,exp))  # Weight gradient
        δ = errprop!(l.w, δ, deltas[1])
    end
    return grad,δ
end


function backprop{T}(net::Vector{T}, x, t, lossd::Array{None,1}, deltas, weights, gain)  ## Backprop for cannonical activation/loss function pairs
    if length(net) == 0   	# Final layer
        δ  = x .- t     	# Error (δ) is simply difference with target
        δ[isnan(δ)] = 0.
        grad = T[]        	# Initialize weight gradient array
    elseif length(net) == 1                	# Last hidden layer
    	l = net[1]
        h = gain.*(l * x)           # Not a typo!
        y,idx = l.a(h)
        grad,δ = backprop(net[2:end], y, t, lossd, deltas[2:end], weights, gain)
        δ = gain.*(weights.* δ)
        unshift!(grad,typeof(l)(δ*x',vec(sum(sum(δ,2),3)),exp,exp))  # Weight gradient
        δ = errprop!(l.w, δ, deltas[1])

    else
        l = net[1]
        h = l * x
        y,idx = l.a(h)
        grad,δ = backprop(net[2:end], y, t, lossd, deltas[2:end], weights, gain)
        δ = l.ad(h,idx) .* δ
        unshift!(grad,typeof(l)(δ*x',vec(sum(sum(δ,2),3)),exp,exp))  # Weight gradient
        δ = errprop!(l.w, δ, deltas[1])
    end
    return grad,δ
end



## Error back-propagation for mlp, with preallocated memory in deltas
function errprop!{T}(w::Array{T,2}, d::Array{T,2},deltas)
    deltas.d[:] = w' * d
end


## Error back-propagation for tdmlp, with preallocated memory in deltas
function errprop!{T}(w::Array{T,3}, d::Array{T,3}, deltas)
	deltas.d[:] = 0.
	for ti=1:size(w,3), ti2 = 1:size(d,3)
    	Base.LinAlg.BLAS.gemm!('T', 'N', one(T), view(w,:,:,ti), view(d,:,:,ti2), one(T), view(deltas.d,:,:,ti+ti2-1))
	end
	deltas.d
end



## Functions for finite difference gradient checking

function mprop{T}(net::Vector{T}, x, gain) 
    if length(net) == 1                	# Last hidden layer
    	  l = net[1]
        h = gain.*(l * x)           # Not a typo!
        y,idx = l.a(h)
    else
        l = net[1]
        h = l * x           # Not a typo!
        y,idx = l.a(h)
        y = mprop(net[2:end], y, gain)
    end
    return y
end


function finite_diff(mlp,x,t,loss)
  w = flatten_net(mlp)
  w1 = deepcopy(w)

  g = zeros(size(w));

  for ii = 1:length(w)
    w1[:] = deepcopy(w);
    w1[ii] = w1[ii] + 1e-8;
    unflatten_net!(mlp, w1)
    #y=mprop(mlp.net,x)
    y=prop(mlp,x)

    err1 = loss(vec(y),vec(t));

    w1[:] = deepcopy(w);
    w1[ii] = w1[ii] - 1e-8;
    unflatten_net!(mlp, w1)
    y=prop(mlp,x)
    #y=mprop(mlp.net,x)
    err2 = loss(vec(y),vec(t));

    g[ii] =(err1 - err2)./2e-8;
  end
  g
end

function finite_diff_m(mlp,x,t,loss)
  w = flatten_net(mlp)
  w1 = deepcopy(w)

  g = zeros(size(w));

  for ii = 1:length(w)
    w1[:] = deepcopy(w);
    w1[ii] = w1[ii] + 1e-8;
    unflatten_net!(mlp, w1)
    y=mprop(mlp.net,x,mlp.gain)
    #y=prop(mlp,x)

    err1 = loss(vec(y),vec(t));

    w1[:] = deepcopy(w);
    w1[ii] = w1[ii] - 1e-8;
    unflatten_net!(mlp, w1)
    #y=prop(mlp,x)
    y=mprop(mlp.net,x,mlp.gain)
    err2 = loss(vec(y),vec(t));

    g[ii] =(err1 - err2)./2e-8;
  end
  g
end
