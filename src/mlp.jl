# Types and function definitions for multi-layer perceptrons

using ArrayViews

abstract MLNN

type NNLayer{T}
    w::AbstractMatrix{T}
    b::AbstractVector{T}
    a::Function
    ad::Function
end

type MLP <: MLNN
    net::Vector{NNLayer}
    dims::Vector{(Int,Int)}  # topology of net
    buf::AbstractVector      # in-place data store
    offs::Vector{Int}    # indices into in-place store
    trained::Bool
    gain::FloatingPoint
end

# In all operations between two NNLayers, the activations functions are taken from the first NNLayer
*(l::NNLayer, x::Array{Float64}) = l.w*x .+ l.b
*(l::NNLayer, x::Array{Float32}) = l.w*x .+ l.b
.*(c::FloatingPoint, l::NNLayer) = NNLayer(c.*l.w, c.*l.b, l.a, l.ad)
.*(l::NNLayer, c::FloatingPoint) = NNLayer(l.w.*c, l.b.*c, l.a, l.ad)
.*(l::NNLayer, m::NNLayer) = NNLayer(l.w.*m.w, l.b.*m.b, l.a, l.ad)
*(l::NNLayer, m::NNLayer) = NNLayer(l.w.*m.w, l.b.*m.b, l.a, l.ad)
/(l::NNLayer, m::NNLayer) = NNLayer(l.w./m.w, l.b./m.b, l.a, l.ad)
^(l::NNLayer, c::FloatingPoint) = NNLayer(l.w.^c, l.b.^c, l.a, l.ad)
-(l::NNLayer, m::NNLayer) = NNLayer(l.w .- m.w, l.b .- m.b, l.a, l.ad)
.-(l::NNLayer, c::FloatingPoint)  = NNLayer(l.w .- c, l.b .- c, l.a, l.ad)
.-(c::FloatingPoint, l::NNLayer)  = NNLayer(c .- l.w, c .- l.b, l.a, l.ad)
+(l::NNLayer, m::NNLayer) = NNLayer(l.w + m.w, l.b + m.b, l.a, l.ad)
.+(l::NNLayer, c::FloatingPoint)  = NNLayer(l.w .+ c, l.b .+ c, l.a, l.ad)
.+(c::FloatingPoint, l::NNLayer)  = NNLayer(l.w .+ c, l.b .+ c, l.a, l.ad)


.*(net::Array{NNLayer}, c::FloatingPoint)  =  begin
												net2=deepcopy(net)
												for l in net2
											    	l.w = l.w .* c
											        l.b = l.b .* c
											    end
											    net2
											end
.*(c::FloatingPoint, net::Array{NNLayer})  =  begin
												net2=deepcopy(net)
												for l in net2
											    	l.w = c.* l.w
											        l.b = c.*l.b
											    end
											    net2
											end
./(net::Array{NNLayer}, c::FloatingPoint)  =  begin
												net2=deepcopy(net)
												for l in net2
											    	l.w = l.w ./ c
											        l.b = l.b ./ c
											    end
											    net2
											end
./(c::FloatingPoint, net::Array{NNLayer})  =  begin
												net2=deepcopy(net)
												for l in net2
											    	l.w = c./ l.w
											        l.b = c./l.b
											    end
											    net2
											end
.+(net::Array{NNLayer}, c::FloatingPoint)  =  begin
												net2=deepcopy(net)
												for l in net2
											    	l.w = l.w .+ c
											        l.b = l.b .+ c
											    end
											    net2
											end
.+(c::FloatingPoint,net::Array{NNLayer})  =  begin
												net2=deepcopy(net)
												for l in net2
											    	l.w = l.w .+ c
											        l.b = l.b .+ c
											    end
											    net2
											end
.-(net::Array{NNLayer}, c::FloatingPoint)  =  begin
												net2=deepcopy(net)
												for l in net2
											    	l.w = l.w .- c
											        l.b = l.b .- c
											    end
											    net2
											end
.-(c::FloatingPoint, net::Array{NNLayer})  =  begin
												net2=deepcopy(net)
												for l in net2
											    	l.w = c .- l.w
											        l.b = c .- l.b
											    end
											    net2
											end
import Base.sign
sign(l::NNLayer) = NNLayer(sign(l.w), sign(l.b), l.a, l.ad)

function Base.min(net::Array{NNLayer}, c::FloatingPoint)
    for l in net
    	l.w = min(l.w,c)
        l.b = min(l.b,c)
    end
    net
end

function Base.max(net::Array{NNLayer}, c::FloatingPoint)
    for l in net
    	l.w = max(l.w,c)
        l.b = max(l.b,c)
    end
    net
end

function Base.show(io::IO, l::NNLayer)
    print(io, summary(l),":\n")
    print(io, "activation functions:\n")
    print(io, l.a,", ",l.ad,"\n")
    print(io, "node weights:\n",l.w,"\n")
    print(io, "bias weights:\n",l.b)
end

# For the NNLayer type, given a set of layer dimensions,
# compute offsets into the flattened vector.
# We only have to do this once, when a net is initialized.
function calc_offsets(::Type{NNLayer}, dims)
	nlayers = length(dims)
	offs = Array(Int,nlayers)
	sumd = 0	# current data cursor
	for i = 1 : nlayers
		sumd += prod(dims[i])+dims[i][1]
		offs[i] = sumd
	end
	offs
end

function MLP(layer_sizes::Vector{Int}, act::Vector{Function}; gain=1., datatype = Float32)
	# some initializations
	nlayers = length(layer_sizes) - 1
	dims = [(layer_sizes[i+1],layer_sizes[i]) for i in 1:nlayers]

    # generate vector of activation derivatives
    actd = Function[]
    for f in act # if native deriv not found then calculate one with ForwardDiff
        d = haskey(derivs,f) ? derivs[f] : autodiff(f)
        push!(actd,d)
    end

	# offsets into the parameter vector
	offs = calc_offsets(NNLayer, dims)

	# our single data vector
	buf = datatype[]
	for ii = 1:length(layer_sizes)-1
		nw = layer_sizes[ii]*layer_sizes[ii+1]
		nb = layer_sizes[ii+1]
		buf = [buf; 2.*(rand(datatype, nw,1)-.5).*sqrt(6.)./ sqrt(layer_sizes[ii] + layer_sizes[ii+1]); zeros(datatype, nb,1)]
	end
	buf = vec(buf)

	net = [NNLayer(Array(eltype(buf),0,0),Array(eltype(buf),0),act[i],actd[i]) for i=1:nlayers]

	mlp = MLP(net, dims, buf, offs, false, gain)
	unflatten_net!(mlp, buf)

	mlp
end

# Given a flattened vector (buf), update the neural
# net so that each weight and bias vector points into the
# offsets provided by offs
function unflatten_net!(mlp::MLP, buf::AbstractVector)
	mlp.buf = buf

	for i = 1 : length(mlp.net)
		toff = i > 1 ? mlp.offs[i-1] : 0
		tdims = mlp.dims[i]
		lenw = prod(tdims)
		mlp.net[i].w = reshape_view(view(buf, toff+1:toff+lenw), tdims)
		toff += lenw
		mlp.net[i].b = view(buf, toff+1:toff+tdims[1])
	end
end
