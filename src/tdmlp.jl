# Types and function definitions for multi-layer perceptrons

using ArrayViews

type TDNNLayer{T}
    w::AbstractMatrix{T}
    b::AbstractVector{T}
    a::Function
    ad::Function
end

type TDMLP
    net::Vector{TDNNLayer}
    dims::Vector{(Int,Int)}  # topology of net
    buf::AbstractVector      # in-place data store
    offs::Vector{Int}    # indices into in-place store
    trained::Bool
end

# In all operations between two TDNNLayers, the activations functions are taken from the first TDNNLayer
#*(l::TDNNLayer, x::Array{Float64}) = l.w*x .+ l.b

*(l::TDNNLayer, x::Array{Float64}) = begin
	for ti=1:size(z,3)-size(W,3)+1
    	for ti2=1:size(W,3)
      		z2(:,:,ti) = z2(:,:,ti) + z(:,:,ti+ti2-1) * W(:,:,ti2);
    	end
  	end
end


#.*(c::FloatingPoint, l::TDNNLayer) = TDNNLayer(c.*l.w, c.*l.b, l.a, l.ad)
#.*(l::TDNNLayer, c::FloatingPoint) = TDNNLayer(l.w.*c, l.b.*c, l.a, l.ad)
#.*(l::TDNNLayer, m::TDNNLayer) = TDNNLayer(l.w.*m.w, l.b.*m.b, l.a, l.ad)
#*(l::TDNNLayer, m::TDNNLayer) = TDNNLayer(l.w.*m.w, l.b.*m.b, l.a, l.ad)
/(l::TDNNLayer, m::TDNNLayer) = TDNNLayer(l.w./m.w, l.b./m.b, l.a, l.ad)
^(l::TDNNLayer, c::FloatingPoint) = TDNNLayer(l.w.^c, l.b.^c, l.a, l.ad)
-(l::TDNNLayer, m::TDNNLayer) = TDNNLayer(l.w .- m.w, l.b .- m.b, l.a, l.ad)
.-(l::TDNNLayer, c::FloatingPoint)  = TDNNLayer(l.w .- c, l.b .- c, l.a, l.ad)
.-(c::FloatingPoint, l::TDNNLayer)  = TDNNLayer(c .- l.w, c .- l.b, l.a, l.ad)
+(l::TDNNLayer, m::TDNNLayer) = TDNNLayer(l.w + m.w, l.b + m.b, l.a, l.ad)
.+(l::TDNNLayer, c::FloatingPoint)  = TDNNLayer(l.w .+ c, l.b .+ c, l.a, l.ad)
.+(c::FloatingPoint, l::TDNNLayer)  = TDNNLayer(l.w .+ c, l.b .+ c, l.a, l.ad)


#.*(net::Array{TDNNLayer}, c::FloatingPoint)  =  begin
												net2=deepcopy(net)
												for l in net2
											    	l.w = l.w .* c
											        l.b = l.b .* c
											    end
											    net2
											end
#.*(c::FloatingPoint, net::Array{TDNNLayer})  =  begin
												net2=deepcopy(net)
												for l in net2
											    	l.w = c.* l.w
											        l.b = c.*l.b
											    end
											    net2
											end
./(net::Array{TDNNLayer}, c::FloatingPoint)  =  begin
												net2=deepcopy(net)
												for l in net2
											    	l.w = l.w ./ c
											        l.b = l.b ./ c
											    end
											    net2
											end
./(c::FloatingPoint, net::Array{TDNNLayer})  =  begin
												net2=deepcopy(net)
												for l in net2
											    	l.w = c./ l.w
											        l.b = c./l.b
											    end
											    net2
											end
.+(net::Array{TDNNLayer}, c::FloatingPoint)  =  begin
												net2=deepcopy(net)
												for l in net2
											    	l.w = l.w .+ c
											        l.b = l.b .+ c
											    end
											    net2
											end
.+(c::FloatingPoint,net::Array{TDNNLayer})  =  begin
												net2=deepcopy(net)
												for l in net2
											    	l.w = l.w .+ c
											        l.b = l.b .+ c
											    end
											    net2
											end											
.-(net::Array{TDNNLayer}, c::FloatingPoint)  =  begin
												net2=deepcopy(net)
												for l in net2
											    	l.w = l.w .- c
											        l.b = l.b .- c
											    end
											    net2
											end
.-(c::FloatingPoint, net::Array{TDNNLayer})  =  begin
												net2=deepcopy(net)
												for l in net2
											    	l.w = c .- l.w
											        l.b = c .- l.b
											    end
											    net2
											end											
import Base.sign
sign(l::TDNNLayer) = TDNNLayer(sign(l.w), sign(l.b), l.a, l.ad)

function Base.min(net::Array{TDNNLayer}, c::FloatingPoint)
    for l in net
    	l.w = min(l.w,c)
        l.b = min(l.b,c)
    end
    net
end

function Base.max(net::Array{TDNNLayer}, c::FloatingPoint)
    for l in net
    	l.w = max(l.w,c)
        l.b = max(l.b,c)
    end
    net
end

function Base.show(io::IO, l::TDNNLayer)
    print(io, summary(l),":\n")
    print(io, "activation functions:\n")
    print(io, l.a,", ",l.ad,"\n")
    print(io, "node weights:\n",l.w,"\n")
    print(io, "bias weights:\n",l.b)
end

# For the TDNNLayer type, given a set of layer dimensions,
# compute offsets into the flattened vector.
# We only have to do this once, when a net is initialized.
function calc_offsets(::Type{TDNNLayer}, dims)
	nlayers = length(dims)
	offs = Array(Int,nlayers)
	sumd = 0	# current data cursor
	for i = 1 : nlayers
		sumd += prod(dims[i])+dims[i][1]
		offs[i] = sumd
	end
	offs
end

function TDMLP(genf::Function, layer_sizes::Vector{Int}, layer_delays::Vector{Int}, act::Vector{Function})
	# some initializations
	nlayers = length(layer_sizes) - 1
	dims = [(layer_sizes[i+1],layer_sizes[i],layer_delays[i]) for i in 1:nlayers]

    # generate vector of activation derivatives
    actd = Function[]
    for f in act # if native deriv not found then calculate one with ForwardDiff
        d = haskey(derivs,f) ? derivs[f] : autodiff(f)
        push!(actd,d)
    end

	# offsets into the parameter vector
	offs = calc_offsets(TDNNLayer, dims)

	# our single data vector
	buf = genf(offs[end])

	net = [TDNNLayer(Array(eltype(buf),0,0,0),Array(eltype(buf),0),act[i],actd[i]) for i=1:nlayers]

	tdmlp = TDMLP(net, dims, buf, offs, false)
	unflatten_net!(tdmlp, buf)

	tdmlp
end

# Given a flattened vector (buf), update the neural
# net so that each weight and bias vector points into the
# offsets provided by offs
function unflatten_net!(tdmlp::TDMLP, buf::AbstractVector)
	tdmlp.buf = buf

	for i = 1 : length(tdmlp.net)
		toff = i > 1 ? tdmlp.offs[i-1] : 0
		tdims = tdmlp.dims[i]
		lenw = prod(tdims)
		tdmlp.net[i].w = reshape_view(view(buf, toff+1:toff+lenw), tdims)
		toff += lenw
		tdmlp.net[i].b = view(buf, toff+1:toff+tdims[1])
	end
end
