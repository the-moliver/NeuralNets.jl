using StatsBase


## create indicies for minibatches
function sample_epoch(datasize, batch_size)
    fitpoints = [1:datasize]
    numtoadd = batch_size - length(fitpoints)%batch_size
    append!(fitpoints, sample(fitpoints,numtoadd))
    numfitpoints = length(fitpoints)
    fitpoints = reshape(fitpoints[randperm(numfitpoints)], batch_size, int(numfitpoints/batch_size))
end

## Initialize mini-batches
function mini_batch_init(x,t,w, fitpoints, mlp::MLP)
  x_batch = x[:,fitpoints]
  t_batch = t[:,fitpoints]
  w_batch = w[:,fitpoints]

  x_batch,t_batch,w_batch
end

function mini_batch_init(x,t,w, fitpoints, tdmlp::TDMLP)
  delays = tdmlp.delays
  x_batch = zeros(eltype(x),size(x,1), length(fitpoints), delays+1)
  t_batch = t[:,fitpoints]
  w_batch = w[:,fitpoints]

  x_batch,t_batch,w_batch
end

## In-place update minibatches
function mini_batch!(x,t,w,x_batch,t_batch,w_batch,fitpoints, mlp::MLP)
  x_batch[:,:] = x[:,fitpoints]
  t_batch[:,:] = t[:,fitpoints]
  w_batch[:,:] = w[:,fitpoints]

  x_batch,t_batch,w_batch
end

function mini_batch!(x,t,w,x_batch,t_batch,w_batch,fitpoints, tdmlp::TDMLP)
  delays = tdmlp.delays
  t_batch[:,:] = t[:,fitpoints]
  t_batch[:,fitpoints.<=tdmlp.delays] = NaN
  w_batch[:,:] = w[:,fitpoints]
  for i=0:delays
    for ii = [1:length(fitpoints)]
      x_batch[:,ii,i+1] = view(x,:,fitpoints[ii])
    end
    fitpoints .-= 1
    fitpoints = max(fitpoints,1)
  end

  x_batch,t_batch,w_batch
end


## Create delta type to hold back-propagated errors in memory
type deltaLayer{T}
    d::AbstractArray{T}
end

type Deltas
    deltas::Vector{deltaLayer}
end


## Intialize deltas
function deltas_init(tdmlp::TDMLP, batch_size)
  layer_delays = Int[]
  nlayers = size(tdmlp.net,1)
  for l in tdmlp.net
    push!(layer_delays, size(l.w,3))
  end
  deltadelays = flipud(cumsum(flipud(layer_delays))-[0:length(layer_delays)-1])

  datatype = eltype(tdmlp.net[1].w)

  ds = [deltaLayer(Array(datatype,0,0,0)) for i=1:nlayers]

  D = Deltas(ds)
  for i=1:nlayers
    D.deltas[i].d = zeros(datatype, size(tdmlp.net[i].w,2),batch_size,deltadelays[i])
  end
  D
end

function deltas_init(mlp::MLP, batch_size)
  nlayers = size(mlp.net,1)

  datatype = eltype(mlp.net[1].w)

  ds = [deltaLayer(Array(datatype,0,0)) for i=1:nlayers]

  D = Deltas(ds)
  for i=1:nlayers
    D.deltas[i].d = zeros(datatype, size(mlp.net[i].w,2),batch_size)
  end
  D
end


## Apply max-norm regularization in-place
function maxnormreg!(net, maxnorm)
 for l = 1:length(net)-1
  s1 = size(net[l].w)
   for hu = 1:s1[1]
     norms = sqrt(sum(view(net[l].w,hu,:) .^2.0))
     if norms>maxnorm
        for ii=1:prod(s1[2:end])
         net[l].w[hu,ii] .*= maxnorm/norms
       end
     end
   end
 end
end



# Train a MLP using gradient decent with Nesterov momentum.
# mlp.net:        array of neural network layers
# x:              input data
# t:              target data
# batch_size:     samples in mini-batch
# maxiter:        number of epochs to train for
# tol:            convergence criterion
# learning_rate:  step size for weight update
# momentum_rate:  amount of momentum
# loss:           loss function to minimize
# maxnorm:        maximum norm of weights to each hidden unit
# eval:           how often we evaluate the loss function
# verbose:        train with printed feedback about the error function
function gdmtrain(mlp::MLNN,
                  x,
                  t;
                  batch_size=size(x,2),
                  maxiter::Int=1000,
                  tol::Real=1e-5,
                  learning_rate::FloatingPoint=.3,
                  learning_rate_factor::FloatingPoint=.999,
                  momentum_rate=.6,
                  eval::Int=10,
                  loss=squared_loss,
                  weights=Array[],
                  maxnorm::FloatingPoint=0.0,
                  minibatchfn=mini_batch!,
                  verbose::Bool=true,
                  verboseiter::Int=100)

  n = size(x,2)
  η, c, m, b = learning_rate, tol, momentum_rate, batch_size
  i = e_old = Δw_old = epoch = 0.
  e_new = loss(prop(mlp,x),t)
  converged::Bool = false
  gain = mlp.gain

  if isempty(weights)
    weights=ones(eltype(t),size(t))  # set weights to 1 if no weights are declared
  end

  # if check if loss function is paired with cannonical output activation function
  if haskey(cannonical,mlp.net[end].a) && cannonical[mlp.net[end].a] == loss
      lossd = [] # if cannonical, derivative function is not needed
  else
      lossd = haskey(lossderivs,loss) ? lossderivs[loss] : autodiff(loss)  # if non-cannonical, use derivative and autodiff if not specified
  end

  # Initialize mini-batch and delta structure
  x_batch,t_batch,w_batch = mini_batch_init(x,t,weights, [1:batch_size], mlp)              
  D = deltas_init(mlp, batch_size)

  while epoch < maxiter
    epoch += 1

    fitpoints = sample_epoch(n, batch_size)                                  # Create mini-batch sample points for epoch

    if epoch > 1
      η .*= learning_rate_factor
    end

    i = 0

    while (!converged && i < size(fitpoints,2))
      i += 1

      x_batch,t_batch,w_batch = minibatchfn(x,t,weights,x_batch,t_batch,w_batch,fitpoints[:,i], mlp)   # Create mini-batch

      mlp.net = mlp.net .+ m*Δw_old      # Nesterov Momentum, update with momentum before computing gradient
      ∇,δ = backprop(mlp.net,x_batch,t_batch,lossd,D.deltas,w_batch,gain)
      Δw_new = -η*∇                     # calculate Δ weights
      mlp.net = mlp.net .+ Δw_new       # update weights

      if maxnorm > 0.0
        maxnormreg!(mlp.net, maxnorm)
      end

      Δw_old = Δw_new .+ m*Δw_old       # keep track of all weight updates

      if i % eval == 0  # recalculate loss every eval number iterations
          e_old = e_new
          e_new = loss(prop(mlp,x),t)
          converged = abs(e_new - e_old) < c # check if converged
      end
      if verbose && i % verboseiter == 0
          println("i: $i\tLoss=$(round(e_new,6))\tΔLoss=$(round((e_new - e_old),6))\tAvg. Loss=$(round((e_new/n),6))")
      end
    end
      convgstr = converged ? "converged" : "didn't converge"
      println("Training $convgstr in less than $i iterations; average error: $(round((e_new/n),4)).")
      println("* learning rate η = $η")
      println("* momentum coefficient m = $m")
      println("* convergence criterion c = $c")

  end

  mlp.trained=true
  return mlp
end

# Train a MLP using Adagrad
# mlp.net:        array of neural network layers
# x:              input data
# t:              target data
# batch_size:     samples in mini-batch
# maxiter:        number of epochs to train for
# lambda:         parameter to prevent divide by zero
# tol:            convergence criterion
# learning_rate:  step size for weight update
# loss:           loss function to minimize
# maxnorm:        maximum norm of weights to each hidden unit
# eval:           how often we evaluate the loss function
# verbose:        train with printed feedback about the error function
function adatrain(mlp::MLNN,
                  x,
                  t;
                  batch_size=size(x,2),
                  maxiter::Int=100,
                  tol::Real=1e-5,
                  learning_rate::FloatingPoint=.3,
                  learning_rate_factor::FloatingPoint=.999,
                  lambda=1e-6,
                  loss=squared_loss,
                  weights=Array[],
                  maxnorm::FloatingPoint=0.0,
                  minibatchfn=mini_batch!,
                  eval::Int=10,
                  verbose::Bool=true)

  η, c, λ, b = learning_rate, tol, lambda, batch_size
  i = e_old = Δnet = sumgrad = epoch = 0.0
  e_new = loss(prop(mlp,x),t)
  n = size(x,2)
  converged::Bool = false
  gain = mlp.gain

  if isempty(weights)
    weights=ones(eltype(t),size(t))  # set weights to 1 if no weights are declared
  end

  # if check if loss function is paired with cannonical output activation function
  if haskey(cannonical,mlp.net[end].a) && cannonical[mlp.net[end].a] == loss
      lossd = [] # if cannonical, derivative function is not needed
  else
      lossd = haskey(lossderivs,loss) ? lossderivs[loss] : autodiff(loss)  # if non-cannonical, use derivative and autodiff if not specified
  end

  # Initialize mini-batch and delta structure
  x_batch,t_batch,w_batch = mini_batch_init(x,t,weights, [1:batch_size], mlp)              
  D = deltas_init(mlp, batch_size)

  while epoch < maxiter
    epoch += 1

    fitpoints = sample_epoch(n, batch_size)                                  # Create mini-batch sample points for epoch

    if epoch > 1
      η .*= learning_rate_factor
    end

    i = 0

    while (!converged && i < size(fitpoints,2))
      i += 1

      x_batch,t_batch,w_batch = minibatchfn(x,t,weights,x_batch,t_batch,w_batch,fitpoints[:,i], mlp)   # Create mini-batch

      ∇,δ = backprop(mlp.net,x_batch,t_batch,lossd,D.deltas,w_batch,gain)
      sumgrad += ∇ .^ 2.       # store sum of squared past gradients
      Δw = η * ∇ ./ (λ .+ (sumgrad .^ 0.5))   # calculate Δ weights
      mlp.net = mlp.net .- Δw                 # update weights

      if maxnorm > 0.0
        maxnormreg!(mlp.net, maxnorm)
      end

      if i % eval == 0  # recalculate loss every eval number iterations
          e_old = e_new
          e_new = loss(prop(mlp,x),t)
          converged = abs(e_new - e_old) < c # check if converged
      end
      if verbose && i % 100 == 0
          println("i: $i\tLoss=$(round(e_new,6))\tΔLoss=$(round((e_new - e_old),6))\tAvg. Loss=$(round((e_new/n),6))")
      end
    end
  end
  convgstr = converged ? "converged" : "didn't converge"
  println("Training $convgstr in less than $i iterations; average error: $(round((e_new/n),4)).")
  println("* learning rate η = $η")
  println("* convergence criterion c = $c")
  mlp.trained=true
  return mlp
end

# Train a MLP using RMSProp with Nesterov momentum and adaptive step sizes.
# mlp.net:        array of neural network layers
# x:              input data
# t:              target data
# batch_size:     samples in mini-batch
# maxiter:        number of epochs to train for
# learning_rate:  step size for weight update
# momentum_rate:  amount of momentum
# loss:           loss function to minimize
# maxnorm:        maximum norm of weights to each hidden unit
# eval:           how often we evaluate the loss function
# verbose:        train with printed feedback about the error function
function rmsproptrain(mlp::MLNN,
                  x,
                  t;
                  batch_size=size(x,2),
                  maxiter::Int=100,
                  learning_rate::FloatingPoint=.3,
                  learning_rate_factor::FloatingPoint=.999,
                  momentum_rate::FloatingPoint=.5,
                  stepadapt_rate::FloatingPoint=.01,
                  minadapt::FloatingPoint=.5,
                  maxadapt::FloatingPoint=5.0,
                  sqgradupdate_rate::FloatingPoint=.1,
                  maxnorm::FloatingPoint=0.0,
                  loss=squared_loss,
                  weights=Array[],
                  minibatchfn=mini_batch!,
                  verbose::Bool=true,
                  verboseiter::Int=100,
                  xval=Array[],
                  tval=Array[])
  n = size(x,2)
  η, m, b = learning_rate, momentum_rate, batch_size
  e_old = Δw_old = epoch = 0.
  f0 = convert(eltype(x), 0.0)
  f2 = convert(eltype(x), 2.0)
  f05 = convert(eltype(x), 0.5)
  minadapt = convert(eltype(x), minadapt)
  maxadapt = convert(eltype(x), maxadapt)
  stepadapt = ∇2 = mlp.net.^f0
  gain = mlp.gain

   if isempty(weights)
    weights=ones(eltype(t),size(t))
  end

  if isempty(xval)
    e_new = loss(prop(mlp,x),t)
  else
    e_new = loss(prop(mlp,xval),tval)
  end

  # if check if loss function is paired with cannonical output activation function
  if haskey(cannonical,mlp.net[end].a) && cannonical[mlp.net[end].a] == loss
      lossd = [] # if cannonical, derivative function is not needed
  else
      lossd = haskey(lossderivs,loss) ? lossderivs[loss] : autodiff(loss)  # if non-cannonical, use derivative and autodiff if not specified
  end

  x_batch,t_batch,w_batch = mini_batch_init(x,t,weights, [1:batch_size], mlp)  # Initialize mini-batch
  D = deltas_init(mlp, batch_size)

  while epoch < maxiter
    epoch += 1

    fitpoints = sample_epoch(n, batch_size)   # Create mini-batch sample points for epoch

    if epoch > 1
      η .*= learning_rate_factor
    end

    i = 0

    while i < size(fitpoints,2)
        i += 1

        x_batch,t_batch,w_batch = minibatchfn(x,t,weights,x_batch,t_batch,w_batch,fitpoints[:,i], mlp)   # Create mini-batch

        mlp.net = mlp.net .+ m*Δw_old                                        # Nesterov Momentum, update with momentum before computing gradient
        ∇,δ = backprop(mlp.net,x_batch,t_batch,lossd,D.deltas,w_batch,gain)

        if i > 1 || epoch > 1
          stepadapt .*= (1.0 .-(stepadapt_rate.*(sign(∇) .* sign(Δw_old))))  # step size adaptation
          stepadapt = max(min(stepadapt, maxadapt), minadapt)                # keep step size adaptation within range

        end

        ∇2 = sqgradupdate_rate.*∇.^f2 + (1.0 .-sqgradupdate_rate).*∇2        # running estimate of squared gradient
        Δw_new = stepadapt .* (-η .* (1-m) .* ∇ ./  (∇2.^f05))                        # calculate Δ weights

        mlp.net = mlp.net .+ Δw_new                                          # update weights

        if maxnorm > 0.0
          maxnormreg!(mlp.net, maxnorm)
        end

        Δw_old = Δw_new .+ m.*Δw_old                                         # keep track of all weight updates

    end

    if verbose
      e_old = e_new
      if isempty(xval)
        e_new = loss(prop(mlp,x),t)
      else
        e_new = loss(prop(mlp,xval),tval)
      end
      println("epoch: $epoch\tLoss=$(round(e_new,6))\tΔLoss=$(round((e_new - e_old),6))\tAvg. Loss=$(round((e_new/n),6))")
    end

  end

  println("Training finished in $epoch epochs; average error: $(round((e_new/n),4)).")
  println("* learning rate η = $η")
  println("* momentum coefficient m = $m")
  mlp.trained=true
  return mlp
end
