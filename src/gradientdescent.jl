using StatsBase

# batch
# function to retrieve a random subset of data
# currently quite ugly, if anyone knows how to do this better go ahead
function batch(b::Int,x::Array,t::Array)
    n = size(x,2)
    b == n && return x,t
    b > n && throw("Error: Batch size larger than the number of data points supplied.")
    index = rand(1:n, b)
    return x[:,index],t[:,index]
end

function mini_batch(x,t, fitpoints, mlp::MLP)
  x_batch = x[:,fitpoints]
  t_batch = t[:,fitpoints]

  x_batch,t_batch
end

function mini_batch(x,t, fitpoints, tdmlp::TDMLP)
  delays = tdmlp.delays
  x_batch = zeros(eltype(x),size(x,1), length(fitpoints), delays+1)
  for i=0:delays
    x_batch[:,:,i+1] = x[:,max(fitpoints-i,1)]
  end

  t_batch = t[:,fitpoints]

  x_batch,t_batch
end

#function maxnormreg!(net, maxnorm)
#  for l in net
#    for hu = 1:size(l.w,1)
#      norms = sqrt(sum(view(l.w,hu,:) .^2.0))
#      if norms>maxnorm
#        for ii=1:size(l.w,2)
#          l.w[hu,ii] .*= maxnorm/norms
#        end
#      end
#    end
#  end
#end

function maxnormreg!(net, maxnorm)
    for hu = 1:size(net[1].w,1)
      norms = sqrt(sum(view(net[1].w,hu,:) .^2.0))
      if norms>maxnorm
        for ii=1:size(net[1].w,2)
          net[1].w[hu,ii] .*= maxnorm/norms
        end
      end
    end
end

function sample_epoch(datasize, batch_size)
    fitpoints = [1:datasize]
    numtoadd = batch_size - length(fitpoints)%batch_size
    append!(fitpoints, sample(fitpoints,numtoadd))
    numfitpoints = length(fitpoints)
    fitpoints = reshape(fitpoints[randperm(numfitpoints)], batch_size, int(numfitpoints/batch_size))
end

# Train a MLP using gradient decent with Nesterov momentum.
# mlp.net:  array of neural network layers
# x:        input data
# t:        target data
# η:        learning rate
# m:        momentum coefficient
# c:        convergence criterion
# eval:     how often we evaluate the loss function
# verbose:  train with printed feedback about the error function
function gdmtrain(mlp::MLNN,
                  x,
                  t;
                  batch_size=size(x,2),
                  maxiter::Int=1000,
                  tol::Real=1e-5,
                  learning_rate=.3,
                  momentum_rate=.6,             
                  eval::Int=10,
                  loss=squared_loss,
                  verbose::Bool=true,
                  verboseiter::Int=100)
    n = size(x,2)
    η, c, m, b = learning_rate, tol, momentum_rate, batch_size
    i = e_old = Δw_old = 0
    e_new = loss(prop(mlp,x),t)
    converged::Bool = false

    if haskey(cannonical,mlp.net[end].a) && cannonical[mlp.net[end].a] == loss
        lossd = []
    else
        lossd = haskey(lossderivs,loss) ? lossderivs[loss] : autodiff(loss)
    end

    while (!converged && i < maxiter)
        i += 1
        x_batch,t_batch = batch(b,x,t)
        mlp.net = mlp.net .+ m*Δw_old      # Nesterov Momentum, update with momentum before computing gradient
        ∇,δ = backprop(mlp.net,x_batch,t_batch,lossd)
        Δw_new = -η*∇                     # calculate Δ weights   
        mlp.net = mlp.net .+ Δw_new       # update weights                       
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
    mlp.trained=true
    return mlp
end

# Train a MLP using Adagrad
# mlp.net:  array of neural network layers
# x:        input data
# t:        target data
# η:        learning rate
# c:        convergence criterion
# ε:        small constant for numerical stability
# eval:     how often we evaluate the loss function
# verbose:  train with printed feedback about the error function
function adatrain(mlp::MLNN,
                  x,
                  t;
                  batch_size=size(x,2),                  
                  maxiter::Int=1000,
                  tol::Real=1e-5,
                  learning_rate=.3,
                  lambda=1e-6,
                  loss=squared_loss,
                  eval::Int=10,
                  verbose::Bool=true)

    η, c, λ, b = learning_rate, tol, lambda, batch_size
    i = e_old = Δnet = sumgrad = 0.0
    e_new = loss(prop(mlp,x),t)
    n = size(x,2)
    converged::Bool = false

    if haskey(cannonical,mlp.net[end].a) && cannonical[mlp.net[end].a] == loss
        lossd = []
    else
        lossd = haskey(lossderivs,loss) ? lossderivs[loss] : autodiff(loss)
    end

    while (!converged && i < maxiter)
        i += 1
        x_batch,t_batch = batch(b,x,t)
        ∇,δ = backprop(mlp.net,x_batch,t_batch,lossd)
        sumgrad += ∇ .^ 2.       # store sum of squared past gradients
        Δw = η * ∇ ./ (λ .+ (sumgrad .^ 0.5))   # calculate Δ weights
        mlp.net = mlp.net .- Δw                 # update weights

        if i % eval == 0  # recalculate loss every eval number iterations
            e_old = e_new
            e_new = loss(prop(mlp,x),t)
            converged = abs(e_new - e_old) < c # check if converged
        end
        if verbose && i % 100 == 0
            println("i: $i\tLoss=$(round(e_new,6))\tΔLoss=$(round((e_new - e_old),6))\tAvg. Loss=$(round((e_new/n),6))")
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
# mlp.net:  array of neural network layers
# x:        input data
# t:        target data
# η:        learning rate
# m:        momentum coefficient
# c:        convergence criterion
# eval:     how often we evaluate the loss function
# verbose:  train with printed feedback about the error function
function rmsproptrain(mlp::MLNN,
                  x,
                  t;
                  batch_size=size(x,2),
                  maxiter::Int=100,
                  learning_rate::FloatingPoint=.3,
                  learning_rate_factor::FloatingPoint=.999,
                  momentum_rate::FloatingPoint=.6,
                  stepadapt_rate::FloatingPoint=.01,
                  minadapt::FloatingPoint=.5,
                  maxadapt::FloatingPoint=5.0,
                  sqgradupdate_rate::FloatingPoint=.1,
                  maxnorm::FloatingPoint=0.0,
                  loss=squared_loss,            
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
  stepadapt = ∇2 = mlp.net.^f0
  if isempty(xval)
    e_new = loss(prop(mlp,x),t)
  else
    e_new = loss(prop(mlp,xval),tval)
  end

  if haskey(cannonical,mlp.net[end].a) && cannonical[mlp.net[end].a] == loss
      lossd = []
  else
      lossd = haskey(lossderivs,loss) ? lossderivs[loss] : autodiff(loss)
  end
    
  while epoch < maxiter
    epoch += 1

    fitpoints = sample_epoch(n, batch_size)                                  # Create mini-batch sample points for epoch

    if epoch > 1
      η .*= learning_rate_factor
    end

    i = 0

    while i < size(fitpoints,2)
        i += 1

        x_batch,t_batch = mini_batch(x,t, fitpoints[:,i], mlp)               # Create mini-batch
        
        mlp.net = mlp.net .+ m*Δw_old                                        # Nesterov Momentum, update with momentum before computing gradient
        ∇,δ = backprop(mlp.net,x_batch,t_batch,lossd)

        if i > 1 || epoch > 1
          stepadapt .*= (1.0 .-(stepadapt_rate.*(sign(∇) .* sign(Δw_old))))  # step size adaptation
          stepadapt = max(min(stepadapt, maxadapt), minadapt)                # keep step size adaptation within range

        end

        ∇2 = sqgradupdate_rate.*∇.^f2 + (1.0 .-sqgradupdate_rate).*∇2        # running estimate of squared gradient
        Δw_new = stepadapt .* (-η .* ∇ ./  (∇2.^f05))                        # calculate Δ weights

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
  #println("* convergence criterion c = $c")
  mlp.trained=true
  return mlp
end
