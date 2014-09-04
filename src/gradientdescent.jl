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

# Train a MLP using gradient decent with Nesterov momentum.
# mlp.net:  array of neural network layers
# x:        input data
# t:        target data
# η:        learning rate
# m:        momentum coefficient
# c:        convergence criterion
# eval:     how often we evaluate the loss function
# verbose:  train with printed feedback about the error function
function gdmtrain(mlp::MLP,
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
    e_new = loss(prop(mlp.net,x),t)
    converged::Bool = false

    lossd = haskey(lossderivs,loss) ? lossderivs[loss] : autodiff(lossderivs)

    while (!converged && i < maxiter)
        i += 1
        x_batch,t_batch = batch(b,x,t)
        mlp.net = mlp.net .+ m*Δw_old      # Nesterov Momentum, update with momentum before computing gradient
        ∇,δ = backprop(mlp.net,x_batch,t_batch,lossd=lossd)
        Δw_new = -η*∇                     # calculate Δ weights   
        mlp.net = mlp.net .+ Δw_new       # update weights                       
        Δw_old = Δw_new .+ m*Δw_old       # keep track of all weight updates

        if i % eval == 0  # recalculate loss every eval number iterations
            e_old = e_new
            e_new = loss(prop(mlp.net,x),t)
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
function adatrain(mlp::MLP,
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
    e_new = loss(prop(mlp.net,x),t)
    n = size(x,2)
    converged::Bool = false

    lossd = haskey(lossderivs,loss) ? lossderivs[loss] : autodiff(lossderivs)

    while (!converged && i < maxiter)
        i += 1
        x_batch,t_batch = batch(b,x,t)
        ∇,δ = backprop(mlp.net,x_batch,t_batch,lossd=lossd)
        sumgrad += ∇ .^ 2.       # store sum of squared past gradients
        Δw = η * ∇ ./ (λ .+ (sumgrad .^ 0.5))   # calculate Δ weights
        mlp.net = mlp.net .- Δw                 # update weights

        if i % eval == 0  # recalculate loss every eval number iterations
            e_old = e_new
            e_new = loss(prop(mlp.net,x),t)
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
    return mlp
end

# Train a MLP using RMSProp with momentum and adaptive step sizes.
# mlp.net:  array of neural network layers
# x:        input data
# t:        target data
# η:        learning rate
# m:        momentum coefficient
# c:        convergence criterion
# eval:     how often we evaluate the loss function
# verbose:  train with printed feedback about the error function
function rmsproptrain(mlp::MLP,
                  x,
                  t;
                  batch_size=size(x,2),
                  maxiter::Int=1000,
                  tol::Real=1e-5,
                  learning_rate=.3,
                  learning_rate_factor=.999,
                  momentum_rate=.6,
                  stepadapt_rate=.01,
                  minadapt=.5,
                  maxadapt=5.0,
                  sqgradupdate_rate=.1,
                  loss=squared_loss,            
                  eval::Int=10,
                  verbose::Bool=true,
                  verboseiter::Int=100)
    n = size(x,2)
    η, c, m, b = learning_rate, tol, momentum_rate, batch_size
    i = e_old = Δw_old = 0
    stepadapt = ∇2 = mlp.net.^0.0
    e_new = loss(prop(mlp.net,x),t)
    converged::Bool = false

    lossd = haskey(lossderivs,loss) ? lossderivs[loss] : autodiff(lossderivs)

    while (!converged && i < maxiter)
        i += 1
        x_batch,t_batch = batch(b,x,t)
        mlp.net = mlp.net .+ m*Δw_old      # Nesterov Momentum, update with momentum before computing gradient
        ∇,δ = backprop(mlp.net,x_batch,t_batch,lossd=lossd)
        if i > 1
          stepadapt .*= (1.0 .-(stepadapt_rate.*(sign(∇) .* sign(Δw_old))))  # step size adaptation
          stepadapt = max(min(stepadapt, maxadapt), minadapt)               # keep step size adaptation within range
          η .*= learning_rate_factor
        end

        ∇2 = sqgradupdate_rate.*∇.^2. + (1.0 .-sqgradupdate_rate).*∇2       # running estimate of squared gradient
        Δw_new = stepadapt .* (-η .* ∇ ./  (∇2.^0.5))  # calculate Δ weights   
        mlp.net = mlp.net .+ Δw_new       # update weights                       
        Δw_old = Δw_new .+ m.*Δw_old       # keep track of all weight updates

        if i % eval == 0  # recalculate loss every eval number iterations
            e_old = e_new
            e_new = loss(prop(mlp.net,x),t)
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