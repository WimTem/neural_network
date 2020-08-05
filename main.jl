using LinearAlgebra, Plots

function sigmoid(x::Real, derive::Bool=false)
    (derive==true) ? x*(one(x)-x) : one(x)/(one(x) + exp(-x))
end

function ReLU(x::Real)
    return x > 0.0 ? x : 0
end

function ELU(x::Real, α=1)
    return x > 0.0 ? x : α*exp(x)-one(x) 
end
#Test DATA
X = [0 0 1; 0 1 1; 1 0 1; 1 1 1]
Y = [0 1 1 0]'
# 1 Hidden Layer with variable number of neurons
function neural_network(X, Y, η, n_iter)
    w1 = rand(size(X)[2], 2) #3x5
    w2 = rand(2, 1)

    errors = []

    for i = 1:n_iter
        for j = 1:length(Y)

            #Forward
            z1 = sigmoid.(X[j,:]'*w1) #1x5
            z2 = sigmoid.(z1*w2)      #1x1
            
            #Error 
            e = Y[j] - z2[1]
            push!(errors, e)

            #Backward
            tmp = 2*e*sigmoid.(z2, true)
            dw2 = z1'*tmp #5x1

            tmp2 = (tmp*w2'.*sigmoid.(z1,true))
            dw1 = X[j,:]*tmp2 #3x5

            #Update weights

            w1 = w1 +  η*dw1
            w2 = w2 +  η*dw2
        end
    end
    return plot(1:length(errors), errors)
end

neural_network(X, Y, 0.03, 150)

