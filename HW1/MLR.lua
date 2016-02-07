
require 'torch'

function ml(W, X, Y)

    local W = W:reshape(Y:size(2), X:size(2)+1)
    local b = W:sub(1, W:size(1), W:size(2),W:size(2)):t()
    W = W:sub(1, W:size(1),1,W:size(2)-1)
    local p = X*W:t()
    
    b:expand(b,p:size(1),b:size(2))
    p:add(b)
    local arr = p:clone()
    arr = arr:t()
    local vmax = arr:max(1)

    local evmax = torch.expand(vmax,arr:size(1),vmax:size(2))
    
    arr:csub(evmax)

    arr:exp()

    arr = arr:sum(1)
    
    arr:log()
    
    arr:add(vmax)

    arr = arr:t()
    arr:expand(arr, arr:size(1), p:size(2))

    p:csub(arr)
    
    local loss = (torch.sum(torch.cmul(Y,p))*-1) + 1.0 *0.5 * torch.norm(W)
    
    p:exp()
    
    return loss, p, W
end

function mlg(W, X, Y)
    local grad = torch.zeros(Y:size(2), X:size(1)+1)
    local loss 
    loss, p, W = ml(W, X, Y)
    local diff = torch.csub(p,Y)
    local grad = diff:t()*X
    grad:add(W)
    grad = grad:cat(torch.zeros(grad:size(1),1), 2)
    grad:sub(1, grad:size(1), grad:size(2), grad:size(2)):add(diff:sum(1))
    --print(loss)
    return loss, grad:reshape(grad:size(1)*grad:size(2), 1), p
end



function fit(X, Y, rate)
    local W = torch.zeros(Y:size(2), X:size(2)+1)
    W = W:reshape(W:size(1) * W:size(2), 1)
    
    local func = function(W)
        loss, grad, p = mlg(W, X, Y)
        return loss, grad
    end
    state = {learningRate = rate, maxIter=100, tolX=1e-9}
    W, f_hist, currentFuncEval = optim.lbfgs(func, W, state)
    W = W:reshape(Y:size(2), X:size(2)+1)
    b = W:sub(1, W:size(1), W:size(2), W:size(2))
    W = W:sub(1, W:size(1), 1, W:size(2)-1)
    return W, b
end
        
function predict(X, W, b)
    local b = b:t()
    return (X*W:t()):add(b:expand(b, X:size(1), b:size(2)))
end

function predict_score(ypred, ytrue)
    local c = 0
    for i=1,ypred:size(1) do
        if ypred[i][1] == ytrue[i][1] then
            c = c + 1       
        end
    end
    return c/ypred:size(1)
end

X_train = torch.randn(1000, 5000)
Y_train = torch.Tensor(1000)
Y_train:random(1,5)
Y_train = onehotencode(5, Y_train)

local x = os.clock()
W, b = fit(X_train, Y_train, 0.1)
print(string.format("elapsed time: %.2f\n", os.clock() - x))
