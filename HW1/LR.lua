require("hdf5")

function logsumexp(z)
    --Log Sum Exp Trick 
        --https://hips.seas.harvard.edu/blog/2013/01/09/computing-log-sum-exp/
        --Let a = max_n (XW^T+b)_n
        --so that
        --a + \log \sum \exp (XW^T+b - a) 
    --find the maximum values in each column
    local a = z:max(2)
    --subtract constant from XW^T+b
    z:csub(torch.expand(a,a:size(1), z:size(2)))   
    z:exp()
    z = z:sum(2)
    z:log()
    --add constant back in
    z:add(a)
    return z
end

neval=0

function loss_minimization(W, X, Y)
    W = W:reshape(Y:size(2), X:size(2)+1)
    --intercept
    local b = W:sub(1, W:size(1), W:size(2),W:size(2)):t()
    --coefficients
    W = W:sub(1, W:size(1),1,W:size(2)-1)
    --z_c = XW^T + b
    local z = (X*W:t()):add(b:expand(b,X:size(1),b:size(2)))
    --\log \sum \exp z_c
    z_c = logsumexp(z:clone())
    --z - \log \sum \exp z_c 
    z:csub(torch.expand(z_c, z_c:size(1), z:size(2)))
    --L2 regularization
    local norm = W:reshape(W:size(1)*W:size(2), 1)
    --L1 regularization
    --torch.sum(W) --put that above return 
    --Cross Entropy Loss
    local loss = (torch.sum(torch.cmul(Y,z))*-1) + 0.5 * torch.sum(W)--torch.dot(norm, norm)
    return loss, z:exp(), W
end

function minibatch(X, Y, bsize)
    --random ordering of ints [1,nexamples] and take first bsize
    local idx = torch.randperm(X:size(1)):sub(1,bsize)
    --training minibatches
    local X_batch = torch.Tensor(bsize, X:size(2))
    local Y_batch = torch.Tensor(bsize, Y:size(2))
    for i=1,bsize do
        X_batch[i] = X[idx[i]]
        Y_batch[i] = Y[idx[i]]
    end
    return X_batch, Y_batch
end

function grad_loss_minimization(W, X, Y, bsize)
    --do minibatch sampling
    local X_batch, Y_batch = minibatch(X, Y, bsize)
    local loss, mu, W = loss_minimization(W, X_batch, Y_batch)
    
    --calculate the gradient
    --g(W) = \sum (\mu_i - y_i) \times x_i
    --from Murphy pg. 253
    local mu_y = torch.csub(mu, Y_batch)
    local grad = mu_y:t()*X_batch
    grad:add(W)
    grad = grad:cat(torch.zeros(grad:size(1),1), 2)
    grad:sub(1, grad:size(1), grad:size(2), grad:size(2)):add(mu_y:sum(1))
    neval = neval + 1
    print(neval, loss)
    return grad:reshape(grad:size(1)*grad:size(2), 1)
end

function fit(X, Y, bsize, rate, iter)
    --Weight matrix must be passed in as vector
    local W = torch.zeros(Y:size(2) * (X:size(2)+1), 1)

    
    --params
    local lr = rate
    local b1 = 0.9
    local b2 = 0.999
    local e = 1e-8
    local t = 0
    local m
    local v
    local denom

    function adam(W)
        --quicker and smoother than sgd
        --https://github.com/torch/optim/blob/master/adam.lua
        --http://arxiv.org/pdf/1412.6980.pdf
        local grad = grad_loss_minimization(W, X, Y, bsize)
        m = m or W.new(grad:size()):zero()
        v = v or W.new(grad:size()):zero()
        denom = denom or W.new(grad:size()):zero()
        t = t + 1
        m:mul(b1):add(1-b1, grad)
        v:mul(b2):addcmul(1-b2, grad, grad)
        denom:copy(v):sqrt():add(e)
        local biasCorrection1 = 1 - b1^t
        local biasCorrection2 = 1 - b2^t
        local stepSize = lr * math.sqrt(biasCorrection2)/biasCorrection1
        W:addcdiv(-stepSize, m, denom)
        return W
    end
    
    --[[
    function sgd(W)
        local grad = grad_loss_minimization(W, X, Y, bsize)
        grad:mul(lr)
        W:csub(grad)
        return W
    end
    ]]
    
    for i=1,iter do
        --W = sgd(W)
        W = adam(W)
    end

    W = W:reshape(Y:size(2), X:size(2)+1)
    --intercept
    b = W:sub(1, W:size(1), W:size(2), W:size(2))
    --coefficients
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

--feature weight: counts
function createDocWordMatrix(vocab, max_sent_len, sparseMatrix)
    docword = torch.zeros(sparseMatrix:size(1), vocab)
    for i=1,sparseMatrix:size(1) do
        for j=1, max_sent_len do
            local idx = (sparseMatrix[i][j])
            if idx ~= 0 then
                docword[i][idx] = 1 --+ docword[i][idx]
            end
        end
    end
    return docword
end
 
function onehotencode(classes, target)
    onehot = torch.zeros(target:size(1), classes)
    for i=1,target:size(1) do
        onehot[i][target[i]] = 1
    end
    return onehot
end

function write2file(fname, pred) 
    f = io.open(fname, "w")
    f:write("ID,Category\n")
    for i=1,pred:size(1) do
        f:write(tostring(i) .. "," .. tostring(pred[i][1]) .. "\n")
    end
    f:close()
end

f = hdf5.open("SST2.hdf5", "r")

X_train = f:read("train_input"):all()
Y_train = f:read("train_output"):all()
X_valid = f:read("valid_input"):all()
Y_valid = f:read("valid_output"):all()
--X_test = f:read("test_input"):all()
nclasses = f:read('nclasses'):all():long()[1]
nfeatures = f:read('nfeatures'):all():long()[1]

f:close()

X_train =createDocWordMatrix(nfeatures, 53, X_train)
Y_train = onehotencode(nclasses, Y_train)
X_test = createDocWordMatrix(nfeatures, 53, X_valid)
Y_test = onehotencode(nclasses, Y_valid)

start_time = os.time()
W, b = fit(X_train, Y_train, 10000, 0.01, 100)
end_time = os.time()
print(end_time - start_time)

Y_pred = predict(X_train, W, b)
_, Y_pred = torch.max(Y_pred, 2)
_,Y_true = torch.max(Y_train, 2)
acc_score = predict_score(Y_pred, Y_true)
print(acc_score)