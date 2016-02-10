require('hdf5')
require 'math'

function isnan(m)
    if m ~= m then
        return true
    else
        return false
    end
end

function linear(b, W, x)
    --weight matrix W (dout x din), bias b (dout x 1), input matrix x (n x din)
    local B = b:clone()
    local B = B:expand(B,B:size(1),x:size(1))
    return ((W*x:t()):add(B)):t()
end
    
function second_max(Y_pred, Y_true)
    --Y_pred is a nx5 tensor
    local second_maxes = torch.zeros(Y_pred:size(1),1)
    local indices = second_maxes:clone()
    local Y_prime = (Y_pred):clone()
    for i = 1, Y_pred:size(1) do
        Y_prime[i][Y_true[i]] = -100
        z1,z2 = torch.max(Y_prime[i],1)
        second_maxes[i] = z1[1]
        indices[i] = z2[1]
    end
    return second_maxes, indices
end

function correct_val(Y_pred, Y_true)
    return_val = torch.zeros(Y_true:size(1),1)
    for i = 1, Y_true:size(1) do
        return_val[i] = Y_pred[i][Y_true[i]]
    end
    return return_val
end

function ReLU(input)
    for i = 1,input:size(1) do
        if input[i][1]<0 then
            input[i][1] =0
        end
    end
    return input
end

function hinge_loss(Y_pred, Y_true, W,lambda,b, mode)
    --Y_pred is a nx5 tensor, Y_true is an nx1 tensor
    --mode = 'vanilla' for vanilla hinge
    --mode = 'L2' for hinge^2
    if mode == 'vanilla' then
        reg = 1
    elseif mode == 'squared' then
        reg = 2
    end
    
    params = torch.cat(W,b,2)
    local ones_vector = torch.ones(Y_true:size(1),1)
    local loss_vector = torch.pow(ReLU(ones_vector+second_max(Y_pred, Y_true)-correct_val(Y_pred,Y_true)),reg)
    loss1 = lambda*torch.pow((torch.norm(params)),2)
    loss2 = loss_vector:sum(1)[1][1]
    
    return (loss1 + loss2)
end

function hinge_grad(W,x,b,Y_true,lambda, mode)
    local Y_pred = linear(b,W,x)
    local loss = hinge_loss(Y_pred, Y_true, W, lambda, b, mode)
    local condition = torch.Tensor(Y_true:size(1),1)
    local correct = correct_val(Y_pred,Y_true) 
    local second, s_indices = second_max(Y_pred,Y_true)
    local condition = correct - second
    local W_grad = torch.zeros(W:size())
    local W_temp = torch.zeros(W:size())
    local b_grad = torch.zeros(b:size())
    local b_temp = torch.zeros(b:size())
    local debug = torch.zeros(4)
    for i = 1, condition:size(1) do
        if condition[i][1] < 1 then
            for j = 1, W:size(1) do
                if j == Y_true[i] then
                    b_temp[j] = -1
                    W_temp[j] = x[i]*(-1)*loss
                    debug[1] = debug[1] + 1
                elseif j == s_indices[i][1] then
                    debug[2] = debug[2] + 1
                    b_temp[j] = 1
                    W_temp[j] = x[i]*loss
                else
                    debug[3] = debug[3] + 1
                    b_temp[j] = 0
                    W_temp[j] = W_temp[j]*0
                end
            end
            b_grad:add(b_temp)
            W_grad:add(W_temp)
        else
            debug[4] = debug[4] + 1
            b_temp = torch.zeros(b_temp:size())
            W_temp = torch.zeros(W_temp:size())
        end
    end
    return W_grad:add(2*lambda,W),b_grad:add(2*lambda,b)
    --return W_grad,0--b_grad
end



function obj(params,X,Y,batch,lambda, mode)
    if batch > 0 then
        bsize= batch
        local idx = torch.randperm(X:size(1)):sub(1,bsize)

        x = torch.Tensor(bsize, X:size(2))

        Y = Y:reshape(Y:size(1), 1)

        y = torch.Tensor(bsize, 1)

        for i=1,bsize do
            x[i] = X[idx[i]]
            y[i] = Y[idx[i]]
        end
        
        y = y:squeeze()
    else
        y = Y
        x = X
    end
    
    local w = (params:sub(1,params:size(1),1,params:size(2)-1))
    
   
    b = params:sub(1,params:size(1),params:size(2),params:size(2))
    local ypred = linear(b,w,x)
    local loss = hinge_loss(ypred,y,w,lambda,b, mode)
    
    local w1,b1 = hinge_grad(w,x,b,y,lambda, mode)
    local grads = torch.zeros(params:size())
    grads[{{},{1,params:size(2)-1}}] = w1
    grads[{{},{params:size(2),params:size(2)}}] = b1
    return loss, grads
end

function obj_final(params)
    
    loss,grads = obj(params, X_train, Y_train, 50, 1, 'vanilla')
    print(loss)
    return loss,grads
end

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
    local _,grad = obj_final(W)
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

function sgd(W)
    local _, grad = obj_final(W)
    grad:mul(lr)
    W = W:csub(grad)
    return W
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

--feature weight: one hot
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

X_train=createDocWordMatrix(nfeatures, 53, X_train)
X_valid=createDocWordMatrix(nfeatures, 53, X_valid)

W0 = torch.randn(5,nfeatures)*.01
b0 =torch.zeros(5,1)*.001
p0 = torch.cat(W0,b0,2)
rate = 0.0001

--d = torch.Tensor(50)
for i = 1,10 do
    --p0 = adam(p0)
    p0 = sgd(p0)
end


function predict(X, W, b)
    local b = b:t()
    return (X*W:t()):add(b:expand(b, X:size(1), b:size(2)))
end
function predict_score(ypred, ytrue)
    local c = 0
    for i=1,ypred:size(1) do
        if ypred[i][1] == ytrue[i] then
            c = c + 1       
        end
    end
    return c/ypred:size(1)
end



 --intercept
b = p0:sub(1, p0:size(1), p0:size(2), p0:size(2))

--coefficients
W = p0:sub(1, p0:size(1), 1, p0:size(2)-1)


Y_pred = predict(X_valid, W, b)
_, score = torch.max(Y_pred, 2)
--_,Y_true = torch.max(Y_valid, 2)
acc_score = predict_score(score, Y_valid)
print(acc_score)
