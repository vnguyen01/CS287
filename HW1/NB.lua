require('hdf5')

f = hdf5.open("SST2.hdf5", "r")
X_train = f:read("train_input"):all()
Y_train = f:read("train_output"):all()
X_valid = f:read("valid_input"):all()
Y_valid = f:read("valid_output"):all()
X_test = f:read("test_input"):all()

nclasses = f:read('nclasses'):all():long()[1]
nfeatures = f:read('nfeatures'):all():long()[1]

f:close()

function createDocWordMatrix(vocab, max_sent_len, sparseMatrix)
    docword = torch.zeros(sparseMatrix:size(1), vocab)
    for i=1,sparseMatrix:size(1) do
        for j=1, max_sent_len do
            local idx = (sparseMatrix[i][j])
            if idx ~= 0 then
                docword[i][idx] = 1 + docword[i][idx]
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

X_train = createDocWordMatrix(nfeatures, 53, X_train)
Y_train = onehotencode(nclasses, Y_train)
X_test = createDocWordMatrix(nfeatures, 53, X_valid)
Y_test = onehotencode(nclasses, Y_valid)

function counts(X,Y)
    --calculate log posterior
    local cc = torch.zeros(1, Y:size(2))
    local fc = torch.zeros(Y:size(2), X:size(2))
    local clp = torch.zeros(1, Y:size(2))
    
    fc:add(Y:t()*(X))
    cc:add(Y:sum(1))
    
    --THIS IS ALPHA
    --smoothing
    fc:add(5)
    local scc = fc:sum(2)
    
    fc:log()
    scc:log()
    
    scc:expand(scc, scc:size(1), fc:size(2))
    
    --calculate log prior
    --local total = cc:sum(2)
    --cc:div(total[1][1])
    --cc:log()
    clp:csub(math.log(Y:size(2)))
    
    return fc:csub(scc), clp --:csub(math.log(Y:size(2)))
end

function predict(X, W, b)
    --joint log-likelihood
    local jll = ((X*W:t()):csub(b:expand(b,X:size(1),b:size(2))))
    return jll
end

function predict_score()
    local c = 0
    for i=1,indices_pred:size(1) do

        if indices_pred[i][1] == indices_true[i][1] then
            c = c + 1
        
        end
    
    end
    return c/Y_valid:size()[1]
end

lp, clp = counts(X_train, Y_train)

predictions = predict(X_test, lp, clp)
_, indices_pred = torch.max(predictions, 2)
_, indices_true = torch.max(Y_test, 2)
counts = predict_score()
print(counts)