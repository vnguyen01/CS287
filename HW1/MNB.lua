require "hdf5"

--converts our sparse matrix format to doc by feature matrix
--feature weighting: word counts
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

--one hot encoding of classes
function onehotencode(classes, target)
    onehot = torch.zeros(target:size(1), classes)
    for i=1,target:size(1) do
        onehot[i][target[i]] = 1
    end
    return onehot
end

--MNB fit to train
function fit(X,Y)
    --calculate log posterior
    local cc = torch.zeros(1, Y:size(2))
    local fc = torch.zeros(Y:size(2), X:size(2))
    local clp = torch.zeros(1, Y:size(2))
    
    fc:add(Y:t()*(X))
    cc:add(Y:sum(1))
    
    --smoothing
    fc:add(1)
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

--MNB predict on test or valid
function predict(X, W, b)
    --joint log-likelihood
    local jll = ((X*W:t()):csub(b:expand(b,X:size(1),b:size(2))))
    return jll
end

function predict_score(ypred, ytrue)
    local c = 0
    for i=1,pred:size(1) do
        if ypred[i][1] == ytrue[i][1] then
            c = c + 1       
        end
    end
    return c/ypred:size(1)
end

f = hdf5.open("SST1.hdf5", "r")
X_train = f:read("train_input"):all()
Y_train = f:read("train_output"):all()
X_valid = f:read("valid_input"):all()
Y_valid = f:read("valid_output"):all()
X_test = f:read("test_input"):all()
nclasses = f:read('nclasses'):all():long()[1]
nfeatures = f:read('nfeatures'):all():long()[1]
f:close()

--for final testing ONLY!
--X_train = torch.cat(X_train, X_valid, 1)
--Y_train = torch.cat(Y_train, Y_valid, 1)

X_train = createDocWordMatrix(nfeatures, 53, X_train)
Y_train = onehotencode(nclasses, Y_train)
X_test = createDocWordMatrix(nfeatures, 53, X_test)
Y_test = onehotencode(nclasses, Y_test)

lp, clp = counts(X_train, Y_train)
Y_pred = predict(X_test, lp, clp)
_, Y_pred = torch.max(Y_pred, 2)
_,Y_true = torch.max(Y_test, 2)
acc_score = predict_score(Y_pred, Y_true)
print(acc_score)

f = io.open("MNB_3.csv", "w")
f:write("ID,Category\n")
for i=1,Y_pred:size(1) do
    f:write(tostring(i) .. "," .. tostring(Y_pred[i][1]) .. "\n")
end
f:close()





