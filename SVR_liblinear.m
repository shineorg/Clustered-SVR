function [Rpred, Rtest] = SVR_liblinear(Dtrain, Rtrain, Dtest, Rtest, ...
    C, espilon, solverOpt, noCluster)

if isempty(strfind(solverOpt, 'cluster'))
    % liblinear: L1-regularized logistic regression
    eval(['svr_struct = train(Rtrain, sparse(Dtrain), ''-q -s 6 -c ', num2str(C), ...
        ' -p ',num2str(espilon), ''');']);
    Rpred = predict(Rtest, sparse(Dtest), svr_struct, '-q');
elseif ~isempty(strfind(solverOpt, 'cluster'))
    RpredAll = [];
    RtestAll = [];
    [idx_train, center, ~, ~, ~] = litekmeans(Dtrain, noCluster);
    cDist_test = dist(Dtest, center');
    [~, idx_test] = min(cDist_test, [], 2);
    
    for iCluster = 1:noCluster
        idx = find(idx_train == iCluster);
        DtrainCluster = Dtrain(idx, :);
        RtrainCluster = Rtrain(idx);
        idx = find(idx_test == iCluster);
        DtestCluster = Dtest(idx, :);
        RtestCluster = Rtest(idx);
        
        eval(['svr_struct = train(RtrainCluster, sparse(DtrainCluster), ''-q -s 6 -c ', ...
            num2str(C), ' -p ',num2str(espilon), ''');']);
        Rpred = predict(RtestCluster, sparse(DtestCluster), svr_struct, '-q');
        
        RpredAll = [RpredAll; Rpred];
        RtestAll = [RtestAll; RtestCluster];
    end
    Rpred = RpredAll;
    Rtest = RtestAll;
end