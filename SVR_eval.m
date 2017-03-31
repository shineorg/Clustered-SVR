function [Rpred, Rtest] = SVR_eval(feat_nfold, response_nfold, iFold, ...
    noFold, solverOpt, C, espilon, noCluster)

Rpred = [];

[Dtrain, Rtrain, Dtest, Rtest] = SVR_dataPrep(feat_nfold, ...
    response_nfold, iFold, noFold);

if ~isempty(strfind(solverOpt, 'liblinear'))
    [Rpred, Rtest] = SVR_liblinear(Dtrain, Rtrain, Dtest, Rtest, ...
        C, espilon, solverOpt, noCluster);
elseif ~isempty(strfind(solverOpt, 'cplex'))
    [Rpred, Rtest] = SVR_cplex(Dtrain, Rtrain, Dtest, Rtest, ...
    C, espilon, solverOpt, noCluster);
end

