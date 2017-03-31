function [Dtrain, Rtrain, Dtest, Rtest] = SVR_dataPrep(feat_nfold, ...
    response_nfold, iFold, noFold)

Dtrain = []; Rtrain = []; Dtest = []; Rtest = [];
idx = setdiff(1:noFold, iFold);

for i = idx
    eval(['temp = feat_nfold.fold', num2str(i), ';']);
    Dtrain = [Dtrain; temp];
    eval(['temp = response_nfold.fold', num2str(i), ';']);
    Rtrain = [Rtrain; temp];
end

eval(['Dtest = feat_nfold.fold', num2str(i), ';'])
eval(['Rtest = response_nfold.fold', num2str(i), ';'])