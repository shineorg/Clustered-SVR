function [RpredAll, mseAll] = SVR_main(feat, response, solverOpt, ...
    noFold, displayOpt, C, espilon, noCluster)

% feat     - observation by feature
% response - observation by 1
% solver   - choice of your solver (cplex, cplex_cluster, liblinear, liblinear_cluster)
% noFold   - 5, 10
% display  - display output or not

[feat_nfold, response_nfold] = SVR_splitData(feat, response, noFold);
RpredAll = []; RtestAll = [];

for iFold = 1:noFold
    startTime = tic;
    [Rpred, Rtest] = SVR_eval(feat_nfold, response_nfold, ...
        iFold, noFold, solverOpt, C, espilon, noCluster);
    RpredAll = [RpredAll; Rpred];
    RtestAll = [RtestAll; Rtest];
    mse = sqrt(sum((Rpred(:) - Rtest(:)).^2))/size(Rpred,1);
    if displayOpt
        disp(['Fold ', num2str(iFold), ' done! / MSE: ', num2str(mse), ...
            ' / Total time elapsed: ', num2str(toc(startTime)), ' seconds ...']);
    end
end

mseAll = sqrt(sum((RpredAll(:) - RtestAll(:)).^2))/size(RpredAll,1);
if displayOpt
    disp(['Done! Overall MSE: ', num2str(mseAll)]);
end

         