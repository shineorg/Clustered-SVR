% Ref: http://www.saedsayad.com/support_vector_machine_reg.htm
function [Rpred, Rtest] = SVR_cplex(Dtrain, Rtrain, Dtest, Rtest, ...
    C, espilon, solverOpt, noCluster)

Rpred = [];

if isempty(strfind(solverOpt, 'cluster'))
    try
        cplex = Cplex('SVR_v1_cplex_API');
        cplex.Model.sense = 'minimize';
        obj = [ones(size(Dtrain,2)*2, 1);    0; C * ones(size(Dtrain, 1)*2, 1)];
        lb = [zeros(size(Dtrain,2)*2, 1); -inf;    zeros(size(Dtrain, 1)*2, 1)];
        ub = [ inf * ones(size(lb, 1), 1)];
        cplex.addCols(obj, [], lb, ub);

        A = [-Dtrain Dtrain -ones(size(Dtrain,1), 1) -eye(size(Dtrain, 1)) zeros(size(Dtrain, 1)); ...
            Dtrain -Dtrain ones(size(Dtrain,1), 1) zeros(size(Dtrain, 1)) -eye(size(Dtrain, 1))];
        lhs = [-inf * ones(size(A, 1), 1)];
        rhs = [-Rtrain + espilon*ones(size(Rtrain)); ...
                Rtrain + espilon*ones(size(Rtrain))];
        cplex.addRows(lhs, A, rhs);

        cplex.DisplayFunc = [];
        cplex.solve();
    catch m
        disp(m.message);
    end

    w = cplex.Solution.x(1:size(Dtrain,2)) - cplex.Solution.x(size(Dtrain,2)+1:size(Dtrain,2)*2);
    b = cplex.Solution.x(size(Dtrain,2)*2+1);

    Rpred = Dtest * w + b;

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
        
        try
            cplex = Cplex('SVR_v1_cplex_API');
            cplex.Model.sense = 'minimize';
            obj = [ones(size(DtrainCluster,2)*2, 1);    0; C * ones(size(DtrainCluster, 1)*2, 1)];
            lb = [zeros(size(DtrainCluster,2)*2, 1); -inf;    zeros(size(DtrainCluster, 1)*2, 1)];
            ub = [ inf * ones(size(lb, 1), 1)];
            cplex.addCols(obj, [], lb, ub);

            A = [-DtrainCluster DtrainCluster -ones(size(DtrainCluster,1), 1) -eye(size(DtrainCluster, 1)) zeros(size(DtrainCluster, 1)); ...
                DtrainCluster -DtrainCluster ones(size(DtrainCluster,1), 1) zeros(size(DtrainCluster, 1)) -eye(size(DtrainCluster, 1))];
            lhs = [-inf * ones(size(A, 1), 1)];
            rhs = [-RtrainCluster + espilon*ones(size(RtrainCluster)); ...
                    RtrainCluster + espilon*ones(size(RtrainCluster))];
            cplex.addRows(lhs, A, rhs);

            cplex.DisplayFunc = [];
            cplex.solve();
        catch m
            disp(m.message);
        end

        w = cplex.Solution.x(1:size(DtrainCluster,2)) - cplex.Solution.x(size(DtrainCluster,2)+1:size(DtrainCluster,2)*2);
        b = cplex.Solution.x(size(DtrainCluster,2)*2+1);

        Rpred = DtestCluster * w + b;
        
        
        RpredAll = [RpredAll; Rpred];
        RtestAll = [RtestAll; RtestCluster];
    end
    Rpred = RpredAll;
    Rtest = RtestAll;
end
