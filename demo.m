% A piecewise linear regression model using k-means clustering and primal SVR
% Contact: Frank Puk pukkinming@gmail.com
%
% Note: A k-means clustering is run before applying SVR in each local
% region. Parameters such as the number of cluster regions, C and epsilon
% should be well tuned. Clustered SVR shows superior regression quality in
% terms of MSE and CPU time.
%
% Run "demo.m" to get a taste of the model.

clear all; close all; clc;

load autoMPG
displayOpt = 0;
noFold = 5;
C = 1;
espilon = 0.1;
noCluster = 5;
addpath(genpath('/usr/local/opt/ibm/ILOG/CPLEX_Studio127/cplex/matlab'))
addpath(genpath('./liblinear-2.01'))

solverOpt = 'liblinear';
startTime = tic;
[pred_liblinear, MSE_liblinear] = SVR_main(data, response, ...
    solverOpt, noFold, displayOpt, C, espilon, noCluster);
disp([solverOpt, ' / MSE: ', num2str(MSE_liblinear), ...
    ' / Time: ', num2str(toc(startTime)), ' seconds'])

solverOpt = 'liblinear_cluster';
startTime = tic;
[pred_liblinear_cluster, MSE_liblinear_cluster] = SVR_main(data, response, ...
    solverOpt, noFold, displayOpt, C, espilon, noCluster);
disp([solverOpt, ' / MSE: ', num2str(MSE_liblinear_cluster), ...
    ' / Time: ', num2str(toc(startTime)), ' seconds'])

solverOpt = 'cplex';
startTime = tic;
[pred_cplex, MSE_cplex] = SVR_main(data, response, ...
    solverOpt, noFold, displayOpt, C, espilon, noCluster);
disp([solverOpt, ' / MSE: ', num2str(MSE_cplex), ...
    ' / Time: ', num2str(toc(startTime)), ' seconds'])

solverOpt = 'cplex_cluster';
startTime = tic;
[pred_cplex_cluster, MSE_cplex_cluster] = SVR_main(data, response, ...
    solverOpt, noFold, displayOpt, C, espilon, noCluster);
disp([solverOpt, ' / MSE: ', num2str(MSE_cplex_cluster), ...
    ' / Time: ', num2str(toc(startTime)), ' seconds'])


