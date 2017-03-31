function [feat_nfold, response_nfold] = SVR_splitData(feat, response, noFold)

noObsns = size(feat, 1);
noFeat = size(feat, 2);
N = noFold;
L = size(feat,1);
n = floor(L/N);
rem = mod(L, N);
a = n*ones(N,1);
if rem>0
    b = nchoosek(1:N,rem);
    c = ceil(rand*size(b,1));
    idx = b(c,:);
%     a(idx)=a(idx) + 1;
    a(1:rem)=a(1:rem) + 1;
end
nfoldpt =[0; cumsum(a)];
nint = [nfoldpt(1:end-1)+1, nfoldpt(2:end)];

for i = 1:N
    dsub = feat(nint(i,1):nint(i,2), :);
    eval(['feat_nfold.fold' num2str(i), '=dsub;']);
    rsub = response(nint(i,1):nint(i,2), :);
    eval(['response_nfold.fold', num2str(i), '=rsub;'])
end
