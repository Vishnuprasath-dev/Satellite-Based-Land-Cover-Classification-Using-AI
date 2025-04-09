function [Lr2,center_Lab,U,iter_n]=FastFuzzyCMeans(L2,centerLab, Label_n,cluster_n)

data_n = size(centerLab, 1); %the row of input matrix
% Change the following to set default options
default_options = [2;	% exponent for the partition matrix U
    50;	% max. number of iteration
    1e-5;	% min. amount of improvement
    1];	% info display during iteration
    options = default_options;
expo = options(1);		% Exponent for U
max_iter = options(2);		% Max. iteration
min_impro = options(3);		% Min. improvement
display = options(4);		% Display info or not
iter_n=0; % actual number of iteration
U = initfcm(cluster_n, data_n);			% Initial fuzzy partition
Num=ones(cluster_n,1)*Label_n';
for i = 1:max_iter
    mf = Num.*U.^expo;       % MF matrix after exponential modification
    center = mf*centerLab./((ones(size(centerLab, 2), 1)*sum(mf'))'); % new center
    out = zeros(size(center, 1), size(centerLab, 1));
    if size(center, 2) > 1
        for k = 1:size(center, 1)
            out(k, :) = sqrt(sum(((centerLab-ones(size(centerLab, 1), 1)*center(k, :)).^2)'));
        end
    else	% 1-D data
        for k = 1:size(center, 1)
            out(k, :) = abs(center(k)-centerLab)';
        end
    end
    dist=out+eps;
    tmp = dist.^(-2/(expo-1));
    U = tmp./(ones(cluster_n, 1)*sum(tmp)+eps);
    Uc{i}=U;
    if i> 1
        if abs(max(max(Uc{i} - Uc{i-1}))) < min_impro, break; end
    end
end
iter_n = i;
center_Lab=center;
[~,IDX2]=max(U);
%%
Lr2=zeros(size(L2,1),size(L2,2));
for i=1:max(L2(:))
    Lr2=Lr2+(L2==i)*IDX2(i);
end