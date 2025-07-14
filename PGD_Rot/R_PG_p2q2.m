function [est_rots,  MSE_iter, iter]= R_PG_p2q2(C, Param)
% 2019-10-14
%% Object:
%  \sum_{ij} \min 1/2*|| RiCij - RjCji||_2^2
%  s.t  Ri'Ri = I
%  Ri^{k+1} = Ri - t*(RiCij - RjCji)*Cij^{T}
%% Proximal Gradient with Backtracking Line-Search+weight
%
if isfield(Param,'ref_rot'),   ref_rot  = Param.ref_rot;   end
if isfield(Param,'RotsInit'), Rold    = Param.RotsInit; end
if isfield(Param,'LSInit'),   LSInit  = Param.LSInit;   end
if isfield(Param,'MaxIter');  MaxIter = Param.MaxIter;  end

TOL=1.0e-14;
K  = size(C,3);
X = zeros(3,K,K);
X(1:2,:,:) = C(1:2,:,:);
C = X;

tk = 1 %[0.99,0.7,0.618,0.5,0.2,0.08]
alpha =0.618; %0.618;%0.99; % SNR = 1/16 alpha =0.618
%% initial values

R_old   = rand(3,3,K);
for i = 1:K
    [U,~,V]      = svd(R_old(:,:,i));
    R_old(:,:,i) = U * V';
end
R_new = R_old;

iter_k = 0;

%------
for iter = 1:100
    %tk = 0.618;
    flag =1;
    kk = 0;
    [GradRi,JRK] = FunCostAndGradp2q2(R_old(:,1:2,:),C(1:2,:,:));
    %[GradRi,JRK] = FunCostAndGrad(R_old(:,1:2,:),C(1:2,:,:));
    Grad_Ri = zeros(3,3,K);
    Grad_Ri(:,1:2,:)= GradRi;
    objvalue(iter) = JRK;

    while flag
        kk = kk+1;
        GradObj = 0;
        for i =1:K
            temp = R_old(:,:,i) - tk * Grad_Ri(:,:,i);
            [U,~,V] = svd(temp,0);
            R_new(:,:,i) = U*V';

            tmp     = Grad_Ri(:,:,i).* (R_new(:,:,i)-R_old(:,:,i));
            GradObj = GradObj + sum(tmp(:)); % Grad_R*(Rnew-Rold)
        end

        %------
        [~,JR] = FunCostAndGradp2q2(R_new(:,1:2,:),C(1:2,:,:));
        %[~,JR] = FunCostAndGrad(R_new(:,1:2,:),C(1:2,:,:));
        JRKnew = JRK + GradObj + (norm(R_new(:)-R_old(:))^2)/(2*tk);

        if JR > JRKnew && kk<10
            tk = alpha*tk;
        else
            flag = 0;
        end
    end

    Errer(iter) = norm(R_new(:) - R_old(:))/norm(R_new(:));

    R_old = R_new;
    if Errer(iter) <1e-6
        fprintf('iter = %d\n',iter); break; end

%% Make sure that we got rotations.
    est_rots = zeros(3,3,K); 
    for k=1:K
        est_rots(:,:,k) = [R_new(:,1:2,k),cross(R_new(:,1,k),R_new(:,2,k))];
        R = est_rots(:,:,k);
        erro = norm(R*R.'-eye(3));
        if erro > TOL || abs(det(R)-1)> TOL
            [U,~,V] = svd(R);
            est_rots(:,:,k) = U*V.';
        end
    end
    MSE_iter(iter) = check_MSE(est_rots, ref_rot);
end

%% Make sure that we got true rotations.
est_rots = zeros(3,3,K);
for k=1:K
    est_rots(:,:,k) = [R_new(:,1:2,k),cross(R_new(:,1,k),R_new(:,2,k))];
    R = est_rots(:,:,k);
    erro = norm(R*R.'-eye(3));
    if erro > TOL || abs(det(R)-1)> TOL
        [U,~,V] = svd(R);
        est_rots(:,:,k) = U*V.';
    end
end



end

