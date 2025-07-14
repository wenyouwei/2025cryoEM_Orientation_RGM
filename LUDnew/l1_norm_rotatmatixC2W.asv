function est_rots = l1_norm_rotatmatixC2W(common_lines_matrix,n_theta, ref_rot)
% 
epsilon = 1e-3;
TOL=1.0e-14;
K  = size(common_lines_matrix,1);
err_c =0;
C  = clstack2C( common_lines_matrix,n_theta ); % common line matrixs

%%  s,t
kk =0;
 for t = 0.0001:0.005:0.1
     kk = kk+1; 
s = 0.01;%1.618;
%t = 0.065;%1.618;
eta = 1.618;
z_old  = rand(3, K, K);

% R_old  = zeros(3,2,K);
% R_old(1,1,:) = 1;
% R_old(2,2,:) = 1;
R_old = rand(3,3,K);
for i = 1:K
    [U,~,V] = svd(R_old(:,:,i));
    R_old(:,:,i) = U * V';
end
R_new = R_old;

for iter = 1:100
    % W ---------------
    for i = 1:K
        for j = i + 1:K
           Cij = C(:,i,j)*C(:,j,i)'; 
           Rij = R_old(:,1:2,i)'*R_old(:,1:2,j);
           wij = Cij.*Rij;
           W(i,j) = 1./sqrt(2-2*sum(wij(:)) + epsilon^2);
           W(j,i) =  W(i,j);
        end
    end
    
    for i = 1:K
        for j = i + 1:K
           z_new(:,i,j) = z_old(:,i,j) + W(i,j)*s*(R_old(:,1:2,i)*C(:,i,j)- R_old(:,1:2,j)*C(:,j,i));
           z_new(:,j,i) = z_old(:,j,i) + W(i,j)*s*(R_old(:,1:2,j)*C(:,j,i)- R_old(:,1:2,i)*C(:,i,j));
           if z_new(:,i,j) > 1
               z_new(:,i,j) = z_new(:,i,j)/norm(z_new(:,i,j));
           end
           if z_new(:,j,i) > 1
               z_new(:,j,i) = z_new(:,j,i)/norm(z_new(:,j,i));
           end
%             z_new(z_new>1) = 1; z_new(z_new<-1) = -1;
        end
    end
    z_hat = z_old +eta*(z_new - z_old);
    %%  R
    tmp = zeros(3,3);
    for i =1:K
        for j = 1:K
            tmp = tmp +  W(i,j)*(z_hat(:,i,j) - z_hat(:,j,i))*[C(:,i,j)' 0]; 
        end
        tmp2 = R_old(:,:,i) - t * tmp;
        [U,~,V] = svd(tmp2,0);
        tmp = U*V';
        R_new(:,:,i) = U*V';
    end
    
    Errer(iter) = norm(R_new(:) - R_old(:))/norm(R_new(:));
    if norm(R_new(:) - R_old(:))/norm(R_new(:))<1e-2
        break;
    end 
    z_old = z_new;
    R_old = R_new;
end
plot(1:iter, Errer);
% est_inv_rots = R_new;
% return;

% look for a linear transformation (3 x 3 matrix) A such that
% A*V1'=R1 and A*V2=R2 are the columns of the rotations matrices.
V1 = squeeze(R_new(:,1,:));
V2 = squeeze(R_new(:,2,:));
A = ATA_solver(V1,V2,K);% V1*A'=R1 and V2*A'=R2
R1 = A*V1;
R2 = A*V2;
R3 = cross(R1,R2);

est_rots = zeros(3,3,K);
% Make sure that we got rotations.
for k=1:K
    est_rots(:,:,k) = [R1(:,k),R2(:,k),R3(:,k)];
    R = est_rots(:,:,k);
    erro = norm(R*R.'-eye(3));
    if erro > TOL || abs(det(R)-1)> TOL
        [U,~,V] = svd(R);
        est_rots(:,:,k) = U*V.';
    end
    % Enforce R to be a rotation (in case the error is large)
end

inv_est_rot = permute(est_rots, [2 1 3]);
[MSEs(kk),inv_est_rot,~] = check_MSE(inv_est_rot, ref_rot);
end
figure;plot(1:kk,MSEs);



% function [W, res] = Weights(S, Gram, K, epsilon)
% W = S.*Gram;
% weights = W(1:K,1:K)+W(1:K,K+1:2*K)+W(K+1:2*K,1:K)+W(K+1:2*K,K+1:2*K);
% weights = sqrt(abs(2-2*weights));
% res = sum(weights(:));
% weights = 1./sqrt(weights.^2+epsilon^2);
% W = [weights weights; weights weights];
