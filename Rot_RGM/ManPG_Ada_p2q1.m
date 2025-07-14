function est_rots = ManProxGradp2q1(C, Param)
% 2019-10-14
%% Object:
%  \sum_{ij} \min 1/2*|| RiCij - RjCji||_2^2
%  s.t  Ri'Ri = I
%  Ri^{k+1} = Ri - t*(RiCij - RjCji)*Cij^{T}
%% Proximal Gradient with Backtracking Line-Search

TOL=1.0e-14;
MaxIter = 500;
K  = size(C,3);
X = zeros(3,K,K);
X(1:2,:,:) = C(1:2,:,:);
C = X;
LSInit = 1;
Rold   = rand(3,3,K);
Rnew   = rand(3,3,K);
for i = 1:K
    [U,~,V]     = svd(Rold(:,:,i));
    Rold(:,:,i) = U * V';
end

if isfield(Param,'RefRot'),   RefRot  = Param.RefRot;   end
if isfield(Param,'RotsInit'), Rold    = Param.RotsInit; end
if isfield(Param,'LSInit'),   LSInit  = Param.LSInit;   end
if isfield(Param,'MaxIter');  MaxIter = Param.MaxIter;  end

if LSInit
    Pa.MaxIter =15;
    Rold = ManProxGradp2q2(C, Pa);
end
% Rold = RefRot;

% L = 1; % Lipschitz constant
t_min=1e-3; % minimum stepsize
t   = 1;%0.99;    % 0.5
tao  = 1.001;
gamma = 0.618; %0.618;;
alpha = 1;%0.618;
linesearch_flag = 0;
num_inexact = 0;
myeps = 1e-1;

[GradRi,FvalOld] = FunCostAndGradp2q1(Rold(:,1:2,:),C(1:2,:,:));
%[GradRi,FvalOld] = FunCostAndGradp2q1v2(Rold(:,1:2,:),C(1:2,:,:),myeps);
L = norm(GradRi(:));
L = 2;
t0   = max(1/L, t/tao);    % 0.5

for iter = 1:MaxIter
    Grad_Ri = zeros(3,3,K);
    Grad_Ri(:,1:2,:)= GradRi;
    
    %--- descent direction V
    V    = zeros(size(Rold));
    for i = 1:K
        Gk = Grad_Ri(:,:,i);
        Rk = Rold(:,:,i);
        V(:,:,i) = -t * (Gk - Rk * Gk' * Rk)/2;
    end
    
    for kk = 1:8
    for i =1:K
        %Rnew(:,:,i) = CayleyTransRots3X3(Rold(:,:,i), V(:,:,i),alpha);
        Rnew(:,:,i) = PolarDecompRots3X3(Rold(:,:,i), V(:,:,i),alpha);
    end
    [GradRi,FvalNew] = FunCostAndGradp2q1(Rnew(:,1:2,:),C(1:2,:,:));
    
    %% --- Armijo line Search for stepsize alpha
    if FvalNew > FvalOld - alpha*(norm(V(:))^2)/(2*t)
        kk = kk+1;
        linesearch_flag = 1;      
        alpha = alpha*gamma;
    end
%         if alpha<t_min
%             num_inexact = num_inexact + 1;
%             break;
%         end 
        
%         for i =1:K 
%             %Rnew(:,:,i) = CayleyTransRots3X3(Rold(:,:,i), V(:,:,i),alpha);
%             Rnew(:,:,i) = PolarDecompRots3X3(Rold(:,:,i), V(:,:,i),alpha);
%         end
%         [GradRi,FvalNew] = FunCostAndGradp2q1(Rnew(:,1:2,:),C(1:2,:,:));
%         alpha = max([alpha,1e-7]);
    end
    
    Errer  = norm(Rnew(:) - Rold(:))/norm(Rnew(:));
    Rold = Rnew;
    FvalOld = FvalNew;
    if Errer  <1e-6 , break; end
    
    %-- Backtrack search for stepsize t
    if linesearch_flag == 1
        t = tao *t;
    else
        t = max(t0, t/tao);
    end
    linesearch_flag = 0;
    
%     % ---Make sure that we got rotations.
%     if exist('RefRot','var')
%         est_rots = zeros(3,3,K);
%         for k=1:K
%             est_rots(:,:,k) = [Rnew(:,1:2,k),cross(Rnew(:,1,k),Rnew(:,2,k))];
%             R = est_rots(:,:,k);
%             erro = norm(R*R.'-eye(3));
%             if erro > TOL || abs(det(R)-1)> TOL
%                 [U,~,V] = svd(R);
%                 est_rots(:,:,k) = U*V.';
%             end
%         end
%         [MSEs22] = check_MSE(est_rots, RefRot);
%         MSEs(iter) = MSEs22;
%         fprintf('%d, line search %2d: step length %1.2e, MSE: %f: Obj: %1.2e\n',iter,kk,alpha,MSEs22,FvalNew);
%     end
    
end
% figure(9);plot(1:iter-1,MSEs);title('MSE');
%% Make sure that we got rotations.
est_rots = zeros(3,3,K);
for k=1:K
    est_rots(:,:,k) = [Rnew(:,1:2,k),cross(Rnew(:,1,k),Rnew(:,2,k))];
    R = est_rots(:,:,k);
    erro = norm(R*R.'-eye(3));
    if erro > TOL || abs(det(R)-1)> TOL
        [U,~,V] = svd(R);
        est_rots(:,:,k) = U*V.';
    end
end

function Z = CayleyTransRots3X3(X, V,alpha)
I  = eye(3);
W  = (V * X' - X * V')/4;
Z  = ((I - alpha * W)\(I + alpha * W)) * X;

function Z = PolarDecompRots3X3(X, V,alpha)
V = alpha * V;
T = eye(3) + V' * V;
[U,d] = svd(T);
d     = diag(d); d = 1./sqrt(d);
Z     = (X + V)*(U * diag(d) * U');
