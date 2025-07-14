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

t   = 1;  % 0.5
gamma = 0.618;%0.5   
alpha = 1;
myeps = 1e-1; 

[GradRi,FvalOld] = FunCostAndGradp2q1(Rold(:,1:2,:),C(1:2,:,:));
%[GradRi,FvalOld] = FunCostAndGradp2q1v2(Rold(:,1:2,:),C(1:2,:,:),myeps);

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
    %alpha = 1e-3; 
    
    for kk = 1:10
        for i =1:K
            % CayleyTransRots3X3(X, V)
            %I  = eye(3);
            %Rk = Rold(:,:,i);
            %Vk =  V(:,:,i);
            %W  = (Vk * Rk' - Rk * Vk')/4;
            %Rnew(:,:,i) = ((I - alpha * W)\(I + alpha * W)) * Rk;
            %Rnew(:,:,i) = CayleyTransRots3X3(Rold(:,:,i), V(:,:,i),alpha);
            Rnew(:,:,i) = PolarDecompRots3X3(Rold(:,:,i), V(:,:,i),alpha);            
        end
         %-- Armijo line Search for stepsize alpha 
        [GradRi,FvalNew] = FunCostAndGradp2q1(Rnew(:,1:2,:),C(1:2,:,:));
        %[GradRi,FvalNew] = FunCostAndGradp2q1v2(Rold(:,1:2,:),C(1:2,:,:),myeps);
        %JRKnew = FvalOld - alpha*(norm(Rnew(:)-Rold(:))^2)/(2*t);
        JRKnew = FvalOld - alpha*(norm(V(:))^2)/(2*t);
        if FvalNew<JRKnew, break; end
        alpha = alpha*gamma;
    end
    myeps = max([myeps/50 1e-10]);
    
    if exist('RefRot','var')
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
    [MSEs22] = check_MSE(est_rots, RefRot);
    fprintf('%3d, line search %2d: stepsize %1.2e, MSE: %f: Obj: %1.2e\n',iter,kk,alpha,MSEs22,FvalNew);
    end
    
    Errer  = norm(Rnew(:) - Rold(:))/norm(Rnew(:));

    Rold = Rnew;  
    FvalOld = FvalNew;     
    if Errer  <1e-6 , break; end
    alpha = max([alpha,1e-7]);
end

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