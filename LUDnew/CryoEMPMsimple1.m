function [Gram] = CryoEMPMsimple1(C, n_theta,ref_rot)

%% Object:
%  min || Cij - GijCji||1
%  s.t  A(G) = b
%       G >=0
%      ||G||_1 <= alpha *K

%% Solution:  Primal Dual 

K = size(C,2);
alpha = 2/3;

n = 2*K;
m = 3*K;
idx  = 1:n;
idx2 = 1:K;
col = [K: n-1]*n +idx2;
col = col';

% opteigs
nev = 0;
opteigs.maxit = 100; 
opteigs.issym = 1;
opteigs.isreal = 1;
opteigs.disp = 0; 
opteigs.tol = 1e-6;
max_rankX = max(6, floor(K/4));
max_rankZ = max(6, floor(K/4));
kk = 0; nev = 0;
opteigs.maxit = 100; opteigs.issym = 1;
opteigs.isreal = 1; opteigs.disp = 0; opteigs.tol = 1e-6;

EPS = 1e-8;
mu = 1;

k =1;
MSEs = zeros(50,1);
 %% step size
eta = 1.618; s = 0.25;t = 0.01;

 % set initial 
 G_old = eye(2*K);
 b = [ones(n,1); zeros(K,1)];
 nrmb = norm(b);
 theta_old = C2theta(G_old,C,1);
 theta_hat = zeros(2,K,K);
 S = Qtheta(theta_old, C);
 AS = ComputeAX(S);
 
 y_old =  ComputeAX(zeros(2*K));
%  Z_old = eye(n);
%  X_old = eye(n);
%  

       
 iter = 0;
%  while 1
for k = 1:500
    iter = iter +1 ;
%% First step
    % G 
    % G
     ATy = ComputeATy(y_old);
     S = Qtheta(theta_old, C);
     H = G_old +t*(S + ATy);
     H = (H+H')/2;
     % compute nev
     if iter == 1 
        nev = max_rankX;
     else
         if nev  >= 0
             drops = dH(1:end-1)./dH(2:end);
             [dmx, imx] = max(drops);
             rel_drp = (nev-1)*dmx/(sum(drops)-dmx);
             if rel_drp >50
                nev= max(imx,6);
             else 
                 nev = nev+5;    
             end
         else
             nev =6;
         end    
     end  
     % by PH
     if nev>200
        nev = 200; 
     end 
     %  eig(H) 
     [V , dH] = eigs(H,nev, 'la',opteigs);
     dH = diag(dH);
     
   % By PH ------------------------  
     dH(dH>alpha*K)= alpha*K;
     dH(dH<0) = 0;  
     nD = dH>EPS;
     nev = nnz(nD);  
   % By PH ------------------------
   
     if nev>0
         dH = dH(nD);
         WmH = V(:,nD)*diag(dH)*V(:,nD)';  
     else
         WmH = sparse(n,n);   
     end 
     % X 
    G_new = WmH;
    G_hat = G_old + eta*(G_new-G_old);

     % y
     y_new = y_old - s*(ComputeAX(G_hat) -b);
     
     % theta
     for k1 = 1:K
       for k2 = k1+1:K
          c_ij = C(:,k1,k2);
          c_ji = C(:,k2,k1);
          phij = G_hat(2*k1-1:2*k1, 2*k2-1:2*k2);
          theta_new(:,k1,k2) = (theta_old(:,k1,k2)+ s*(c_ij - phij*c_ji))./max(1,abs(phij*c_ji-c_ij));
       end     
     end 

   % converge 
  ER_G(iter) =  norm(G_old(:)-G_new(:));
%    if iter >1
%        if norm(G_old(:)-G_new(:))<1e-4 || iter> 80
%            break
%        end
%    end
     
   y_old = y_new;
   theta_old = theta_new;
   G_old = G_new;
   
   
%%  MSE
Gram = eye(2*K);
Gram=[G_old(1:2:2*K,1:2:2*K) G_old(1:2:2*K,2:2:2*K);...
    G_old(2:2:2*K,1:2:2*K) G_old(2:2:2*K,2:2:2*K)];
fprintf('Startig the rounding procedure\n');
est_inv_rots = deterministic_rounding(Gram);
MSEs(iter)  = check_MSE(est_inv_rots, ref_rot);
   
 end
  Gram = G_old;
  figure;plot(1:iter,MSEs);title('MSE = |R-est(R)|');
  figure;plot(1:iter,ER_G);title('Error = |Gnew-Gold|');


 %% sub function
 function AX = ComputeAX(X)
    AX = [spdiags(X,0); sqrt(2)*X(col)] ;
 end

    function ATy = ComputeATy(y)
        ATy = sparse(n,n);
        ATy(col) = (sqrt(2)/2)*y(n+1:m);
        ATy = ATy + ATy'+sparse(idx,idx,y(1:n),n,n);
    end

end

 




function theta = C2theta(Phi,C,mu)
   K = size(C, 2);
   theta = zeros(2,K,K);
   
   for k1 = 1:K
       for k2 = k1+1:K
          c_ij = C(:,k1,k2);
          c_ji = C(:,k2,k1);
          phij = Phi(2*k1-1:2*k1, 2*k2-1:2*k2);
%           phji = Phi(2*k2-1:2*k2, 2*k1-1:2*k1);
          
          theta(:,k1,k2) = (c_ij - phij*c_ji)/norm(phij*c_ji-c_ij);
%           theta(:,k2,k1) = (c_ji - phji*c_ji)/norm(phji*c_ij-c_ji);

       end     
   end
end
   
   
    
   function S = Qtheta(theta,C)
        % Update S
        % Lanhui Wang
        % Jul 18, 2012
        
        K=size(C,2);
        S=zeros(2*K);
        
        for k1 = 1:K
            for k2 = k1+1:K
                theta_ij = theta(:,k1,k2);
%                 theta_ji = theta(:,k2,k1);
                c_ji = C(:,k2,k1);
                c_ij = C(:,k1,k2);
                S(2*k1-1:2*k1,2*k2-1:2*k2) = theta_ij*c_ji';
%                 S(2*k2-1:2*k2,2*k1-1:2*k1) = theta_ji*c_ij';

            end
        end
        
        S=(S+S')/2;
    end
 
 
 




