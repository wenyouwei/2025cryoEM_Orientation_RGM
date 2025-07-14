function [JR,JRK] = convenge_condition (R_new,R_old,C,tk)

% JR & JRk =\sum_{i,j}||RiCij - RjCji||

K = size(R_new,3);

%% JR, JRk
JR = 0;
JRk = 0;
for  i = 1:K
    for j = 1:K
        JR  = JR  + (norm(R_new(:,1:2,i)*C(1:2,i,j) - R_new(:,1:2,j)*C(1:2,j,i),2))^2;
        JRk = JRk + (norm(R_old(:,1:2,i)*C(1:2,i,j) - R_old(:,1:2,j)*C(1:2,j,i),2))^2;
    end
end

%% Grad_R * (R_{k+1}-R_k)
Grad_R = 0;
for i = 1:K
    Grad_Ri = zeros(3,2);
    for j = 1:K
%         if i ~=j
%             Grad_Ri = Grad_Ri + R_old(:,:,j) *C(:,j,i)*C(:,i,j)';
            Grad_Ri = Grad_Ri + (R_old(:,1:2,i) *C(1:2,i,j)-R_old(:,1:2,j) *C(1:2,j,i))*C(1:2,i,j)';

%         end
    end
    Grad_R = Grad_R + sum(dot(Grad_Ri,R_new(:,1:2,i)-R_old(:,1:2,i)));
end

% || R-R_old ||_F^2
R_Rk = 0;
for i = 1:K
    R_Rk = R_Rk + (norm(R_new(:,1:2,i)-R_old(:,1:2,i), 'fro'))^2;
end
R_Rk  =  R_Rk /(2*tk);
if R_Rk ==Inf
    R_Rk =0;
end
JRK = JRk + Grad_R + R_Rk;