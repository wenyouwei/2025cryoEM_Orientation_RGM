
function est_inv_rots2 = l1_norm_rotatmatixC1(clstack,n_theta, rot_ture)

Kproj  = size(clstack,1);
%% commom matrix C is different from c
K = Kproj;
L = n_theta;
C=zeros(3,K,K);
for k1=1:K
    k2=(k1+1):K;
    l1 = clstack(k1,k2)-1;
    l2 = clstack(k2,k1)-1; 
    l1 = l1(:);
    l2 = l2(:);
    x12 = cos(2*pi*l1/L);
    y12 = sin(2*pi*l1/L);
    x21 = cos(2*pi*l2/L);
    y21 = sin(2*pi*l2/L);
    
    C(1,k1,k2)=x12;
    C(2,k1,k2)=y12;
    C(1,k2,k1)=x21;
    C(2,k2,k1)=y21;
end


%% test optimal s,t
%kk =0;
% for t = 0.0001:0.0005:0.01
%     kk =kk+1; iter =0; i =0; 
    
 s = 0.005;%1.618;
 t = 0.008;%1.618;
% initial value of R
%Rold  = zeros(3,3,Kproj);
Rold  = zeros(3,3,Kproj);

Rold(1,1,:) = 1; Rold(2,2,:) = 1; Rold(3,3,:) = 1;
Rold2  = Rold ;
zold2  = zeros(3, Kproj, Kproj);
zold2(3,:,:)= 1;
tic;

for iter = 1:10 

%% Csae2
    for i = 1:Kproj
        % -- zij
        for j = i+1:Kproj
            zij = zold2(:,i,j) + s*(Rold2(:,:,i)*C(:,i,j)-Rold2(:,:,j)*C(:,j,i));
            zji = zold2(:,j,i) + s*(Rold2(:,:,j)*C(:,j,i)-Rold2(:,:,i)*C(:,i,j));
            szij = sqrt(zij(:)'*zij(:));
            szji = sqrt(zji(:)'*zji(:));
            if szij>1
               zij(:) = zij(:)./szij;  
            end
            if szji>1
               zji(:,:) = zji(:,:)./szji;  
            end  
            % zij = -zji;
            zhat2(:,i,j) = zij;
            zhat2(:,j,i) = zji;     
            
        end
    end
    
    %% -- update R_ij
    tmp = eye(3,3);
    for i =1:Kproj
        for j = 1:Kproj
            %tmp = tmp + c(:,i,j) * (zhat(:,i,j) - zhat(:,j,i))';
            tmp = tmp +   (zhat2(:,i,j) - zhat2(:,j,i))*C(:,i,j)'; % 2019-6-26
        end
        Rnew2(:,:,i) = Rold2(:,:,i) - t * tmp;
%         [U,~,V] = svd(Rnew2(:,:,i),0);
%         R = U * V';
%         est_inv_rots2(:,:,i) = [R(:,1) R(:,2) cross(R(:,1),R(:,2))];
    end
   
    for i = 1 : Kproj
       [U,~,V] = svd(Rnew2(:,:,i),0);
       R = U * V';
       est_inv_rots2(:,:,i) = [R(:,1) R(:,2) cross(R(:,1),R(:,2))]; 
    end
        
    for i = 1:Kproj
        for j = i+1:Kproj
            zij = zold2(:,i,j) + s*(Rnew2(:,:,i)*C(:,i,j)-Rnew2(:,:,j)*C(:,j,i));
            zji = zold2(:,j,i) + s*(Rnew2(:,:,j)*C(:,j,i)-Rnew2(:,:,i)*C(:,i,j));
            
%             sumza = sqrt(tmpa(:)'*tmpa(:));
%             sumzb = sqrt(tmpb(:)'*tmpb(:));
            szij = norm(zij); szij(szij<1) = 1;
            szji = norm(zji);szji(szji<1)  = 1;
            tmpa(:) = zij(:)./szij;
            zji(:) = zji(:)./szji;
          
            zhat2(:,i,j) = zij;            
            zhat2(:,j,i) = zji;            
        end
    end
    
    
    ER_l1a2(iter) = check_MSE(Rnew2,rot_ture);
    ER_l1c2(iter) = check_MSE(est_inv_rots2,rot_ture);
    inv_rotations_l12 = permute(est_inv_rots2,[2,1,3]);
    ER_l1b2(iter) = check_MSE(inv_rotations_l12,rot_ture);
   
    zold2 = zhat2;
    Rnew2 = est_inv_rots2;
    Rold2 = Rnew2;
end


figure;plot(1:iter,ER_l1a2);
figure;plot(1:iter,ER_l1c2);
figure;plot(1:iter,ER_l1b2);
fprintf('t =%1.5f', t);
time = toc

% test optimal s,t
% ER_l1aa(kk)= ER_l1a(iter); ER_l1cc(kk)= ER_l1c(iter); ER_l1bb(kk)=ER_l1b(iter);
% end
% figure;plot(1:kk,ER_l1aa);
% figure;plot(1:kk,ER_l1cc);
% figure;plot(1:kk,ER_l1bb);
 

