%% Riemannian Gradient Method (RGM) and compared methods
%% Experiments with simulated rotation under different P
%% 2024-6-4 % modified by Huan Pan

clc; clear all; close all
addpath(genpath('utilsnufft'))
addpath(genpath('utilscryoem'))
addpath(genpath('other_file'))
addpath(genpath('Objvalue_err'))
addpath(genpath('LUDnew'))
addpath(genpath('Rot_RGM'))
addpath(genpath('PGD_Rot'))


% projection number K
K = 500; %[500 1000 2000 3000]
% reference rotation matrix
ref_rot = rand_rots(K); %ref_rot;

P =  0.9;  %[0.9, 0.8, 0.7, 0.6, 0.5, 0.4] % the proportion of correctly detected common lines
n_theta = 360;  %% Notice: n_theta = 360 for P;
common_lines_matrix= ref_commlines(ref_rot, n_theta,P);
C = clstack2C( common_lines_matrix, n_theta);

%%  Riemannian Gradient Method（RGM for LS model）LS_RGM
k = 1 ;
beta=0.618;
eta=1;
Param.RefRot = ref_rot;
Param.MaxIter = 1000;
t1 = clock;
[est_rots] =ManProxGradp2q2(C,Param,eta, beta);
t2 = clock;
Time(k) = etime(t2,t1);
[MSEs(k),est_inv_rots, err]= check_MSE(est_rots, ref_rot);
LS_RGM = [MSEs(k)  Time(k)];


%% Riemannian Gradient Method for LUD（RGM for LUD-IRLS_RGM）
k = 2;
beta=0.618;eta=1;
Param.RefRot = ref_rot;
Param.MaxIter = 1000;
t1 = clock;
[est_rots] = ManProxGradp2q1(C, Param,eta,beta);
t2 = clock;
Time(k) = etime(t2,t1);
[MSEs(k),est_inv_rots, err]= check_MSE(est_rots, ref_rot);
LUD_IRLS_RGM = [ MSEs(k) Time(k) ];



%% semi-definite relaxation method (SDR) and the iteratively re-weighted least squares method (IRLS)

%%  SDR for LS model  (SDR_LS)
k = 3;
pars.alpha = 0;
tic;
est_inv_rots = est_orientations_LS(common_lines_matrix, n_theta, pars);
Time(k) = toc;
[MSEs(k), est_inv_rots]= check_MSE(est_inv_rots, ref_rot);
LS_SDR = [ MSEs(k) Time(k)];



%% SDR for LUD model  (LUD_SDR)
k = 4;
tic;
est_inv_rots = est_orientations_LUD(common_lines_matrix,n_theta);
Time(k) = toc;
[MSEs(k), est_inv_rots]= check_MSE(est_inv_rots, ref_rot);
LUD_SDR = [MSEs(k) Time(k)];


%% SDR for IRLS model  (LUD_IRLS_SDR)
k= 5;
pars.solver = 'IRLS';
pars.alpha = 0;
tic;
est_inv_rots = est_orientations_LUD(common_lines_matrix,n_theta, pars);
Time(k) = toc;
[MSEs(k), est_inv_rots] = check_MSE(est_inv_rots, ref_rot);
LUD_IRLS_SDR = [MSEs(k) Time(k)];

%% projection gradient decent method (PGD) for LS model  (LS-PGD)
k = 6;
Param.MaxIter = 500;
Param.ref_rot = ref_rot;
t1 = clock;
est_rots = R_PG_p2q2(C, Param);
t2 = clock;
Time(k) = etime(t2,t1);
[MSEs(k),est_inv_rots, err]= check_MSE(est_rots, ref_rot);
LS_PGD = [ MSEs(k) Time(k)];

%%  projection gradient decent method (PGD) for LUD model  (LUD-IRLS-PGD)
k = 7;
Param.MaxIter = 500;
Param.ref_rot = ref_rot;
t1 = clock;
est_rots = R_PG_p2q1(C, Param);
t2 = clock;
Time(k) = etime(t2,t1);
[MSEs(k),est_inv_rots, err]= check_MSE(est_rots, ref_rot);
LUD_IRLS_PGD = [ MSEs(k) Time(k)];



%% save results
Result_MSE = [LS_SDR;LS_PGD; LS_RGM; LUD_SDR; LUD_IRLS_PGD; LUD_IRLS_SDR;  LUD_IRLS_RGM];




