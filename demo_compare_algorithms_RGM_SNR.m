%% Riemannian Gradient Method (RGM) and compared methods
%% Experiments with simulated noisy cryoEM images under different SNR
%% 2024-6-4 % modified by Huan Pan

clc; clear all; close all

addpath(genpath('utilsnufft'))
addpath(genpath('utilscryoem'))
addpath(genpath('other_file'))
addpath(genpath('Objvalue_err'))
addpath(genpath('LUDnew2'))
addpath(genpath('Rot_RGM'))
addpath(genpath('PGD_Rot'))


% download 3D data set
load cleanrib;
n = size(volref,1)
n1 = 129;  % n = 65;129 137
V = NewSizeVol(volref,n1);
volref1 = V;

K = 500; %[1000, 2000 ,3000]% Generate K simulated projections
ref_rot = rand_rots(K);
A1     = OpNufft3D(ref_rot,n1); %% Generate simulated projections
projs1 = A1 * volref1;

%% Add CTF to Projection images
ctfs      = GenerateCTFset(n);% Add CTF to Projection images
defocusID = mod(0:K-1,size(ctfs,3))+1;
ctfprojs  = CTFtimesprojs(projs1, ctfs, defocusID); % add ctfs  to projections

%% add noise to projection images
SNR =  1; %[1, 1/2, 1/4, 1/8, 1/16, 1/32]
% Generate  noise projections
[noisy_projs1, sigma1] = ProjAddNoise(projs1, SNR);
%[noisy_projs1, sigma1] = ProjAddNoise(ctfprojs, SNR);
masked_r = 45; %43;
masked_projs=mask_fuzzy(noisy_projs1,masked_r); % Applly circular mask
n_theta = 144; %60; 72; 90; 120; 144;180; %360;
n_r = n; % n     %100;    %33
[npf,sampling_freqs]=cryo_pft(masked_projs,n_r,n_theta,'single');  % take Fourier transform of projections
% Find common lines from projections
max_shift=0;
shift_step=1;
common_lines_matrix = commonlines_gaussian(npf,max_shift,shift_step);
C = clstack2C( common_lines_matrix,n_theta );
[ref_clstack,~]=clmatrix_cheat(ref_rot,n_theta);
p = comparecl( common_lines_matrix, ref_clstack, n_theta, 10 );%Percentage of correct common lines

%% Test different orientation determination algorithms

%% Riemannian Gradient Method （RGM for LS）LS_RGM
k = 1 ;
beta=0.618;eta=1;
Param.RefRot = ref_rot;
Param.MaxIter = 1000;
t1 = clock;
[est_rots] =ManProxGradp2q2(C,Param,eta, beta);
t2 = clock;
Time(k) = etime(t2,t1);
[MSEs(k),est_inv_rots, err]= check_MSE(est_rots, ref_rot);
LS_RGM = [MSEs(k)  Time(k)];


%% Riemannian Gradient Method for LUD（RGM for LUD-IRLS_RGM）
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
LS_SDR = [ MSEs(k) Time(k) ];


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
LUD_IRLS_SDR = [MSEs(k) Time(k) ];


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





