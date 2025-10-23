%% =================== Comparison between BINNO and other algorithms ===================
clear; close all; clc;

load("traffic_patches.mat");     

iClip = 10;                 % choose clip (1-254)

clip  = imgdb{iClip};      
clip  = mat2gray(double(clip));     
[H,W,T] = size(clip);
sizeim  = [H W];

% matrix 2D: column = frame
M_true = reshape(clip, H*W, T); %ground truth 

%% ============ Noise ============
use_synthetic_noise = true;

if use_synthetic_noise
    rank_true = 9; % with scree plot
    [U,S,V] = svd(M_true,'econ');
    L0 = U(:,1:rank_true) * S(1:rank_true,1:rank_true) * V(:,1:rank_true)';
    sparsity = 0.05; 
    mask = sprand(size(M_true,1), size(M_true,2), sparsity) > 0;

    factor = 0.33;
    S0 = zeros(size(M_true));
    S0(mask) = factor * (2*(rand(nnz(mask),1)>0.5)-1) .* abs(L0(mask));
    % Observed matrix = low-rank + sparse noise
    M_obs = L0 + S0;

    GT_lowrank = L0;        % ground truth lowrank
else
    M_obs = M_true;

    GT_lowrank = [];       
end

M_obs(~isfinite(M_obs)) = 0;


%% =================== 1) SLRF_BiNNO ===================
r = 9;        
itermax = 5*1e3;
lambda1 = 0.05; 
lambda2 = 0.05;
gamma1 = 0.01; 
gamma2  = 0.01;

tic;
[X, Y, Psi1, Psi2] = slrf_binno(M_obs, r, itermax, lambda1, lambda2, gamma1, gamma2);
t_slrf = toc;

L_slrf = X*Y;               % estimated low-rank part
S_slrf = M_obs - L_slrf;    % estimated sparse part 
rank_slrf = rank(L_slrf);

%% =================== 2) RPCA nmfLS2 ===================
tic;
[X2,Y2] = nmfLS2(M_obs, r, itermax);
L_nmfLS2 = X2 * Y2;
S_nmfLS2 = M_obs - L_nmfLS2;
t_nmfLS2 = toc;
rank_nmfLS2 = rank(L_nmfLS2);

%% =================== 3) nsa_v1 =======================
stdev = 1;
tol = 5e-6; % optimality tolerance for stopping_type 1
tic;
L3 = nsa_v1_original(M_obs,stdev,tol,1);
S3 = M_obs - L3;
t_nsa_v1 = toc;
rank_nsa_v1 = rank(L3);

%% =================== 4) nsa_v2 =======================
tic;
[L4,S4,out] = nsa_v2_original(M_obs,stdev,tol);
t_nsa_v2 = toc;
rank_nsa_v2 = rank(L4);

%% =================== Metrics ===================

% Reconstruction vs observations
recon_err_slrf = norm(M_obs - L_slrf, 'fro')/norm(M_obs,'fro');
recon_err_nmfLS2 = norm(M_obs - L_nmfLS2, 'fro')/norm(M_obs,'fro');
recon_err_nsa_v1 = norm(M_obs - L3, 'fro')/norm(M_obs,'fro');
recon_err_nsa_v2 = norm(M_obs - L4, 'fro')/norm(M_obs,'fro');


if ~isempty(GT_lowrank)
    % Comparison with the "true" low-rank
    mse_slrf = mean((L_slrf(:) - GT_lowrank(:)).^2);
    mse_nmfLS2 = mean((L_nmfLS2(:) - GT_lowrank(:)).^2);
    mse_nsa_v1 = mean((L3(:) - GT_lowrank(:)).^2);
    mse_nsa_v2 = mean((L4(:) - GT_lowrank(:)).^2);
    % PSNR global 
    psnr_slrf = 10*log10(1/mse_slrf);
    psnr_nmfLS2 = 10*log10(1/mse_nmfLS2);
    psnr_nsa_v1 = 10*log10(1/mse_nsa_v1);
    psnr_nsa_v2 = 10*log10(1/mse_nsa_v2);
end

%% =================== Visualizations ===================
% Qualitative
Mobs3  = reshape(M_obs,   H, W, T);

Lsl3  = reshape(L_slrf,  H, W, T);
Lnm3  = reshape(L_nmfLS2,   H, W, T);
Lns13 = reshape(L3,  H, W, T);
Lns23 = reshape(L4,  H, W, T);

Ssl3  = reshape(S_slrf,  H, W, T);
Snm3  = reshape(S_nmfLS2,   H, W, T);
Sns13 = reshape(S3,  H, W, T);
Sns23 = reshape(S4,  H, W, T);

to4d   = @(A) reshape(A, size(A,1), size(A,2), 1, size(A,3));
idxShow = unique(round(linspace(1,T,6)));

figure('Name','Comparison: SLRF vs NMF vs NSA v1/v2','Color','w');
subplot(4,5,1);  imshow(clip(:,:,1),[]);        title('Original');
subplot(4,5,2);  imshow(Mobs3(:,:,1),[]);       title('Observed');
subplot(4,5,3);  imshow(Lsl3(:,:,1),[]);        title('BINNO: L');
subplot(4,5,4);  imshow(Lnm3(:,:,1),[]);        title('NMF: L');
subplot(4,5,5);  imshow(Lns23(:,:,1),[]);       title('NSA v2: L');

subplot(4,5,6);  montage(to4d(Mobs3(:,:,idxShow)),'Size',[2 3]);       title('Observed');
subplot(4,5,7);  montage(to4d(Lsl3(:,:,idxShow)),'Size',[2 3]);        title('BINNO: L');
subplot(4,5,8);  montage(to4d(Lnm3(:,:,idxShow)),'Size',[2 3]);        title('NMF: L');
subplot(4,5,9);  montage(to4d(Lns13(:,:,idxShow)),'Size',[2 3]);       title('NSA v1: L');
subplot(4,5,10); montage(to4d(Lns23(:,:,idxShow)),'Size',[2 3]);       title('NSA v2: L');

subplot(4,5,11); montage(to4d(abs(Ssl3(:,:,idxShow))),'Size',[2 3]);   title('BINNO: |S|');
subplot(4,5,12); montage(to4d(abs(Snm3(:,:,idxShow))),'Size',[2 3]);   title('NMF: |S|');
subplot(4,5,13); montage(to4d(abs(Sns13(:,:,idxShow))),'Size',[2 3]);  title('NSA v1: |S|');
subplot(4,5,14); montage(to4d(abs(Sns23(:,:,idxShow))),'Size',[2 3]);  title('NSA v2: |S|');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if exist('idxShow','var') && ~isempty(idxShow)
    idxRef = idxShow(1);
else
    idxRef = 1;
end
%% === Figure 1
figure('Name','Original vs Observed','Color','w');
tiledlayout(1,2,'Padding','compact','TileSpacing','compact');

nexttile; imshow(clip(:,:,idxRef),[]);  title('Original');
nexttile; imshow(Mobs3(:,:,idxRef),[]); title('Observed');

%% === Figure 2
figure('Name','Single clip: 4 methods (L & |S|)','Color','w');
tiledlayout(2,4,'Padding','compact','TileSpacing','compact');

% --- L ---
nexttile; imshow(Lsl3(:,:,idxRef),[]);   title('BINNO: L');
nexttile; imshow(Lnm3(:,:,idxRef),[]);   title('NMF: L');
nexttile; imshow(Lns13(:,:,idxRef),[]);  title('NSA v1: L');
nexttile; imshow(Lns23(:,:,idxRef),[]);  title('NSA v2: L');

% --- |S| ---
nexttile; imshow(abs(Ssl3(:,:,idxRef)),[]);   title('BINNO: |S|');
nexttile; imshow(abs(Snm3(:,:,idxRef)),[]);   title('NMF: |S|');
nexttile; imshow(abs(Sns13(:,:,idxRef)),[]);  title('NSA v1: |S|');
nexttile; imshow(abs(Sns23(:,:,idxRef)),[]);  title('NSA v2: |S|');

%% === Figure 3
figure('Name','6 clips: 4 methods (L & |S|)','Color','w');
tiledlayout(2,4,'Padding','compact','TileSpacing','compact');

% --- L (6 clip) ---
nexttile; montage(to4d(Lsl3(:,:,idxShow)),'Size',[2 3]);   title('BINNO: L');
nexttile; montage(to4d(Lnm3(:,:,idxShow)),'Size',[2 3]);   title('NMF: L');
nexttile; montage(to4d(Lns13(:,:,idxShow)),'Size',[2 3]);  title('NSA v1: L');
nexttile; montage(to4d(Lns23(:,:,idxShow)),'Size',[2 3]);  title('NSA v2: L');

% --- |S| (6 clip) ---
nexttile; montage(to4d(abs(Ssl3(:,:,idxShow))),'Size',[2 3]);   title('BINNO: |S|');
nexttile; montage(to4d(abs(Snm3(:,:,idxShow))),'Size',[2 3]);   title('NMF: |S|');
nexttile; montage(to4d(abs(Sns13(:,:,idxShow))),'Size',[2 3]);  title('NSA v1: |S|');
nexttile; montage(to4d(abs(Sns23(:,:,idxShow))),'Size',[2 3]);  title('NSA v2: |S|');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%








figure('Name','Convergenza SLRF','Color','w');
semilogy(Psi1,'b--','LineWidth',2); hold on;
semilogy(Psi2,'r:','LineWidth',2); grid on;
legend('\Psi_1','\Psi_2','Location','best'); title('Convergence of BINNO'); xlabel('iter');

%% =================== Summary ===================
fprintf('\n=== Summary (clip %d) ===\n', iClip);
fprintf('Time SLRF: %.3f s | rank(L_slrf)=%d | recon_err=%.4f\n', t_slrf, rank_slrf, recon_err_slrf);
fprintf('Time nmfLS2: %.3f s | rank(L_nmfLS2)=%d | recon_err=%.4f\n', t_nmfLS2, rank_nmfLS2, recon_err_nmfLS2);
fprintf('Time NSA1: %.3f s | rank(L3)=%d | recon_err=%.4f\n', t_nsa_v1, rank_nsa_v1, recon_err_nsa_v1);
fprintf('Time NSA2: %.3f s | rank(L4)=%d | recon_err=%.4f\n', t_nsa_v2, rank_nsa_v2, recon_err_nsa_v2);

if ~isempty(GT_lowrank)
    fprintf('With GT low-rank:  MSE  BINNO=%.4e  NMFLS=%.4e NSAv1=%.4e NSAv2=%.4e\n', ...
        mse_slrf, mse_nmfLS2, mse_nsa_v1, mse_nsa_v2);
        fprintf('With GT low-rank: PSNR BINNO=%.2f dB  NMFLS=%.2f NSAv1=%.2f NSAv2=%.2fdB\n', ...
        psnr_slrf,psnr_nmfLS2, psnr_nsa_v1,  psnr_nsa_v2);
end


