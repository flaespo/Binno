clear;close all;clc
% synthetic data

m = 100;        % Rows
n = 80;         % Columns
r = 5;          % True rank
sparsity = 0.3; % 30% non-zero elements
noise_level = 0.01; % Small amount of noise
[M, X_true, Y_true] = generate_sparse_low_rank_data(m, n, r, sparsity, noise_level);

%%
itermax = 1e3;     % Number of iterations
lambda1 = 0.005;
lambda2 = 0.005;
gamma1  = 0.005;
gamma2  = 0.005;
[X, Y, Psi1, Psi2] = slrf_binno(M, r, itermax, lambda1, lambda2, gamma1, gamma2);

%%

% Visualize the matrix
figure;
subplot(221),spy(X_true); title('Sparse factor X');
subplot(222),spy(Y_true); title('Sparse factor Y');
subplot(223),imagesc(M); title('Resulting matrix M');colorbar;colormap gray

subplot(224)
semilogy(Psi1,'b--','LineWidth',3),hold on
semilogy(Psi2,'r:','LineWidth',3),grid on
xlabel('iteration $k$','Interpreter','latex','fontsize',22)
legend('$\psi_1$','$\psi_2$','Interpreter','latex','fontsize',22)
set(gca,'FontSize',22)