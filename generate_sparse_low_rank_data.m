function [M, X_true, Y_true] = generate_sparse_low_rank_data(m, n, r, sparsity, noise_level)
% Inputs
%   m, n        : dimensions of the matrix M (m x n)
%   r           : true rank of the matrix
%   sparsity    : fraction of non-zero elements in factors (0-1)
%   noise_level : standard deviation of Gaussian noise to add
% Outputs
%   M           : generated matrix (m x n)
%   X_true      : true sparse factor (m x r)
%   Y_true      : true sparse factor (r x n)

X_true   = sprandn(m, r, sparsity);  % Sparse matrix with normal distribution
Y_true   = sprandn(r, n, sparsity);  % Sparse matrix with normal distribution
M_clean  = X_true * Y_true;
noise    = noise_level * randn(m, n);
M        = M_clean + noise;

% Display information
fprintf('Generated matrix of size %d x %d\n', m, n);
fprintf('True rank: %d\n', r);
fprintf('Sparsity of X: %.2f%%\n', 100 * (nnz(X_true) / numel(X_true)));
fprintf('Sparsity of Y: %.2f%%\n', 100 * (nnz(Y_true) / numel(Y_true)));
fprintf('Signal-to-noise ratio: %.2f dB\n', 20*log10(norm(M_clean,'fro')/norm(noise,'fro')));
end