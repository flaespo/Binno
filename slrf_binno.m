function [X, Y, Psi1, Psi2] = slrf_binno(M, r, itermax, lambda1, lambda2, gamma1, gamma2)
[m, n] = size(M);
X      = rand(m, r); % random Xini
Y      = rand(r, n); % random Yini

Psi1   = zeros(itermax,1);
Psi2   = zeros(itermax,1);
tol    = 1e-6;

for k = 1:itermax
  % Store previous values for convergence check
  X_prev = X;
  Y_prev = Y;
  
  % Y*Y'
  YYt = Y*Y';

  % Compute L1 for nu_X bound (using previous Y)
  L1 = norm(YYt, 2);
  
  % Compute nu_X_min according to Theorem 2
  %nu_X = 1 /(sqrt((L1 + gamma1) * (L1 + lambda1 * sqrt(m * r))) - L1);
  nu_X_theory = 1 /(sqrt((L1 + gamma1) * (L1 + lambda1 * sqrt(m * r))) - L1);
  nu_X        = min(nu_X_theory, 1 / L1);  %%%NON HO CAPITO A CHE SERVE

  % Compute gradient for X
  gradH_X = X*YYt-M*Y';
  
  % X-update: proximal steps
  Xu = prox_l1(X - nu_X * gradH_X, nu_X * lambda1);
  Xl = prox_nuclear(X - nu_X * gradH_X, nu_X * gamma1);
  
  % Compute alpha range according to Theorem 2
  alpha_min = (1/nu_X + L1) / (lambda1 * sqrt(m * r) + 1/nu_X + 2 * L1);
  alpha_max = (gamma1 + L1) / (gamma1 + 1/nu_X + 2 * L1);
  alpha = (alpha_min + alpha_max) / 2; % Choose midpoint: è una scelta, ok

  % Convex combination for X
  X = alpha * Xu + (1 - alpha) * Xl;
  
  % Compute L2 matrices for nu_Y bound (using updated Xu and Xl)
  XutXu = Xu' * Xu;
  XltXl = Xl' * Xl;

  L2_Xu = norm(XutXu, 2);
  L2_Xl = norm(XltXl, 2);

  L2_max = max(L2_Xu, L2_Xl);

  % Compute nu_Y_min according to Theorem 3
  A = L2_Xl + L2_Xu;
  B = lambda2 * gamma2 * sqrt(r * n) + gamma2 * L2_Xu + L2_Xl * lambda2 * sqrt(r * n);
  %nu_Y = 2 / (sqrt(A^2 + 4 * B) - A);
  nu_Y_theory = 2 / (sqrt(A^2 + 4 * B) - A);
  nu_Y = min(nu_Y_theory, 1 / L2_max);

  % Compute gradients for Y
  gradH_Yu = XutXu * Y - Xu'*M; % Using Xu and previous Y
  gradH_Yl = XltXl * Y - Xl'*M; % Using Xl and previous Y
  
  % Y-update: proximal steps
  Yu = prox_l1(Y - nu_Y * gradH_Yu, nu_Y * lambda2);
  Yl = prox_nuclear(Y - nu_Y * gradH_Yl, nu_Y * gamma2);
  
  % Compute beta range according to Theorem 3
  beta_min = (1/nu_Y + L2_Xu) / (lambda2 * sqrt(r * n) + 1/nu_Y + 2 * L2_Xu);
  beta_max = (gamma2 + L2_Xl) / (gamma2 + 1/nu_Y + 2 * L2_Xl);
  beta = (beta_min + beta_max) / 2; % Choose midpoint
  
  % Convex combination for Y
  Y = beta * Yu + (1 - beta) * Yl;
  
  % Display progress every 200 iterations
  if mod(k, 200) == 0
    k
  end

 % Function values
 H_fun   = 0.5*norm(X*Y - M,'fro')^2;
 Psi1(k) = lambda1*sum(abs(X(:))) + lambda2*sum(abs(Y(:))) + H_fun;
 Psi2(k) = gamma1*sum(svd(X,'econ')) + gamma2*sum(svd(Y,'econ')) + H_fun;
end
end
%% Proximal operator for nuclear norm (singular value thresholding)
function X = prox_nuclear(X, threshold)
 [U, S, V] = svd(X, 'econ');
 S_thresh  = diag(max(diag(S) - threshold, 0));
 X         = U * S_thresh * V';
end
%% Proximal operator for l1 norm (soft thresholding)
function X = prox_l1(X, threshold)
 X         = sign(X) .* max(abs(X) - threshold, 0);
end