function [solution, mu, sigma] = refit(t, adjacency, phi, l, time, m, p, ...
    h, n, kernel_type, options)
% refit the model based on learned sparsity structure at time t, to reduce 
% the over-shrinkage effect
% INPUTS
% adjacency: input estimated adjacency matrix at time t. p*p matrix
% options: ADMM control
%   'rho': ADMM penalty parameter, default 2
%   'tol': ADMM relative tolerance, default 5e-4
%   'maxiter': maximum number of iterations, default 999
%   'incr', 'decr': rho adaption, default 1.1
%
% OUTPUTS
% solution: (sum(m)+1) * (sum(m)+1) matrix output
% mu, sigma: sum(m)-dimensional vectors used for standardization

%%
[rho, tol, maxiter, incr, decr] = SetOptions(options); 
K = length(time); 

%% standardize phi and D
weight = zeros(K, 1); 
for k = 1:K
    weight(k) = n(k)*kernel(kernel_type, h, time(k), t); 
end
weight = weight ./ sum(weight); 
mu = zeros(1, sum(m)); 
sigma2 = ones(1, sum(m)); 
sigma = ones(1, sum(m)); 
for i = 1:p
    [i_lower, i_upper] = getindex(m, i); 
    for j = i_lower:i_upper
        for k = 1:K
            suff = phi{k}(:,j); 
            mu(j) = mu(j) + weight(k) * mean(suff(:)); 
        end
        for k = 1:K
            suff = phi{k}(:,j); 
            sigma2(j) = sigma2(j) + weight(k) * sum((suff(:)-mu(j)).^2) / n(k); 
        end
        sigma(j) = sqrt(sigma2(j)); 
        for k = 1:K
            phi{k}(:,j) = (phi{k}(:,j) - mu(j)) ./ sigma(j); 
        end
    end
    l(i_lower:i_upper) = l(i_lower:i_upper) ./ max(sigma2(i_lower:i_upper)); 
end

%% construct Theta using kernel smoothing
D = diag(l); 
H = zeros(sum(m), sum(m)); 
for k = 1:K
    H = H + weight(k) .* (phi{k}'*phi{k}./n(k)); 
end
H = H + D; 
mu0 = zeros(sum(m), 1); 
for k = 1:K
    mu0 = mu0 + weight(k) .* (phi{k}'*ones(n(k),1)./n(k)); 
end
Sigma = [1, mu0'; mu0, H];    % sum(m)+1 by sum(m)+1 matrix

%% ADMM (sparsity pattern constraint)
M = sum(m); 
Z = eye(M+1); 
U = zeros(M+1, M+1); 
for iter = 1:maxiter
    % update Theta
    A = rho*(Z-U) - Sigma; 
    [V, D] = eig(A); 
    newTheta = V*(D + sqrt(D^2+4*rho*eye(M+1)))*V' ./ (2*rho);
    
    % update Z
    newZ = newTheta + U; 
    for r = 1:(p-1)
        [r_lower,r_upper] = getindex(m, r);
        for s = (r+1):p
            [s_lower,s_upper] = getindex(m, s); 
            if adjacency(r,s) == 0
                newZ((r_lower+1):(r_upper+1), (s_lower+1):(s_upper+1)) = 0; 
                newZ((s_lower+1):(s_upper+1), (r_lower+1):(r_upper+1)) = 0; 
            end
        end
    end
    
    % dual residual
    g = rho*(Z(:)-newZ(:)); 
    % primal residual
    r = newTheta(:) -newZ(:); 
    % update U
    U = U + newTheta - newZ; 
    % check convergence
    epsilon.dual = tol*rho*norm(U(:), 2);
    epsilon.pri = tol*max([norm(newTheta(:), 2), norm(newZ(:), 2)]); 
    if norm(g, 2) <= epsilon.dual && norm(r, 2) <= epsilon.pri
        break
    end
    
    Z = newZ; 
    
    % update rho
    if norm(r, 2)/epsilon.pri > 30*norm(g, 2)/epsilon.dual
        rho = rho*incr;
    end
    if norm(g, 2)/epsilon.dual > 30*norm(r, 2)/epsilon.pri
        rho = rho/decr;
    end
end

solution = newZ; 

end



%% column indices of sufficient statistics of node m
function [lower, upper] = getindex(m, node)
% get the index corresponding to each variable
lower = sum(m(1:node)) - m(node) + 1;
upper = sum(m(1:node));
end



%% Set options
function [rho, tol, maxiter, incr, decr] = SetOptions(options)
rho = 2; 
tol = 5e-4; 
maxiter = 999; 
incr = 1.1; 
decr = 1.1; 

if (isfield(options, 'rho'))
    rho = options.rho; 
end
if (isfield(options, 'tol'))
    tol = options.tol; 
end
if (isfield(options, 'maxiter'))
    maxiter = options.maxiter; 
end
if (isfield(options, 'incr'))
    incr = options.incr; 
end
if (isfield(options, 'decr'))
    decr = options.decr; 
end
end



