function [adjacency, rtime] = local_tvgm(t, h, d, phi, lambda, p, ...
    l, time, m, n, w, kernel_type, options)
% local tvgm via approximate likelihood at time t
% INPUTS
% t: 1*1 scalar. 
% d: neighborhood width
% h: kernel bandwidth
% kernel_type: 'e' for Epanechnikov kernel, 'g' for Gaussian kernel
% time: K-dimentional vector
% lambda: penalty strength. can be a vector. 
% phi: 1*K cell array of unstandardized sufficient statistics. Each cell
%      is n(k)*sum(m) matrix. 
% l: the diagonal used to modify the empirical covariance matrix, 
%    a sum(m)-dimensional vector. For example, 1/12 for categorical and 
%    count variables, 0/12 for continuous variables. 
% m: p-dimensional vector. Dimensions of sufficient statistics. 
% n: K-dimensional vector. sample sizes. 
% w: p*p penalty weight matrix
% options: ADMM control
%   'rho': ADMM penalty parameter, default 2
%   'tol': ADMM relative tolerance, default 5e-4
%   'maxiter': maximum number of iterations, default 999
%   'incr', 'decr': rho adaption, default 1.1
% 
% OUTPUTS
% adjacency: an array of graph adjacency matrices. Length equals the number
%            of lambda. 
% rtime: run time in seconds

%%
[rho, tol, maxiter, incr, decr] = SetOptions(options); 
t1 = clock; 
K = length(time); 
adjacency = cell(1, length(lambda)); 
for i = 1:length(lambda)
    adjacency{i} = zeros(p, p); 
end
% local neighborhood of t
ind = find(abs(time - t) <= d); 
I = length(ind); 

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

%% construct H using kernel smoothing
D = diag(l); 
H = zeros(I, sum(m), sum(m)); 
for i = 1:I
    weight = zeros(K, 1); 
    for k = 1:K
        weight(k) = n(k)*kernel(kernel_type, h, time(k), time(ind(i))); 
    end
    weight = weight ./ sum(weight); 
    H_i = zeros(sum(m), sum(m)); 
    for k = 1:K
        H_i = H_i + weight(k) .* (phi{k}'*phi{k}./n(k)); 
    end
    mu0 = zeros(sum(m), 1); 
    for k = 1:K
        mu0 = mu0 + weight(k) .* (phi{k}'*ones(n(k),1)./n(k)); 
    end
    H(i,:,:) = H_i - mu0*mu0' + D; 
end

%% initialize
ini = zeros(I, sum(m), sum(m)); 
for i = 1:I
    ini(i,:,:) = eye(sum(m)); 
end
V = zeros(I, sum(m), sum(m)); 
for g = 1:length(lambda)
    %% Verify the condition for block diagonal solution
    T = eye(p); 
    for r = 1:(p-1)
        [r_lower, r_upper] = getindex(m, r); 
        for s = (r+1):p
            [s_lower, s_upper] = getindex(m, s); 
            a = 0; 
            for i = 1:I
                H_irs = H(i,r_lower:r_upper,s_lower:s_upper);
                a = a + 1/I*(norm(H_irs(:), 2)^2); 
            end
            if a > lambda(g)^2 * w(r,s)^2
                T(r,s) = 1; 
                T(s,r) = 1; 
            end
        end
    end
    %% Tarjan's graph component searching algorithm
    G = graph(T); 
    conn = conncomp(G); 
    %% solve the problem separately using ADMM
    for i = 1:max(conn)
        m_g = m(conn == i); 
        w_g = w(conn == i, conn == i); 
        p_g = sum(conn == i); 
        index = []; 
        for j = find(conn == i)
            [j_lower, j_upper] = getindex(m, j); 
            index = [index, j_lower:j_upper]; 
        end
        H_g = H(:,index,index); 
        ini_g = ini(:,index,index); 
        if sum(conn == i) > 1
            [V_g, adjacency_g] = ADMM(lambda(g), H_g, p_g, I, m_g, ...
                w_g, rho, tol, maxiter, incr, decr, ini_g); 
            V(:,index,index) = V_g; 
            adjacency{g}(conn == i, conn == i) = adjacency_g; 
        else
            for k = 1:I
                V(k,index,index) = inv(squeeze(H_g(k,:,:))); 
                adjacency{g}(conn == i, conn == i) = 0; 
            end
        end
    end
    ini = V; 
end

%% running time
t2 = clock; 
rtime = etime(t2, t1); 

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



%% ADMM algorithm for each subproblem
function [solution, adjacency] = ADMM(lambda, H, p, I, m, ...
    w, rho, tol, maxiter, incr, decr, ini)
% ADMM algorithm to solve the separated problem on a subset of variables
% INPUTS
% H: input sub-matrix of H
% incr, decr: modify rho
% ini: initial value for warm start

%% initialize
M = sum(m); 
if nargin == 11
    Z = zeros(I, M, M); 
    for i = 1:I
        Z(i,:,:) = eye(M); 
    end
else
    Z = ini; 
end
U = zeros(I, M, M); 
newTheta = Z; 

%% ADMM
for iter = 1:maxiter
    % update Theta
    for i = 1:I
        A = rho*(Z(i,:,:)-U(i,:,:))-1/sqrt(I)*H(i,:,:); 
        [V, D] = eig(squeeze(A)); 
        newTheta(i,:,:) = V*(D + sqrt(D^2+4*rho/sqrt(I)*eye(M)))*V' ./ (2*rho);
    end
    
    % update Z
    newZ = Z;
    for r = 1:p
        [r_lower,r_upper] = getindex(m, r);
        newZ(:,r_lower:r_upper,r_lower:r_upper) = ...
            newTheta(:,r_lower:r_upper,r_lower:r_upper) + ...
            U(:,r_lower:r_upper,r_lower:r_upper); 
    end
    for r = 1:(p-1)
        [r_lower,r_upper] = getindex(m, r);
        for s = (r+1):p
            [s_lower,s_upper] = getindex(m, s);
            bigsoft = double(0);
            for i = 1:I
                A = newTheta(i,r_lower:r_upper,s_lower:s_upper) + ...
                    U(i,r_lower:r_upper,s_lower:s_upper); 
                bigsoft = bigsoft + norm(A(:), 2)^2;
            end
            bigsoft = sqrt(bigsoft); 
            if bigsoft <= lambda*w(r,s)/rho
                newZ(:,r_lower:r_upper,s_lower:s_upper) = 0;
                newZ(:,s_lower:s_upper,r_lower:r_upper) = 0;
            else
                a = 1 - lambda*w(r,s)/rho/bigsoft; 
                for i = 1:I
                    A = newTheta(i,r_lower:r_upper,s_lower:s_upper) + ...
                        U(i,r_lower:r_upper,s_lower:s_upper); 
                    A = squeeze(A); 
                    newZ(i,r_lower:r_upper,s_lower:s_upper) = a*A; 
                    newZ(i,s_lower:s_upper,r_lower:r_upper) = a*A'; 
                end
            end
        end
    end
    
    % dual residual
    g = rho * (Z(:) - newZ(:)); 
    % primal residual
    r = newTheta(:) - newZ(:); 
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

%% get output p*p adjacency matrix
solution = newZ; 
adjacency = zeros(p, p); 
for r = 1:(p-1)
    [r_lower,r_upper] = getindex(m, r); 
    for s = (r+1):p
        [s_lower,s_upper] = getindex(m, s); 
        if any(abs(solution(1,r_lower:r_upper,s_lower:s_upper)) > 1e-7)
            adjacency(r, s) = 1; 
            adjacency(s, r) = 1; 
        end
    end
end

end




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
if all(class(adjacency) == 'cell')
    adjacency = adjacency{1}; 
end

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




