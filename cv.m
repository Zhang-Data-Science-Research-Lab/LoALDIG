function [lambda_opt, d_opt, h_opt, cv_err_1, cv_err_2, cv_err_3, rtime] = cv(nfold, ...
    t, h, d, phi, lambda, p, l, time, m, n, w, kernel_type, options)
% V-fold cross validation
% INPUTS
% nfold: number of folds for cv
% t: a sequence of time points to be estimated
% h, d, lambda: candidate hyperparameters
%
% OUTPUTS
% lambda_opt, d_opt: vectors of the same length as t
% h_opt: a scalar

t1 = clock; 
%% Tune lambda specifically. One lambda for each estimated time point. 
fprintf('================================\n'); 
fprintf(' Tune lambda \n'); 
fprintf('--------------------------------\n'); 
cv_err = zeros(length(t), length(lambda), nfold); 
for fold = 1:nfold
    % construct training data and validation data
    a = fold:nfold:length(time); 
    b = setdiff(1:length(time), a); 
    t_phi = phi(b); 
    v_phi = phi(a); 
    t_time = time(b); 
    v_time = time(a); 
    t_n = n(b); 
    v_n = n(a); 
    
    for i = 1:length(t)
        fprintf('fold = %.0f, t = %.4f \n', fold, t(i))
        % fit model to get graph topology using t_phi
        adjacency = local_tvgm(t(i), median(h), median(d), t_phi, ...
            lambda, p, l, t_time, m, t_n, w, kernel_type, options); 
        % refit model to eliminate over-shrinkage
        for j = 1:length(lambda)
            [solution, t_mu, t_sigma] = refit(t(i), adjacency{j}, ...
                t_phi, l, t_time, m, p, median(h), t_n, ...
                kernel_type, options); 
            % standardize v_phi and l
            s_phi = v_phi; 
            s_l = l; 
            for r = 1:sum(m)
                for k = 1:length(v_time)
                    s_phi{k}(:,r) = (v_phi{k}(:,r) - t_mu(r)) ./ t_sigma(r); 
                end
            end
            for r = 1:p
                [r_lower, r_upper] = getindex(m, r); 
                s_l(r_lower:r_upper) = l(r_lower:r_upper) ./ (max(t_sigma(r_lower:r_upper))^2); 
            end
            
            % construct v_Sigma using min(h)
            D = diag(s_l); 
            weight = zeros(length(v_time), 1); 
            for k = 1:length(v_time)
                weight(k) = v_n(k)*kernel(kernel_type, min(h), v_time(k), t(i)); 
            end
            weight = weight ./ sum(weight); 
            H = zeros(sum(m), sum(m)); 
            for k = 1:length(v_time)
                H = H + weight(k) .* (s_phi{k}'*s_phi{k}./v_n(k)); 
            end
            H = H + D; 
            mu0 = zeros(sum(m), 1); 
            for k = 1:length(v_time)
                mu0 = mu0 + weight(k) .* (s_phi{k}'*ones(v_n(k),1)./v_n(k)); 
            end
            v_Sigma = [1, mu0'; mu0, H];    % sum(m)+1 by sum(m)+1 matrix
            
            % calculate cv error
            cv_err(i,j,fold) = trace(solution*v_Sigma) - log(det(solution)); 
        end
    end 
end
cv_err_1 = mean(cv_err, 3); 
[~, ind] = min(cv_err_1, [], 2); 
lambda_opt = lambda(ind); 

%% Tune d specifically for each estimated time point. 
fprintf('================================\n'); 
fprintf(' Tune d \n'); 
fprintf('--------------------------------\n'); 
cv_err = zeros(length(t), length(d), nfold); 
for fold = 1:nfold
    % construct training data and validation data
    a = fold:nfold:length(time); 
    b = setdiff(1:length(time), a); 
    t_phi = phi(b); 
    v_phi = phi(a); 
    t_time = time(b); 
    v_time = time(a); 
    t_n = n(b); 
    v_n = n(a); 
    
    for i = 1:length(t)
        for j = 1:length(d)
            fprintf('fold = %.0f, t = %.4f \n', fold, t(i))
            if sum(abs(t_time-t(i)) <= d(j)) > 0 
                % fit model to get graph topology using t_phi
                adjacency = local_tvgm(t(i), median(h), d(j), t_phi, ...
                    lambda_opt(i), p, l, t_time, m, t_n, w, kernel_type, options); 
                % refit model to eliminate over-shrinkage
                [solution, t_mu, t_sigma] = refit(t(i), adjacency{1}, ...
                    t_phi, l, t_time, m, p, median(h), t_n, ...
                    kernel_type, options); 
                % standardize v_phi and l
                s_phi = v_phi; 
                s_l = l; 
                for r = 1:sum(m)
                    for k = 1:length(v_time)
                        s_phi{k}(:,r) = (v_phi{k}(:,r) - t_mu(r)) ./ t_sigma(r); 
                    end
                end
                for r = 1:p
                    [r_lower, r_upper] = getindex(m, r); 
                    s_l(r_lower:r_upper) = l(r_lower:r_upper) ./ (max(t_sigma(r_lower:r_upper))^2); 
                end
            
                % construct v_Sigma using min(h)
                D = diag(s_l); 
                weight = zeros(length(v_time), 1); 
                for k = 1:length(v_time)
                    weight(k) = v_n(k)*kernel(kernel_type, min(h), v_time(k), t(i)); 
                end
                weight = weight ./ sum(weight); 
                H = zeros(sum(m), sum(m)); 
                for k = 1:length(v_time)
                    H = H + weight(k) .* (s_phi{k}'*s_phi{k}./v_n(k)); 
                end
                H = H + D; 
                mu0 = zeros(sum(m), 1); 
                for k = 1:length(v_time)
                    mu0 = mu0 + weight(k) .* (s_phi{k}'*ones(v_n(k),1)./v_n(k)); 
                end
                v_Sigma = [1, mu0'; mu0, H];    % sum(m)+1 by sum(m)+1 matrix
            
                % calculate cv error
                cv_err(i,j,fold) = trace(solution*v_Sigma) - log(det(solution)); 
            else
                cv_err(i,j,fold) = inf; 
            end
            
        end
    end 
end
cv_err_2 = mean(cv_err, 3); 
[~, ind] = min(cv_err_2, [], 2); 
d_opt = d(ind); 

%% Tune a common h for all estimated time points
fprintf('================================\n'); 
fprintf(' Tune h \n'); 
fprintf('--------------------------------\n'); 
cv_err = zeros(length(t), length(h), nfold); 
for fold = 1:nfold
    % construct training data and validation data
    a = fold:nfold:length(time); 
    b = setdiff(1:length(time), a); 
    t_phi = phi(b); 
    v_phi = phi(a); 
    t_time = time(b); 
    v_time = time(a); 
    t_n = n(b); 
    v_n = n(a); 
    
    for i = 1:length(t)
        for j = 1:length(h)
            fprintf('fold = %.0f, t = %.4f \n', fold, t(i))
            % fit model to get graph topology using t_phi
            adjacency = local_tvgm(t(i), h(j), d_opt(i), t_phi, ...
                lambda_opt(i), p, l, t_time, m, t_n, w, kernel_type, options); 
            % refit model to eliminate over-shrinkage
            [solution, t_mu, t_sigma] = refit(t(i), adjacency{1}, ...
                t_phi, l, t_time, m, p, h(j), t_n, ...
                kernel_type, options); 
            % standardize v_phi and l
            s_phi = v_phi; 
            s_l = l; 
            for r = 1:sum(m)
                for k = 1:length(v_time)
                    s_phi{k}(:,r) = (v_phi{k}(:,r) - t_mu(r)) ./ t_sigma(r); 
                end
            end
            for r = 1:p
                [r_lower, r_upper] = getindex(m, r); 
                s_l(r_lower:r_upper) = l(r_lower:r_upper) ./ (max(t_sigma(r_lower:r_upper))^2); 
            end
            
            % construct v_Sigma using min(h)
            D = diag(s_l); 
            weight = zeros(length(v_time), 1); 
            for k = 1:length(v_time)
                weight(k) = v_n(k)*kernel(kernel_type, min(h), v_time(k), t(i)); 
            end
            weight = weight ./ sum(weight); 
            H = zeros(sum(m), sum(m)); 
            for k = 1:length(v_time)
                H = H + weight(k) .* (s_phi{k}'*s_phi{k}./v_n(k)); 
            end
            H = H + D; 
            mu0 = zeros(sum(m), 1); 
            for k = 1:length(v_time)
                mu0 = mu0 + weight(k) .* (s_phi{k}'*ones(v_n(k),1)./v_n(k)); 
            end
            v_Sigma = [1, mu0'; mu0, H];    % sum(m)+1 by sum(m)+1 matrix
            
            % calculate cv error
            cv_err(i,j,fold) = trace(solution*v_Sigma) - log(det(solution)); 
        end
    end 
end

fprintf('--------------------------------\n'); 
t2 = clock; 
rtime = etime(t2, t1); 
cv_err = mean(cv_err, 3); 
cv_err_3 = mean(cv_err, 1); 
[~, ind] = min(cv_err_3); 
h_opt = h(ind); 
end



%% column indices of sufficient statistics of node m
function [lower, upper] = getindex(m, node)
% get the index corresponding to each variable
lower = sum(m(1:node)) - m(node) + 1;
upper = sum(m(1:node));
end

