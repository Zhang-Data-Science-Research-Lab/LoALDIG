function [phi, X] = sampling(theta, Theta, K, p, m, n, warmup, ...
    thin, seed, nodetype)
% Gibbs sampling for mixed graphical models
% INPUTS
% theta: 1*K cell array. Each cell is sum(m)*1
% Theta: 1*K cell array. Each cell is sum(m)*sum(m). 
% nodetype: 'd' for categorical, 'g' for gaussian, 'p' for poisson, 
%           't' for truncated poission at 10, 'e' for exponential
% warmup: # steps for burn in
% thin: thinning to reduce autocorrelation
%
% OUTPUTS
% phi: 1*K cell array of unstandardized sufficient statistics. Each cell
%      is n(k)*sum(m) matrix. 

M = sum(m); 
phi = cell(1, K); 
X = cell(1, K); 
rng(seed); 

for k = 1:K
    N = warmup + 1 + thin*(n(k)-1); 
    suff = zeros(M, N); 
    raw = zeros(p, N); 
    for j = 2:N
        % first feature
        natu_par = theta{k}(1:m(1));
        natu_par = natu_par + 2*Theta{k}(1:m(1),(m(1)+1):M)*suff((m(1)+1):M,j-1);
        if nodetype(1) == 'd'
            prob = zeros(1, m(1)+1);
            prob(1) = 1/(1+sum(exp(natu_par))); 
            for t = 1:m(1)
                prob(t+1) = exp(natu_par(t))/(1+sum(exp(natu_par))); 
            end
            cat = randsample(0:m(1), 1, true, prob);
            for t = 1:m(1)
                suff(t,j) = cat == t; 
            end
            raw(1,j) = cat; 
        end
        
        if nodetype(1) == 'g'
            v = 1/(-2*Theta{k}(1,1));
            mu = natu_par*v;
            suff(1,j) = randn(1)*sqrt(v)+mu;
            raw(1,j) = suff(1,j); 
        end
        
        if nodetype(1) == 't'  % truncated poisson, truncated at 10
            prob = exp(natu_par*(0:10)-log(factorial(0:10)));
            prob = prob / sum(prob);
            suff(1,j) = randsample(0:10, 1, true, prob); 
            raw(1,j) = suff(1,j); 
        end
        
        if nodetype(1) == 'e'
            rate = -natu_par;
            suff(1,j) = random('exp', 1/rate); 
            raw(1,j) = suff(1,j); 
        end
        
        % feature 2~(p-1)
        for r = 2:(p-1)
            [r_lower, r_upper] = getindex(m, r);
            natu_par = theta{k}(r_lower:r_upper);
            natu_par = natu_par + 2*Theta{k}(r_lower:r_upper,1:(r_lower-1))*suff(1:(r_lower-1),j);
            natu_par = natu_par + 2*Theta{k}(r_lower:r_upper,(r_upper+1):M)*suff((r_upper+1):M,j-1);
            if nodetype(r) == 'd'
                prob = zeros(1, m(1)+1);
                prob(1) = 1/(1+sum(exp(natu_par))); 
                for t = 1:m(r)
                    prob(t+1) = exp(natu_par(t))/(1+sum(exp(natu_par))); 
                end
                cat = randsample(0:m(r), 1, true, prob);
                for t = 1:m(r)
                    suff(t+r_lower-1,j) = cat == t; 
                end
                raw(r,j) = cat; 
            end
            
            if nodetype(r) == 'g'
                v = 1/(-2*Theta{k}(r_lower,r_lower));
                mu = natu_par*v;
                suff(r_lower,j) = randn(1)*sqrt(v)+mu; 
                raw(r,j) = suff(r_lower,j); 
            end
        
            if nodetype(r) == 't'  % truncated poisson, truncated at 10
                prob = exp(natu_par*(0:10)-log(factorial(0:10)));
                prob = prob / sum(prob);
                suff(r_lower,j) = randsample(0:10, 1, true, prob); 
                raw(r,j) = suff(r_lower,j); 
            end
        
            if nodetype(r) == 'e'
                rate = -natu_par;
                suff(r_lower,j) = random('exp', 1/rate); 
                raw(r,j) = suff(r_lower,j); 
            end
        end
        
        % the last feature
        [r_lower, r_upper] = getindex(m, p);
        natu_par = theta{k}(r_lower:r_upper);
        natu_par = natu_par + 2*Theta{k}(r_lower:r_upper,1:(r_lower-1))*suff(1:(r_lower-1),j);
        if nodetype(p) == 'd'
            prob = zeros(1, m(1)+1);
            prob(1) = 1/(1+sum(exp(natu_par))); 
            for t = 1:m(r)
                prob(t+1) = exp(natu_par(t))/(1+sum(exp(natu_par))); 
            end
            cat = randsample(0:m(p), 1, true, prob);
            for t = 1:m(p)
                suff(t+r_lower-1,j) = cat == t; 
            end
            raw(p,j) = cat; 
        end
            
        if nodetype(p) == 'g'
            v = 1/(-2*Theta{k}(r_lower,r_lower));
            mu = natu_par*v;
            suff(r_lower,j) = randn(1)*sqrt(v)+mu; 
            raw(p,j) = suff(r_lower,j); 
        end
        
        if nodetype(p) == 't'  % truncated poisson, truncated at 10
            prob = exp(natu_par*(0:10)-log(factorial(0:10)));
            prob = prob / sum(prob);
            suff(r_lower,j) = randsample(0:10, 1, true, prob); 
            raw(p,j) = suff(r_lower,j); 
        end
        
        if nodetype(p) == 'e'
            rate = -natu_par;
            suff(r_lower,j) = random('exp', 1/rate); 
            raw(p,j) = suff(r_lower,j); 
        end  
    end
    % thinning
    suff = suff';
    phi{k} = suff((warmup+1):thin:N,:); 
    raw = raw'; 
    X{k} = raw((warmup+1):thin:N,:); 
end


