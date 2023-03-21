%% categorical (3-category), Poisson, Gaussian mixed graphical model
p1 = 10;
p2 = 10; 
p3 = 10; 
p = p1 + p2 + p3; 
time = 0:0.01:1; 
K = length(time);  % 101 observed time points
n = repmat(20, K, 1); 
m = [repmat(2, 1, p1), ones(1, p2 + p3)]; 
w = sqrt(m'*m); 
l = [repmat(1/12, 1, p1*2), repmat(1/12, 1, p2), zeros(1, p3)]; 

%% generate data
nodetype = [repmat('d', 1, p1), repmat('t', 1, p2), repmat('g', 1, p3)]; 
thin = 10; 
warmup = 1000; 
seed = 1234; 
load('model.mat'); 
phi = sampling(theta, Theta, K, p, m, n, warmup, thin, seed, nodetype); 


%% 5-fold cv
t = 0:0.2:1; 
h = 0.1:0.1:0.5; 
d = [0.015, 0.03, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4]; 
lambda = 10.^(-1:-0.1:-2.5); 
kernel_type = 'g'; 
options.rho = 1; 

nfold = 5; 
[lambda_opt, d_opt, h_opt] = cv(nfold, t, h, d, phi, lambda, p, l, ...
    time, m, n, w, kernel_type, options); 


%% solution at time point t = 0.2
t = 0.2; 
h = h_opt; 
d = d_opt(2); 
lambda = lambda_opt(2); 

[adjacency, rtime] = local_tvgm(t, h, d, phi, lambda, p, ...
    l, time, m, n, w, kernel_type, options); 

solution = refit(t, adjacency{1}, phi, l, time, m, p, ...
    h, n, kernel_type, options); 



