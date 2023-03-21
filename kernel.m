function w = kernel(type, h, t1, t2)
% given time points t1, t2, return its kernel weight
u = abs(t1 - t2)/h; 
if type == 'e'
    w = 0.75*(1-u^2)*(u <= 1); 
end
if type == 'g'
    w = 1/sqrt(2*pi)*exp(-0.5*u^2); 
end
