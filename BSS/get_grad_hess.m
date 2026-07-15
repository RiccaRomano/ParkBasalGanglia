function [grad, Hessian] = get_grad_hess(x, inv_V, pred_vec, T)
%% INPUTS

% x: 2-by-1 vector, estimated states (lambda, kappa) during update step
% inv_V: inverse of covariance matrix during update step
% pred_vec: 2-by-1 vector, predicted states at current time step
% T: Current interspike interval, in seconds

%% OUTPUTS

% grad: 2-by-1 gradient vector, calculated during update step
% Hessian: 2-by-2 Hessian matrix

l = x(1); 
k = x(2);
dL_dl = k/l - k*T;
dL_dk = log(l) + log(k) + 1 - psi(k) + log(T) - l*T;
grad_lik = [dL_dl; dL_dk];
grad_prior = -inv_V * (x - pred_vec);
grad = grad_lik + grad_prior;

d2L_dl2 = -k / l^2;
d2L_dkdl = 1/l - T;
d2L_dk2 = -psi(1, k) + 1/k; 
Hess_Lik = [d2L_dl2, d2L_dkdl; 
            d2L_dkdl, d2L_dk2];
Hess_Prior = -inv_V;
Hessian = Hess_Lik + Hess_Prior;
end

