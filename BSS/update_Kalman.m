function [update_theta, update_var] = update_Kalman(pred_vec, V, T)
%% INPUTS:

% pred_vec: 2-by-1 predicted state vector (lambda, kappa)
% V: 2-by-2 predicted state covariance matrix
% T: Current observed interspike interval (ISI), in seconds

%% OUTPUTS:

% update_theta: 2-by-1 updated state vector (lambda, kappa)
% update_var: 2-by-2 updated state covariance matrix

inv_V = inv(V); % Inverse of predicted covariance matrix

% Ensure strict positivity constraint on predicted states (lambda, kappa)
current_theta = max(pred_vec, 1e-4);

% Optimization constraints
max_iter = 100;
tol_grad = 1e-5; 
tol_step = 1e-5; 
max_step = 5.0;

% Newton-Raphson optimization loop
for iter = 1:max_iter

    % Compute gradient and Hessian of log-posterior of current state
    [grad, H] = get_grad_hess(current_theta, inv_V, pred_vec, T);
    
    % Hessian regularization, ensures numerical stability
    H = H - 1e-6 * eye(size(H)); 

    % Stopping criteria, if gradient is below tolerance
    if norm(grad) < tol_grad
        break;
    end
    
    % Calculate raw Newton step
    delta = -H \ grad;
    
    % Capping the step to prevent overshooting
    if norm(delta) > max_step
        delta = (delta / norm(delta)) * max_step;
    end
    
    % Backtrack line search
    step_scale = 1.0;
    proposed_theta = current_theta + step_scale * delta;
    
    % Check that boundaries are enforced
    while any(proposed_theta <= 1e-4) && step_scale > 1e-12
        step_scale = step_scale * 0.5;
        proposed_theta = current_theta + step_scale * delta;
    end
    
    % If backtracking reached precision limits, enforce positivity
    if any(proposed_theta <= 1e-4)
        proposed_theta = max(proposed_theta, 1e-4);
    end
    
    % Gradient norm reduction, ensure the next step produces a flatter gradient
    [prop_grad, ~] = get_grad_hess(proposed_theta, inv_V, pred_vec, T);
    while norm(prop_grad) > norm(grad) && step_scale > 1e-12
        step_scale = step_scale * 0.5;
        proposed_theta = current_theta + step_scale * delta;
        proposed_theta = max(proposed_theta, 1e-4); 
        [prop_grad, ~] = get_grad_hess(proposed_theta, inv_V, pred_vec, T);
    end

    % Define current state updates
    current_theta = proposed_theta;
    
    if norm(step_scale * delta) < tol_step
        break; 
    end
end

% Finalize updated state estimate
update_theta = current_theta;

% Laplace Approximation for covariance matrix
[~, H_final] = get_grad_hess(update_theta, inv_V, pred_vec, T);

% Ensure Hessian symmetry
H_final = (H_final + H_final') / 2;
Precision_Matrix = -H_final;

% Ridge regression for increased numerical stability
ridge_epsilon = 0; 
Precision_Matrix = Precision_Matrix + ridge_epsilon * eye(2);

% Updated variance: inverse of Precision matrix, also negative inverse of the Hessian (Laplace approximation)
update_var = inv(Precision_Matrix);
update_var = (update_var + update_var') / 2;

% Covariance Lower Bounding
if update_var(1,1) <= 1e-6 || update_var(2,2) <= 1e-6
    update_var(1,1) = max(update_var(1,1), 1e-6);
    update_var(2,2) = max(update_var(2,2), 1e-6);

    update_var(1,2) = 0; 
    update_var(2,1) = 0; 
end

% Cap state variances, can lead to blow-up and non-invertibility of covariance matrix
max_var_lambda = 1e4; 
max_var_kappa  = 1e2;

if update_var(1,1) > max_var_lambda || update_var(2,2) > max_var_kappa
    v1 = min(update_var(1,1), max_var_lambda);
    v2 = min(update_var(2,2), max_var_kappa);

    corr_coef = update_var(1,2) / sqrt(update_var(1,1) * update_var(2,2));
    cov12 = corr_coef * sqrt(v1 * v2);

    update_var = [v1, cov12; cov12, v2];
end
end