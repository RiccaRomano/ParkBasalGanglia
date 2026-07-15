function [save_predtheta,save_predvar,save_updtheta,save_updvar,smooth_theta,smooth_var,save_gain,total_logL]=get_Kalman_smooth(isi,init_theta,init_var,gl,gk)
%% INPUTS:

% isi: column vector of interspike intervals, in seconds
% init_theta: 2-by-1 vector of initial state values (lambda, kappa)
% init_var: 2-by-2 matrix of initial state variances, covariances (lambda, kappa)
% gl, gk: hyperparameter estimates


%% OUTPUTS:

% save_predtheta: n_isi-by-2 matrix, saves predicted states from Kalman filtering
% save_predvar: n_isi-by-1 cell, saves predicted variances from Kalman filtering
% save_updtheta: n_isi-by-2 matrix, saves updated states from Kalman filtering
% save_updvar: n_isi-1 cell, saves updated variances from Kalman filtering
% smooth_theta: n_isi-by-2 vector, fixed-interval smoothed estimates for lambda and kappa
% smooth_var: n_isi-1 cell, fixed-interval smoothed variances, covariances of lambda and kappa
% save_gain: cell, saves gain matrix during smoothing
% total_logL: log-likelihood of observation model, summation of probability of ISIs given states and probability of updated state conditioned on the predicted state
    % this log-likelihood is implemented as a proxy in a optimization framework if EM algorithm cannot converge



% 1. Forward: Kalman Filtering

n_isi = length(isi); % number of ISIs
Tinit = isi(1); % First ISI

% Perform initial state update using the first ISI And prior estimates via Newton-Raphson
[update_theta, update_var] = update_Kalman(init_theta, init_var, Tinit); % Update states at first ISI, based on initial estimate

% Preallocate memory for forward filtering predictions
save_predtheta = zeros(n_isi, 2);
save_predvar = cell(n_isi, 1);
save_predtheta(1, :) = init_theta';
save_predvar{1} = init_var;

% Preallocate memory for forward filtering updates
save_updtheta = zeros(n_isi, 2);
save_updvar = cell(n_isi, 1);
save_updtheta(1, :) = update_theta';
save_updvar{1} = update_var;

% Initialize log-likelihood summation
total_logL=0;

for jj = 1:n_isi-1
    Tprev = isi(jj); % Elapsed time during the previous interspike interval
    Tnew = isi(jj+1); % Elapse time during the current interspike interval
    Q = Tprev * diag([gl^2; gk^2]); % Noise matrix Q, assumes random walk
    
    % Predict current state and covariance forward in time
    [pred_theta, pred_var] = pred_Kalman(update_theta, update_var, Q);
    save_predtheta(jj+1, :) = pred_theta';
    save_predvar{jj+1} = pred_var;
    
    % Update the predicted states using the observation of the current ISI
    [update_theta, update_var] = update_Kalman(pred_theta, pred_var, Tnew);
    save_updtheta(jj+1, :) = update_theta';
    save_updvar{jj+1} = update_var;
    
    % Extract updated states: firing rate (lambda) and irregularity (kappa)
    upd_lambda=update_theta(1);
    upd_kappa=update_theta(2);
    
    % Log-likelihood of observation under Gamma ISI density
    term1 = upd_kappa * log(upd_lambda) + upd_kappa * log(upd_kappa);
    term2 = -gammaln(upd_kappa);
    term3 = (upd_kappa - 1) * log(Tnew);
    term4 = -upd_lambda * upd_kappa * Tnew;
    log_obs_lik = term1 + term2 + term3 + term4;
    
    % Mahalanobis distance penalizing deviation between updated and predicted states
    state_diff = update_theta - pred_theta;
    prior_penalty = 0.5 * (state_diff' * (pred_var \ state_diff));
    det_ratio = 0.5 * real(log(det(update_var) / det(pred_var)));
    
    % Accumulate log-likelihood for hyperparameter optimization
    total_logL = total_logL + log_obs_lik - prior_penalty + det_ratio;
end


% 2. Backward: Fixed-Interval Smoothing
% Preallocate memory for smoothed estimates
smooth_theta = zeros(n_isi, 2);
smooth_var = cell(n_isi, 1);
save_gain = cell(n_isi - 1, 1);

% Initialize backward smoother with last updated state values (last interspike interval)
smooth_theta(end, :) = save_updtheta(end, :); 
smooth_var{end} = save_updvar{end};

% Step backward in time from second-to-last observation to the first
for jj = n_isi-1:-1:1
    % Calculate gain matrix
    A_temp = save_updvar{jj} / (save_predvar{jj+1} + 1e-6*eye(2));
    save_gain{jj} = A_temp;
    
    % Extract smoothed states
    curr_upd = save_updtheta(jj, :)';
    next_smooth = smooth_theta(jj+1, :)';
    next_pred = save_predtheta(jj+1, :)';
    
    raw_smooth = curr_upd + A_temp * (next_smooth - next_pred);
    
    % Numerical safeguard: enforce positivity for rate and irregularity states
    min_smooth_val = 1e-4;
    raw_smooth(1) = max(raw_smooth(1), min_smooth_val);
    raw_smooth(2) = max(raw_smooth(2), min_smooth_val);
    smooth_theta(jj, :) = raw_smooth';
    
    % Extract covariance matrices
    P_curr_upd = save_updvar{jj};
    P_next_smooth = smooth_var{jj+1};
    P_next_pred = save_predvar{jj+1};
    
    % Calculate smoothed covariance estimate
    P_smooth = P_curr_upd + A_temp * (P_next_smooth - P_next_pred) * A_temp';
    
    % Enforce symmetry of covariance matrix
    smooth_var{jj} = (P_smooth + P_smooth') / 2; 
end
end