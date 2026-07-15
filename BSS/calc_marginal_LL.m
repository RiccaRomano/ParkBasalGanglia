function neg_LL = calc_marginal_LL(params, isi, init_theta, init_var,ub,lb)
%% INPUTS

% params: 2-by-1 vector, hyperparameters gamma_lambda and gamma_kappa
% isi: interspike intervals, in seconds
% init_theta: 2-by-1 vector with initial values for lambda and kappa
% init_var: 2-by-2 covariance matrix (lambda, kappa)
% ub: Hyperparameter upper bounds
% lb: Hyperparameter lower bounds

%% OUTPUTS

% neg_LL: negative log-likelihood of the observation model

% Extract hyperparameter values
gl = params(1);
gk = params(2);

% Enforce strict bounds
if gl < lb(1) || gk < lb(2) || gl > ub(1) || gk > ub(2)
    neg_LL = Inf;
    return;
end

% Number of interspike intervals
n_isi = length(isi);

% Obtain predicted and updated Kalman filter state estimates, and log-likelihood of observation model
[~,save_predvar,save_updtheta,save_updvar,~,~,~,total_logL]=get_Kalman_smooth(isi,init_theta,init_var,gl,gk);

for jj = 1:n_isi
    % Extract updated states
    upd_lambda = save_updtheta(jj,1);
    upd_kappa  = save_updtheta(jj,2);
    
    % Extract covariance matrices
    upd_var = save_updvar{jj};
    pred_var = save_predvar{jj};
    
    % Enforce positivity constraints on the firing rate and irregularity
    if upd_lambda <= 1e-5 || upd_kappa <= 1e-5 || ~isreal(upd_lambda) || ~isreal(upd_kappa)
        neg_LL = 1e6; 
        return;
    end
    
    % Enforce positive covariance matrix determinants
    if det(upd_var) <= 0 || det(pred_var) <= 0
        neg_LL = 1e6;
        return;
    end
end

% Negative log-likelihood, trying to minimize this!
neg_LL = -total_logL;
end