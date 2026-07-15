function [pred_theta, pred_var] = pred_Kalman(update_theta, update_var, Q)
%% INPUTS:

% update_theta: 2-by-1 vector, updated state estimates (lambda, kappa)
% update_var: 2-by-2 covariance matrix of updated states (lambda, kappa)
% Q: noise matrix, predicted variance modeled by random walk from updated variance

%% OUTPUTS:

% pred_theta: 2-by-1 vector, predicted states at next ISI are equal to updated states from previous ISI
% pred_var: 2-by-2 covariance, predicted state variances at next ISI

pred_theta = update_theta;
pred_var = update_var + Q;
end
