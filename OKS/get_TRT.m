function [U, KS_stat] = get_TRT(spkTms, t_cont, lambda_cont)
%% INPUTS

% spkTms: vector of spike times, in seconds
% t_cont: vector of time points, in seconds
% lambda_cont: vector of firing rate, in Hz

%% OUTPUTS

% U: CDF of rescaled ISIs
% KS_stat: Kolmogorov-Smirnov statistics, determine maximal distance between empirical and theoretical CDFs

% Spike times and time domain
spkTms = spkTms(:);
t_cont = t_cont(:);
% Calculate cumulative intensity function from the firing rate, lambda_cont
lambda_cont = lambda_cont(:); % Firing rate
cum_lambda = cumtrapz(t_cont, lambda_cont);

% Cumulative intensity calculated at spike times
L_at_spikes = interp1(t_cont, cum_lambda, spkTms, 'linear', 'extrap');

% Calculate rescaled ISIs
X = diff(L_at_spikes);

% Calculate CDF, exponential distribution
U = 1 - exp(-X);

U_valid = U(~isnan(U)); 
n_valid = length(U_valid);

U_sorted = sort(U_valid);
uniform_cdf = ((1:n_valid) - 0.5)' / n_valid;

% Determine how closely distribution matches exponential distribution
KS_stat = max(abs(U_sorted - uniform_cdf));
cb_95 = 1.36 / sqrt(n_valid);
end