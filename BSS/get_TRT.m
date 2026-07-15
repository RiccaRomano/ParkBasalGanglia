function [U, KS_stat] = get_TRT(spkTms, t_cont, lambda_cont, kappa_cont)
%% INPUTS

% spkTms: vector, contains spike times in seconds
% t_cont: vector, time domain, must have fine-time resolution to allow for high definition integral between spike times
% lambda_cont: vector, MAP firing rate obtained via Bayesian state-space
% kappa_cont: vector, MAP irregularity, obtained via Bayesian state-space

%% OUTPUTS

% U: vector, cumulative distribution function of rescaled ISIs
% KS_stat: scalar, Kolmogorov-Smirnov statistic

% Vectorize inputs
spkTms = spkTms(:);
t_cont = t_cont(:);
lambda_cont = lambda_cont(:);
kappa_cont = kappa_cont(:);

% Cumulative sum of rate and irregularity across the time domain
cum_lambda = cumtrapz(t_cont, lambda_cont);
cum_kappa  = cumtrapz(t_cont, kappa_cont); 

% Interpolate cumulative sum values are spike times
L_at_spikes = interp1(t_cont, cum_lambda, spkTms, 'linear', 'extrap');
K_at_spikes = interp1(t_cont, cum_kappa,  spkTms, 'linear', 'extrap');

% Calculate rescaled ISIs
X = diff(L_at_spikes);

% Calculate interspike intervals and rescaled irregularity
delta_t = diff(spkTms);
k_val = diff(K_at_spikes) ./ delta_t;

% Calculate CDF of rescaled ISIs, Gamma
U = gamcdf(X, k_val, 1 ./ k_val);

U_valid = U(~isnan(U)); 
n_valid = length(U_valid);

% Calculate uniform CDF, linear
U_sorted = sort(U_valid);
uniform_cdf = ((1:n_valid) - 0.5)' / n_valid;

% Calculate KS statistic
KS_stat = max(abs(U_sorted - uniform_cdf));

% Calculate confidence interval, 95%
cb_95 = 1.36 / sqrt(n_valid);
end