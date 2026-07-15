function [Z,KS_stat] = get_Poisson(Time, rate, spkTms)
%% INPUTS

% Time: time vector
% rate: vector, contains BAKS firing rate, in Hz
% spkTms: vector of spike times, in seconds

%% OUTPUTS

% Z: CDF of Rescaled ISIs, assuming exponential dist.
% KS_stat: Kolmogorov-Smirnov statistic

C_full = cumtrapz(Time, rate);
C_spikes = interp1(Time, C_full, spkTms, 'linear');

if any(isnan(C_spikes))
    warning('Some spkTms fall outside the Time vector. Dropping them for TRT.');
    C_spikes = C_spikes(~isnan(C_spikes));
end

Lambda = diff(C_spikes);
Lambda = max(Lambda, 1e-12);

Z = 1 - exp(-Lambda);
Z_sorted = sort(Z);

N_intervals = length(Lambda);
uniform_CDF = ((1:N_intervals) - 0.5)' ./ N_intervals;
KS_bound = 1.36 / sqrt(N_intervals);
KS_stat = max(abs(Z_sorted - uniform_CDF));
end