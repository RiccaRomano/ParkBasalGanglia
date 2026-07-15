function [time_vector, firing_rate_Hz] = gauss_kern(spike_times, h, max_t, tbin)
%% INPUTS

% spike_times: vector of spike times, in seconds
% h: kernel bandwidth size, in seconds
% max_t: duration of experiment, in seconds
% tbin: bin size, in seconds

%% OUTPUTS

% time_vector: vector of time points, in seconds
% firing_rate_Hz: vector of firing rate, in Hz

% Make time vector
time_vector = 0:tbin:max_t;

% Sampling frequency in Hz (1 second / bin size in seconds)
fs = 1 / tbin;

% Discretize spike train into binary vector
edges = [time_vector, time_vector(end) + tbin];
spike_train = histcounts(spike_times, edges);

% Create Gaussian kernel
win_size = round(8 * h / tbin);
t_kernel = -win_size:win_size;

% The ratio (bin_size / sigma) is unitless, so the formula structure remains the same
gauss_kernel = exp(-0.5 * (t_kernel * tbin / h).^2);

% Normalize kernel so the area sums to 1.
% This ensures the total number of spikes is preserved (roughly) after convolution.
gauss_kernel = gauss_kernel / sum(gauss_kernel);

% Convolve spike train with kernel
smoothed_rate = conv(spike_train, gauss_kernel, 'same');

% Convert from "spikes per bin" to "spikes per second" (Hz)
firing_rate_Hz = smoothed_rate * fs;
end