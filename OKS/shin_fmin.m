function Cw = shin_fmin(w,spkTms)
%% INPUTS

% w: supposed kernel bandwidth size, in seconds
% spkTms: vector of spike times, in seconds

%% OUTPUTS

% Cw: cost function evaluation for OKS method

tbin = 0.00025; % Time bin, in seconds
N = length(spkTms); % Number of spike times
startTime=spkTms(1); % Start of experiment, in seconds
endTime=spkTms(end); % End of experiment, in seconds

binSpks=binSpikes(spkTms,tbin,startTime,endTime); % bin the spikes

k_radius = ceil(8 * w / tbin); % Set kernel threshold/truncation length
t_kernel = (-k_radius:k_radius) * tbin; % Evaluate time length of kernel with threshold

K_vec = (1/(sqrt(2*pi)*w)) * exp(-t_kernel.^2 / (2*w^2));
Psi_vec = (1/(sqrt(4*pi)*w)) * exp(-t_kernel.^2 / (4*w^2));

conv_Psi = conv(binSpks, Psi_vec, 'same'); % Convolve spike train with cross-term
term1 = sum(binSpks .* conv_Psi); % Sum cross-term convolution

conv_K = conv(binSpks, K_vec, 'same'); % Convolve spike train with kernel function
sum_K_all = sum(binSpks .* conv_K); % sum kernel function convolution

K_0 = 1/(sqrt(2*pi)*w);
term2 = sum_K_all - (N * K_0);

Cw = term1 - 2*term2; % Cost function evaluation
end
