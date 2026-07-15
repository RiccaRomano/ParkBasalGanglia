function [t_cont, lambda_cont, kappa_cont] = get_MAP(isi, res, fs)

%% INPUTS

% isi: vector of interspike intervals, in seconds
% res: data structure, contains Kalman filter, smoothed estimates for lambda and kappa
% fs: Sampling frequency, must be high-definition

%% OUTPUTS

% t_cont: time vector, in seconds
% lambda_cont: vector, MAP estimate for the firing rate
% kappa_cont: vector, MAP estimate for the irregularity

% Determine if inputs are valid
if isempty(res.lambda) || isempty(res.kappa)
    fprintf('--> Warning: Empty Kalman estimates passed to plotter. Returning empty arrays.\n');
    t_cont = [];
    lambda_cont = [];
    kappa_cont = [];
    return;
end

% Calculate spike times, number of spikes and number of ISIs
spkTms = [0; cumsum(isi(:))];
n_spikes = length(spkTms); 
n_intervals = length(isi); 

% Extract Kalman smoothed states (lambda, kappa)
lam_states = res.lambda(:);
kap_states = res.kappa(:);

% Determine validity of states, must have exactly as many interspike intervals
if length(lam_states) == n_intervals
    lam_states = [lam_states(1);lam_states];
    kap_states = [kap_states(1);kap_states];
elseif length(lam_states) ~= n_spikes
    error('Mismatch: %d ISIs require %d boundary states, but got %d.', n_intervals, n_spikes, length(lam_states));
end

% Initialize outputs
t_cont = [];
lambda_cont = [];

% Calculate MAP states
for i = 1:n_intervals
    dt = spkTms(i+1) - spkTms(i);
    lam_start = lam_states(i);
    lam_end   = lam_states(i+1);
    k_start   = kap_states(i);
    gl        = res.final_gl;
    
    p0 = 0.5 * (lam_start + lam_end) * dt;
    
    C_const = (gl^2 * dt^3) / 12;
    c1 = (C_const * k_start) - p0;
    c2 = -C_const * (k_start - 1);
    
    p_roots = roots([1, c1, c2]);
    valid_idx = (abs(imag(p_roots)) < 1e-8) & (real(p_roots) > 0);
    valid_roots = real(p_roots(valid_idx)); 
    
    if isempty(valid_roots)
        A = 0;
        B = (lam_end - lam_start) / dt; 
    else
        p = max(valid_roots); 
        A = (gl^2 / 2) * (k_start - (k_start - 1) / p);
        B = (lam_end - lam_start) / dt - A * dt;
    end
    
    C_0 = lam_start;
    t_local = 0 : (1/fs) : dt;

    lam_local = A * (t_local.^2) + B * t_local + C_0;
    lam_local = max(lam_local, 1e-4);
    
    if i == 1
        t_cont = [t_cont, t_local + spkTms(i)];
        lambda_cont = [lambda_cont, lam_local];
    else
        t_cont = [t_cont, t_local(2:end) + spkTms(i)];
        lambda_cont = [lambda_cont, lam_local(2:end)];
    end
end

kappa_cont = interp1(spkTms, kap_states, t_cont, 'linear');
kappa_cont = max(kappa_cont, 1e-4); % Guardrail
end