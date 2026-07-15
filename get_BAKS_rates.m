function [rate, h_spks] = get_BAKS_rates(spkTms, Time, pars)
%% INPUTS

% spkTms: vector of spike times, in seconds
% Time: vector of time points, in seconds
% pars: vector of BAKS hyperparameter values

%% OUTPUTS

% rate: vector of BAKS firing rate over the time vector
% h_spks: vector of BAKS bandwidths at spike times

a1 = pars(1); a2 = pars(2); 
b1 = pars(3); b2 = pars(4);
p1 = pars(5); p2 = 1 - p1; 

Time = Time(:);       % Column vector [M x 1]
spks = spkTms(:)';    % Row vector    [1 x N]

M = length(Time);
N = length(spks);

h_spks = zeros(1, N);

inv_b1 = 1 / b1;
inv_b2 = 1 / b2;
c1_num = p1 / (b1^a1);
c1_den = c1_num * (gamma(a1 + 0.5) / gamma(a1));
c2_num = p2 / (b2^a2);
c2_den = c2_num * (gamma(a2 + 0.5) / gamma(a2));

chunk_size_spks = 2000;
for c = 1:chunk_size_spks:N
    idx = c:min(c + chunk_size_spks - 1, N);
    
    sq_dist = ((spks(idx)' - spks).^2) ./ 2; 
    
    base1 = sq_dist + inv_b1;
    num1 = base1 .^ (-a1);
    denom1 = num1 ./ sqrt(base1);
    
    base2 = sq_dist + inv_b2;
    num2 = base2 .^ (-a2);
    denom2 = num2 ./ sqrt(base2);
    
    sumnum   = sum((c1_num .* num1) + (c2_num .* num2), 2);
    sumdenom = sum((c1_den .* denom1) + (c2_den .* denom2), 2);
    
    h_spks(idx) = (sumnum ./ sumdenom)';
end

rate = zeros(M, 1);

dt = Time(2) - Time(1);
T_start = Time(1);

num_std = 8; 

for i = 1:N
    s_i = spks(i);
    h_i = h_spks(i);
    
    t_min = s_i - (num_std * h_i);
    t_max = s_i + (num_std * h_i);

    idx_start = max(1, floor((t_min - T_start) / dt) + 1);
    idx_end   = min(M, ceil((t_max - T_start) / dt) + 1);
    
    if idx_start <= idx_end
        t_win = Time(idx_start:idx_end);
        
        kernel = exp(-((t_win - s_i).^2) ./ (2 * h_i^2)) ./ (sqrt(2*pi) * h_i);
        
        rate(idx_start:idx_end) = rate(idx_start:idx_end) + kernel;
    end
end
end