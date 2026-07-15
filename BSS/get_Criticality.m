% Richard Foster and Cheng Ly
% Code uses the MAP firing rate estimate and calculates its proximity to criticality

clear;
close all;

typename='Prk'; % Experiment type (healthy, PD)
varname='GPi'; % Neuron type (GPe, GPi, STN)
filename='gpi';

foldername=[typename varname];
data=load([filename 'data.mat'],['isi' typename 'SpkT' varname],['isi' typename varname]);
fn=fieldnames(data);
all_spkTms=data.(fn{1}); % Load all spike times, in milliseconds
all_isi=data.(fn{2}); % Load all interspike intervals, in milliseconds

load([typename varname '_FinalResults.mat']); % Load BSS final results

n_exps=length(all_spkTms); % Number of experiments
b = 2; % Criticality
fit_method = 'YuleWalker'; % AR model fitting algorithm

fs = 25;                  % 25 Hz sampling frequency
deltaT = 1 / fs;          % 0.04 seconds per bin
win_sec = 60.0;           % 60-second windows
step_sec = 60.0;          % 60-second step (0% overlap for safe statistics)
global_order = 5;         % PACF cut off at around 5-10 model order

d2_results = struct('Experiment',{},'d2',{},'Window',{},'Bin',{},'Order',{});

for ii = 1:n_exps
    isi = all_isi{ii} / 1000; % Extract interspike intervals, in seconds

    % Extract Kalman smoothed estimates for lambda and kappa
    res.lambda = results(ii).Kalman_Lambda;
    res.kappa  = results(ii).Kalman_Kappa;

    % Extract hyperparameter values
    res.final_gl = results(ii).Final_gl;
    
    fs_fine = 4000; % Sampling frequency

    % Calculate MAP estimate for firing rate
    [t_fine, lambda_fine, ~] = get_MAP(isi, res, fs_fine);
    
    % Build new time vector with deltaT increments
    t_cont = (t_fine(1) : deltaT : t_fine(end))'; 
    
    % Interpolate lambda_fine over coarser time domain
    lambda_cont = interp1(t_fine, lambda_fine, t_cont, 'pchip');
    
    win_pts = round(win_sec * fs);   
    step_pts = round(step_sec * fs); 
    
    num_total_pts = length(lambda_cont); % Number of points
    num_windows = floor((num_total_pts - win_pts) / step_pts) + 1;
    
    % Initialize criticality proximity values
    db_vals = NaN(num_windows, 1);
    time_centers = NaN(num_windows, 1);
    
    % Calculate criticality proximity on windowed firing rates
    for w = 1:num_windows
        idx_start = (w-1)*step_pts + 1;
        idx_end = idx_start + win_pts - 1;
        
        segment = lambda_cont(idx_start:idx_end);
        segment=segment-mean(segment);
        time_centers(w) = (idx_start + win_pts/2) * deltaT;
        
        [db, sddb, kern, sigma, H, kernc, exit_status] = calc_db(...
            segment, global_order, deltaT, b, ...
            false, false, false, fit_method);
            
        if exit_status == 0
            db_vals(w) = db;
        else
            fprintf('Window %d at t=%.1fs was explosive/unstable. Skipping.\n', w, time_centers(w));
        end
        
    end
    
    % Save results to data structure array
    d2_results(ii).Experiment = ii;
    d2_results(ii).d2 = db_vals;
    d2_results(ii).Window = win_sec;
    d2_results(ii).Bin = deltaT;
    d2_results(ii).Order = global_order;
end

% Save results .mat file
save([typename varname '_d2Results.mat'],'d2_results');
