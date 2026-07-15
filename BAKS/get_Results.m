% Richard Foster and Cheng Ly
% Code that completes 3 objectives:
% 1) In a mixed Gamma BAKS framework, determines the optimal hyperparameters through a leave-one-out cross validation procedure
% 2) Calculates the resulting firing rate, in Hz, based on the optimal hyperparameters
% 3) Uses the Time-Rescaling Theorem to determine how well the resulting firing rate fits to the spike train assuming
% Poisson and Gamma renewal processes govern spike generation

clear;
close all;

typename='Prk'; % Experiment type (healthy, PD)
varname='STN'; % Neuron type (GPe, GPi, STN)
filename='stn';

data=load([filename 'data.mat'],['isi' typename 'SpkT' varname],['isi' typename varname]);
fn=fieldnames(data);
all_spkTms=data.(fn{1}); % Load all spike times, in milliseconds
all_isi=data.(fn{2}); % Load all interspike intervals, in milliseconds

n_exps=length(all_spkTms); % Number of experiments

results = struct('Experiment', {}, 'Parameters', {}, 'CDF_Gamma', {}, 'KS_Gamma', {},'CDF_Pois',{},'KS_Pois',{});

fs=1e5; % Sampling frequency, required for fine-time precision of TRT testing
tbin=1/fs; % Time bin length, in seconds

for ii=1:n_exps
    spkTms=all_spkTms{ii}(1:100)/1000; % Load spike times, in seconds
    opt_pars = obj_mixture(spkTms); % Optimal BAKS hyperparameters
    
    buffer=1; % Add buffer for kernel
    time=(spkTms(1)-buffer):tbin:(spkTms(end)+buffer); % Full time span
    rate=get_BAKS_rates(spkTms,time,opt_pars); % Calculate firing rate, in Hz

    % Get statistics on goodness-of-fit
    [U_Pois,  KS_stat_Pois] = get_Poisson(time,rate,spkTms); % Assumes inhomogeneous Poisson spike generation
    [U_Gamma,  KS_stat_Gamma] = get_Gamma(time,rate,spkTms); % Assumes inhomogeneous Gamma renewal process
    
    results(ii).Experiment=ii; % Experiment ID
    results(ii).Parameters=opt_pars; % Optimal parameters
    results(ii).CDF_Gamma=U_Gamma; % CDF with Gamma renewal
    results(ii).KS_Gamma=KS_stat_Gamma; % Kolmogorov-Smirnov statistic for Gamma renewal assumption
    results(ii).CDF_Pois=U_Pois; % CDF with Poisson
    results(ii).KS_Pois=KS_stat_Pois; % Kolmogorov-Smirnov statistic for Poisson spike generation
end

% Save results
save([typename varname '_FinalResults.mat'],'results');


