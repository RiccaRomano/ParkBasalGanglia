% Richard Foster and Cheng Ly
% Code finds the optimal constant bandwidth for a Gaussian smoothing kernel function over a single spike train
% Cost function assumes spike generation folows an inhomogeneous Poisson process
% Code based on work in Shimazaki-Shinomoto 2010 Kernel Bandwidth Optimization

clear;
close all;

typename='Cntr'; % Experiment type (healthy, PD)
varname='STN'; % Neuron type (GPe, GPi, STN)
filename='stn'; % File type (same as varname but lowercase)
foldername=[typename varname];

data=load([filename 'data.mat'],['isi' typename 'SpkT' varname],['isi' typename varname]);
fn=fieldnames(data);
all_spkTms=data.(fn{1}); % Load all spike times (in milliseconds)
all_isi=data.(fn{2}); % Load all interspike intervals

n_exps=length(all_spkTms); % number of experiments in each cell type and Cntr/Prk
n_reps=5; % number of runs to calculate cost function, don't need many, the cost function is well-behaved
% Cost function is non-convex, but the local minima are often global minima as well

rng shuffle
lb_h=eps; % lower bound on acceptable bandwidth values, in seconds
ub_h=100; % upper bound on acceptable bandwidth values, in seconds
h0=rand(n_exps,n_reps)*(ub_h-lb_h)+lb_h; % randomize bandwidth for optimization procedure, in seconds

tbin=0.00025; %time bin size in seconds
options=optimoptions("fmincon",'MaxFunctionEvaluations',5e5,'MaxIterations',5e5,'Display','iter');
    
results = struct('Experiment', {}, 'Time', {}, 'Rate', {}, 'Width', {}, 'Cost', {}, 'ExitFlag', {});
for ii=1:n_exps
    spkTms=all_spkTms{ii}/1000; % Extract spike times from experiment, in seconds
    endTime=spkTms(end); % End of experiment
    
    h_temp=zeros(n_reps,1); % Initialize optimal bandwidth vector
    Cw_temp=zeros(n_reps,1); % Initialize cost function vector

    for jj=1:n_reps
        % Run optimization for bandwidth parameter, cost function is obtained from Shimazaki/Shinomoto 2010 paper
        [h_temp(jj),Cw_temp(jj),exitflag]=fmincon(@(h) shin_fmin(h,spkTms),h0(ii,jj),[],[],[],[],lb_h,ub_h,[],options);
    end
    Cw_opt=min(Cw_temp); % Finds minimum cost function value
    h_opt=h_temp(find(Cw_temp==Cw_opt,1)); % Identify optimal bandwidth associated with minimum cost function value
    
    % Smooth spike train with Gaussian kernel (optimal bandwidth)
    [time_vector, rate_Hz] = gauss_kern(spkTms,h_opt,endTime,tbin);
    
    save_time=time_vector; % Crude decimation of the output signal, too large if use all of it
    save_rate=rate_Hz; % Similar to above
    
    % Save results in data structure
    results(ii).Experiment=ii;
    results(ii).Time=save_time;
    results(ii).Rate=save_rate;
    results(ii).Width=h_opt;
    results(ii).Cost=Cw_opt;
    results(ii).ExitFlag=exitflag;
end

% Save data
save_name=[foldername '_FinalResults.mat'];
save(save_name,'results');





