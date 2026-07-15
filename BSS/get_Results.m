% Richard Foster and Cheng Ly
% This code fulfills 3 objectives:
% 1) Uses Expectation-Maximization/Maximum Likelihood Estimation to determine Bayesian hyperparameters
% 2) Calculates the MAP estimate for the rate and irregularity through BVP solution
% 3) Applies the Time-Rescaling Theorem to determine how well an inhomogeneous Gamma renewal process (firing rate given by BSS) fits the observed spike train

clear;
close all;

typename='Cntr'; % Experiment type (healthy, PD)
varname='GPi'; % Neuron type (GPe, GPi, STN)
filename='gpi'; 

% Generate parallel pool to make parameter estimation run faster
user=getenv('USER');
node=getenv('HOSTNAME');
delete(gcp("nocreate"));
parpool('threads');

data=load([filename 'data.mat'],['isi' typename 'SpkT' varname],['isi' typename varname]);
fn=fieldnames(data);
all_spkTms=data.(fn{1}); % Load all spike times, in milliseconds
all_isi=data.(fn{2}); % Load all interspike intervals, in milliseconds

n_exps=length(all_spkTms); % Number of experiments
fs=1e5; % Sampling frequency for firing rate, large quantity needed for integration between successive spike times during TRT statistical testing

% Pre-allocate structure for results, cannot save final rates and irregularity due to large sampling frequency
% Can save Kalman filter estimates and parameters, use function get_MAP.m to re-generate final signals
results = struct('Experiment', {}, 'Init_gl', {}, 'Init_gk', {}, 'Final_gl', {}, ...
                 'Final_gk', {}, 'Info', {}, 'Iterations', {}, 'All_gl', {}, 'All_gk', {}, ...
                 'All_logL', {}, 'Kalman_Lambda', {}, 'Kalman_Kappa', {},'CDF',{},'KS_stat',{});

parfor ii = 1:n_exps
    isi = all_isi{ii} / 1000; % Extract interspike intervals, in seconds
    spkTms = cumsum(isi); % Obtain spike times, in seconds

    res_all=get_Parameters(isi); % Obtain results structure with parameter values and Kalman filter state estimates
    res=res_all(end); % Extract last iteration (optimal iteration)
                                    
    [t_cont, lambda_cont, kappa_cont] = get_MAP(isi, res, fs); % Obtain MAP estimates for rate and irregularity
    
    % Save results in structure
    results(ii).Experiment=ii;
    results(ii).Init_gl=res.init_gl;
    results(ii).Init_gk=res.init_gk;
    results(ii).Final_gl=res.final_gl;
    results(ii).Final_gk=res.final_gk;
    results(ii).Info=res.status;
    results(ii).Iterations=res.iterations;
    if isempty(res.full)
        results(ii).All_gl = [];
        results(ii).All_gk = [];
        results(ii).All_logL = [];
    else
        results(ii).All_gl = res.full(:,1);
        results(ii).All_gk = res.full(:,2);
        results(ii).All_logL = res.full(:,3);
    end
    results(ii).Kalman_Lambda=res.lambda;
    results(ii).Kalman_Kappa=res.kappa;

    [U,  KS_stat] = get_TRT(spkTms, t_cont, lambda_cont, kappa_cont); % Calculate TRT results for MAP estimates, assumes IG renewal process
    
    % Save TRT results in structure
    results(ii).CDF=U;
    results(ii).KS_stat=KS_stat;
end

% Save final results
save([typename varname '_FinalResults.mat'],"results",'-v7.3');
