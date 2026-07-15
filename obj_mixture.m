function best_pars = obj_mixture(spikes)
%% INPUTS

% spikes: vector of spike times, in seconds

%% OUTPUTS

% best_pars: vector of optimal BAKS hyperparameters

vars = [optimizableVariable('a1', [1, 20], 'Transform', 'log')
        optimizableVariable('a2', [1, 20], 'Transform', 'log')
        optimizableVariable('b1', [1, 1e4], 'Transform', 'log')
        optimizableVariable('b2', [1, 1e4], 'Transform', 'log')
        optimizableVariable('p1', [1e-2, 1-1e-2])];

results = bayesopt(@(x) loocv_obj(x, spikes), vars, ...
    'MaxObjectiveEvaluations', 200, ...
    'IsObjectiveDeterministic', true, ...
    'UseParallel', false, ... 
    'AcquisitionFunctionName', 'expected-improvement-plus');
    
best_pars = [results.XAtMinObjective.a1,results.XAtMinObjective.a2,results.XAtMinObjective.b1,results.XAtMinObjective.b2,results.XAtMinObjective.p1];
end

function neg_L = loocv_obj(x, spikes)
%% INPUTS

% x: vector of BAKS hyperparameters
% spikes: vector of spike times, in seconds

%% OUTPUTS

% neg_L: negative log-likelihood of the LOOCV objective function

pars = [x.a1, x.a2, x.b1, x.b2, x.p1];
N = length(spikes);
epsilon = 1e-12;

h_vec=get_BAKS_bandwidths(spikes,pars);
H = h_vec(:)';

D_sq = (spikes - spikes').^2;
Dens = (1 ./ (sqrt(2 * pi) .* H)) .* exp(-D_sq ./ (2 .* H.^2));

Dens(logical(eye(N))) = 0;

loo_rates = sum(Dens, 2) ./ (N - 1);
term1 = sum(log(max(loo_rates, epsilon)));

neg_L=-term1;
end