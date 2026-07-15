function results = get_Parameters(isi)
%% INPUTS:

% isi: vector of interspike intervals, in seconds

%% OUTPUTS:

% results: data structure containing the following information:
    % init_gl, init_gk: initial parameter estimates
    % final_gl, final_gk: hyperparameter values obtained by convergence of EM algorithm
    % status: 'Converged', 'Max Iters Reached', or 'Run Failed'
    % iterations: number of iterations needed for EM algorithm convergence
    % logL: log-likelihood of observation model given the hyperparameter estimates
    % full: matrix, first two columns save parameter saves per iteration, last column saves log-likelihood of observation model
    % lambda: vector of Kalman filter estimate for firing rate given the hyperparameter estimates
    % kappa: vector of Kalman filter estimate for irregularity given the hyperparameter estimates

%% 1. Run EM on Predefined Value

% Hyperparameter test values
gl_test_vals = 1; 
gk_test_vals = 0.1;

% Hyperparameter bounds
ub_vals=[50 2]; % gl gk
lb_vals=[1e-1 1e-2]; % gl gk

% Initial states and variances
init_theta = [1/mean(isi); (mean(isi)/std(isi))^2]; 
init_var = zeros(2);
init_var(1,1) = 1;
init_var(2,2) = 0.1;

% Algorithm constraints, specifications
max_iter = 5e3;  % Maximum number of iterations for the EM algorithm
max_random_attempts = 3; % Keep trying with another 5 randomly generated points (if pre-selected points did not converge)

% Initialize results structure
results = struct('init_gl', {}, 'init_gk', {}, 'final_gl', {}, ...
                 'final_gk', {}, 'status', {}, 'iterations', {}, ...
                 'logL', {}, 'full', {}, 'lambda', {}, 'kappa', {});

run_idx = 1; % Initialize EM run index
valid_run_found = false;

% Run EM algorithm with test points
for i = 1:length(gl_test_vals)
    init_gl = gl_test_vals(i);
    init_gk = gk_test_vals(i);
    res = run_EM_trial(isi, init_theta,init_var,init_gl, init_gk, max_iter);
    results(run_idx) = res;
    if ~isnan(res.logL)
        valid_run_found = true;
    end
    run_idx = run_idx + 1;
end

%% 2. Run EM on Random Points (Only if Predefined did not Converge)
if ~valid_run_found
    fprintf('\n Starting random search... \n');
    random_attempts = 0;
    
    while ~valid_run_found && random_attempts < max_random_attempts
        random_attempts = random_attempts + 1;

        % Generate random starting hyperparameter values
        rand_gl = (ub_vals(1)-lb_vals(1))*rand+lb_vals(1);  
        rand_gk = (ub_vals(2)-lb_vals(2))*rand+lb_vals(2);  
        
        fprintf('\n Random Attempt %d: init_gl = %.4f, init_gk = %.4f\n', random_attempts, rand_gl, rand_gk);
        
        % Run EM algorithm with randomly selected points 
        res = run_EM_trial(isi, init_theta,init_var,rand_gl, rand_gk, max_iter);
        results(run_idx) = res;
        
        if ~isnan(res.logL)
            valid_run_found = true;
            fprintf('--> Success! Valid parameters found via random search.\n');
        else
            fprintf('--> Other attempt failed.\n');
        end
        
        run_idx = run_idx + 1;
    end
end

%% 3. Run fminsearch Fallback (Best of 20 Runs)
if ~valid_run_found
    fprintf('\n Initiating fminsearch fallback over calc_marginal_LL. Running 5 full attempts...\n');
    n_runs = 5; 
    options = optimset('Display', 'off', 'TolX', 1e-5, 'TolFun', 1e-5);
    
    best_logL = -Inf; 
    best_res = [];  
    
    for fmin_attempts = 1:n_runs
        % Generate randomly selected initial hyperparameter values
        rand_init_gl = (ub_vals(1)-lb_vals(1))*rand + lb_vals(1);
        rand_init_gk = (ub_vals(2)-lb_vals(2))*rand + lb_vals(2);
        
        fprintf('fminsearch Attempt %d/%d: init_gl = %.4f, init_gk = %.4f', fmin_attempts, n_runs, rand_init_gl, rand_init_gk);
        
        % Run optimization procedure on proxy function
        objective_func = @(params) calc_marginal_LL(params, isi, init_theta, init_var, ub_vals, lb_vals);
        [optimal_pars, fval, exitflag, output] = fminsearch(objective_func, [rand_init_gl, rand_init_gk], options);
        
        current_logL = -fval;
        
        if exitflag == 1
            fprintf(' --> Converged (logL: %.4f)\n', current_logL);
            
            if current_logL > best_logL
                best_logL = current_logL;
                % Save results
                best_res.init_gl = rand_init_gl;
                best_res.init_gk = rand_init_gk;
                best_res.final_gl = optimal_pars(1);
                best_res.final_gk = optimal_pars(2);
                best_res.iterations = output.iterations;
                best_res.logL = current_logL;
                best_res.full = []; 
                best_res.lambda = [];
                best_res.kappa = [];
                best_res.status = 'Converged (fminsearch)';
            end
        else
            fprintf(' --> Failed (exitflag: %d)\n', exitflag);
        end
    end
    
    if ~isempty(best_res) && best_logL > -Inf
        fprintf('--> Success! Selecting best fminsearch run with overall highest logL: %.4f\n', best_logL);
        valid_run_found = true;
        
        [~,~,~,~,smooth_theta,~]=get_Kalman_smooth(isi, init_theta, init_var, best_res.final_gl, best_res.final_gk);
        best_res.lambda = smooth_theta(:,1);
        best_res.kappa = smooth_theta(:,2);
        
        results(run_idx) = best_res;
    else
        fprintf('--> All %d fminsearch attempts completely failed.\n', n_runs);
        
        res.init_gl = NaN; res.init_gk = NaN; res.final_gl = NaN; res.final_gk = NaN;
        res.status = 'All fminsearch attempts failed'; res.iterations = n_runs;
        res.logL = NaN; res.full = []; res.lambda = []; res.kappa = [];
        results(run_idx) = res;
    end
    run_idx = run_idx + 1;
end
end