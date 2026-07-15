function res = run_EM_trial(isi,init_theta,init_var,init_gl,init_gk,max_iter)
%% INPUTS:

% isi: vector of interspike intervals, in seconds
% init_theta: 2-by-1 vector with initial values for lambda and kappa
% init_var: 2-by-2 variance-covariance matrix for lambda and kappa, initial values
% init_gl: initial scalar value for hyperparameter gl
% init_gk: initial scalar value for hyperparameter gk
% max_iter: maximum number of iterations for EM algorithm

%% OUTPUTS:

% res: data structure containing the following information:
    % init_gl, init_gk: initial parameter estimates
    % final_gl, final_gk: hyperparameter values obtained by convergence of EM algorithm
    % status: 'Converged', 'Max Iters Reached', or 'Run Failed'
    % iterations: number of iterations needed for EM algorithm convergence
    % logL: log-likelihood of observation model given the hyperparameter estimates
    % full: matrix, first two columns save parameter saves per iteration, last column saves log-likelihood of observation model
    % lambda: vector of Kalman filter estimate for firing rate given the hyperparameter estimates
    % kappa: vector of Kalman filter estimate for irregularity given the hyperparameter estimates

%%
n_isi = length(isi); % Number of ISIs

% Initialize results structures
res.init_gl = init_gl;
res.init_gk = init_gk;
res.final_gl = NaN;
res.final_gk = NaN;
res.status = '';
res.iterations = 0;
res.logL = NaN; 
res.full = [];
res.lambda = [];  
res.kappa = [];

% Initial hyperparameter values
gl = init_gl;
gk = init_gk;

diff_vec = ones(2,1); % Initialize difference between successive points
iter = 0;
save_pars=zeros(max_iter,2); % Initialize hyperparameter save matrix
save_log=zeros(max_iter,1); % Initialize log-likelihood save vector
save_pars(1,:)=[init_gl init_gk]; % Input initialized hyperparameter values

try
    while max(abs(diff_vec)) > 1e-4 && iter < max_iter % Run EM algorithm until the successive hyperparameter iterates reach below threshold
        iter = iter + 1;
        
        % Obtain fixed-interval smoothed states and variances
        [~,~,~,~,smooth_theta,smooth_var,save_gain,total_logL]=get_Kalman_smooth(isi,init_theta,init_var,gl,gk);
        
        % Initialize hyperparameter update steps
        save_covar = cell(n_isi-1, 1);
        exp_lambda = zeros(n_isi-1, 1);
        exp_kappa = zeros(n_isi-1, 1);
        
        % Calculate successive state expectation, needed for hyperparameter iterate update
        for jj = 1:n_isi-1
            save_covar{jj} = save_gain{jj} * smooth_var{jj+1};
            
            exp_lambda(jj) = smooth_var{jj+1}(1,1) - 2*save_covar{jj}(1,1) + ...
                             smooth_var{jj}(1,1) + (smooth_theta(jj+1,1) - smooth_theta(jj,1))^2;
                             
            exp_kappa(jj) = smooth_var{jj+1}(2,2) - 2*save_covar{jj}(2,2) + ...
                            smooth_var{jj}(2,2) + (smooth_theta(jj+1,2) - smooth_theta(jj,2))^2;
        end
        
        % Calculate next hyperparameter iterates
        gl_next = sqrt( (1/(n_isi-1)) * sum(exp_lambda ./ isi(1:end-1)) );
        gk_next = sqrt( (1/(n_isi-1)) * sum(exp_kappa ./ isi(1:end-1)) );
    
        % Save results
        save_pars(iter+1,:)=[gl_next gk_next];
        save_log(iter)=total_logL;

        if isnan(gl_next) || isnan(gk_next) || ~isreal(gl_next) || ~isreal(gk_next) || gl_next>50 || gk_next>2 || gl_next<0 || gk_next<0
            error('Algorithm diverged: NaN, complex, or large values encountered.');
        end
        
        % Calculate successive hyperparameter iterate difference
        diff_vec = abs([gl - gl_next, gk - gk_next]);
        gl = gl_next; gk = gk_next;
        fprintf('Iter: %f gl is %f and gk is %f, logL is %f \n',iter,gl,gk,total_logL);
        
        % Prevent orbitting around a fixed point, stops EM Algorithm from running endlessly
        window = 500;
        if iter > window
            recent_gl = save_pars(iter-window+1:iter, 1);
            recent_gk = save_pars(iter-window+1:iter, 2);
            recent_logL = save_log(iter-window+1:iter);
            
            orbit_tol = 0.1; 
            
            if (max(recent_gl) - min(recent_gl)) < orbit_tol && ...
               (max(recent_gk) - min(recent_gk)) < orbit_tol
                
                [best_logL, best_idx] = max(recent_logL);
                
                gl = recent_gl(best_idx);
                gk = recent_gk(best_idx);
                total_logL = best_logL;
                
                fprintf('--> Orbit detected! Stopping early. Snapping to best LogL (%.4f) from last %d iters.\n', best_logL, window);
                res.status = 'Converged (Orbit Early Stop)';
                break;
            end
        end
    end
    
    % Save all results
    save_log(iter+1)=total_logL;
    save_pars(iter+2:end,:)=[];
    save_log(iter+2:end)=[];
    save_all=[save_pars save_log];
    
    % Load into data structure array
    res.final_gl = gl;
    res.final_gk = gk;
    res.iterations = iter;
    res.logL=total_logL;
    res.full = save_all;
    res.lambda = smooth_theta(:,1);
    res.kappa = smooth_theta(:,2);
    
    if iter==max_iter
        res.status = 'Max Iters Reached';
        fprintf('--> Stopped: Reached max iterations (%d). LogL: %.4f\n', max_iter, total_logL);
    else
        res.status = 'Converged';
        fprintf('Successfully converged! LogL: %.4f\n', total_logL);
    end
catch ME
    fprintf('--> Run Failed: %s\n', ME.message);
    res.status = ['Failed: ', ME.message];
    res.iterations = iter;
end
end