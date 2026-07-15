% Richard Foster and Cheng Ly
% Code identifies the maximum likelihood estimated fits of the Log-normal distribution family to the interspike intervals

clear;
close all;

foldername='LogNormal'; % Distribution type
typename='Cntr'; % Experiment type (healthy, PD)
varname='STN'; % Neuron type (GPe, GPi, STN)
filename='stn';
totalname='CntrSTN';

mkdir(foldername);

data=load([filename 'data.mat'],['isi' typename 'SpkT' varname],['isi' typename varname]);
fn=fieldnames(data);
all_spkTms=data.(fn{1}); % Load all spike times, in milliseconds
all_isi=data.(fn{2}); % Load all interspike intervals, in milliseconds

n_exps=length(all_isi); % Number of experiments
save_pars=struct('Parameters',{},'ConfIntervals',{},'Mean_ISI',{},'Variance_ISI',{},'CV_ISI',{},...
    'Mean_PDF',{},'Variance_PDF',{},'CV_PDF',{},'AIC',{},'BIC',{},'logL',{},'NumPars',{},'NumObs',{}); % 


nRuns=50; % Number of optimization runs
log_pdf=@(x,mu,sigma) 1./(x*sigma*sqrt(2*pi)).*exp(-((log(x)-mu).^2/(2*(sigma^2)))); % Interspike interval distribution
num_pars=2; % Number of parameters
opts=statset('MaxIter',5e4,'MaxFunEvals',5e4,'Display','final'); % Set options for optimization

for ii=1:n_exps
    isi=all_isi{ii}/1000; % Extract experiment ISIs, in seconds
    numobs=length(isi); % Number of ISIs
    run_cell=cell(nRuns,3); % Col 1: Parameter values, Col 2: 95% CI Intervals, Col 3: Objective Function Value
    
    mean_isi=mean(isi); % Mean of ISIs
    variance_isi=var(isi); % Variance of ISIs
    cv_isi=sqrt(variance_isi)/mean_isi; % CV of ISIs

    jj=1;
    rng('shuffle');
    while jj<nRuns+1
        lb=eps*ones(1,num_pars); % Parameter lower bound
        ub=200*ones(1,num_pars); % Parameter upper bound
        pars0=(ub-lb).*rand(1,num_pars); % Initialize parameter values within bounds
        
        try
        [opt_pars,pci]=mle(isi,'pdf',log_pdf,'start',pars0,'LowerBound',lb,'UpperBound',ub,'Options',opts);
        
        mu=opt_pars(1);
        sigma=opt_pars(2);
        negL=-sum(log(log_pdf(isi,mu,sigma)));
        
        run_cell{jj,1}=opt_pars;
        run_cell{jj,2}=pci;
        run_cell{jj,3}=negL;
        jj=jj+1;
        catch ME
            if strcmp(ME.identifier,'stats:mle:NonpositivePdfVal')
                disp([num2str(jj) ': Ran Into Error!!']);
                continue;
            end
        end
    end
    
    % Extract objective function evaluations per run
    obj_vec=cell2mat(run_cell(:,3));
    obj_idx=find(obj_vec==min(obj_vec)); % Find objective function minimum across runs
    
    % Identify optimal parameter values and confidence intervals
    opt_pars=run_cell{obj_idx,1};
    opt_pci=run_cell{obj_idx,2};

    mu_opt=opt_pars(1);
    sigma_opt=opt_pars(2);
    
    logL=sum(log(log_pdf(isi,mu_opt,sigma_opt)));

    AIC=2*num_pars-2*logL; % Akaike Information Criterion
    BIC=num_pars*log(numobs)-2*logL; % Bayesian Information Criterion

    mean_pdf=exp(mu_opt+(sigma_opt^2)/2); % Distribution Mean
    variance_pdf=(exp(sigma_opt^2)-1)*exp(2*mu_opt+sigma_opt^2); % Distribution Variance
    cv_pdf=sqrt(variance_pdf)/mean_pdf; % Distribution CV
    
    % Save results to structure array
    save_pars(ii).Parameters=opt_pars;
    save_pars(ii).ConfIntervals=opt_pci;
    save_pars(ii).Mean_ISI=mean_isi;
    save_pars(ii).Variance_ISI=variance_isi;
    save_pars(ii).CV_ISI=cv_isi;
    save_pars(ii).Mean_PDF=mean_pdf;
    save_pars(ii).Variance_PDF=variance_pdf;
    save_pars(ii).CV_PDF=cv_pdf;
    save_pars(ii).AIC=AIC;
    save_pars(ii).BIC=BIC;
    save_pars(ii).logL=logL;
    save_pars(ii).NumPars=num_pars;
    save_pars(ii).NumObs=length(isi);
end
save_name=['.\' foldername  '\' foldername '_' totalname '.mat'];
save(save_name,'nRuns','save_pars','save_pars');