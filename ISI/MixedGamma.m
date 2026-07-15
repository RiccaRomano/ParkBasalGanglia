% Richard Foster and Cheng Ly
% Code identifies the maximum likelihood estimated fits of the mixed Gamma distribution family to the interspike intervals
clear;
close all;

foldername='MixedGamma'; % Distribution type
typename='Cntr'; % Experiment type (healthy, PD)
varname='STN'; % Neuron type (GPe, GPi, STN)
filename='stn';
totalname='CntrSTN';

mkdir(foldername);

data=load([filename 'data.mat'],['isi' typename 'SpkT' varname],['isi' typename varname]);
fn=fieldnames(data);
all_spkTms=data.(fn{1});
all_isi=data.(fn{2});

n_exps=length(all_isi); % Number of experiments
save_pars=struct('Parameters',{},'ConfIntervals',{},'Mean_ISI',{},'Variance_ISI',{},'CV_ISI',{},...
    'Mean_PDF',{},'Variance_PDF',{},'CV_PDF',{},'AIC',{},'BIC',{},'logL',{},'NumPars',{},'NumObs',{}); % 

nRuns=50; % Number of optimization runs
mixed_gamma=@(x,a1,a2,t1,t2,p) p/(gamma(a1)*(t1^a1))*x.^(a1-1).*exp(-x/t1)+(1-p)/(gamma(a2)*(t2^a2))*x.^(a2-1).*exp(-x/t2);
num_pars=5; % Number of parameters
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
        lb=[eps*ones(1,num_pars-1) 0]; % Parameter lower bound
        ub=[20*ones(1,num_pars-1) 1]; % Parameter upper bound
        pars0=(ub-lb).*rand(1,num_pars); % Initialize parameter values within bounds
        
        try
        [opt_pars,pci]=mle(isi,'pdf',mixed_gamma,'start',pars0,'LowerBound',lb,'UpperBound',ub,'Options',opts);
        
        a1=opt_pars(1);
        a2=opt_pars(2);
        b1=opt_pars(3);
        b2=opt_pars(4);
        p=opt_pars(5);

        negL=-sum(log(mixed_gamma(isi,a1,a2,b1,b2,p)));
        
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

    a1_opt=opt_pars(1);
    a2_opt=opt_pars(2);
    b1_opt=opt_pars(3);
    b2_opt=opt_pars(4);
    p_opt=opt_pars(5);
    
    % Calculate log-likelihood
    logL=sum(log(mixed_gamma(isi,a1_opt,a2_opt,b1_opt,b2_opt,p_opt)));
    
    % Calculate Akaike and Bayesian Information Criteria
    AIC=2*num_pars-2*logL;
    BIC=num_pars*log(numobs)-2*logL;

    mean_pdf=p_opt*a1_opt*b1_opt+(1-p_opt)*a2_opt*b2_opt; % Distribution Mean
    variance_pdf=p_opt*a1_opt*b1_opt^2+(1-p_opt)*a2_opt*b2_opt^2; % Distribution Variance
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