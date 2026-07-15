% Richard Foster and Cheng Ly
% Code compares KS-statistics from Time-Rescaling Theorem goodness-of-fit testing of all firing rates
clear;
close all;

typename='Prk';
varname='STN';
filename='stn';

% Load OKS results
cd ..\OKS\
load([typename varname '_FinalResults.mat']);
IP_OKS=[results.KS_stat]';
n_obsIP_OKS=length(IP_OKS);

% Load BAKS results
cd ..\BAKS\
load([typename varname '_FinalResults.mat']);
IP_BAKS=[results.KS_Pois]';
n_obsIP_BAKS=length(IP_BAKS);
IG_BAKS=[results.KS_Gamma]';
n_obsIG_BAKS=length(IG_BAKS);

% Load BSS results
cd ..\BSS\
load([typename varname '_FinalResults.mat']);
IG_BSS=[results.KS_stat]';
n_obsIG_BSS=length(IG_BSS);

% Change directory back to original
cd ..\TRT

% Initialize results structure
results=struct('Model_1',{},'Model_2',{},'KS_stats',{},'Stats',{},'Effect',{},'PVal_Raw',{},'PVal_Adj',{});

% KS statistics
KS_stats = [IP_OKS, IP_BAKS, IG_BAKS, IG_BSS];
med_KS=median(KS_stats); % Median
iqr_KS=iqr(KS_stats); % IQR

group_names = {'Poisson OKS', 'Poisson BAKS', 'Gamma BAKS', 'Bayesian SS'};

% Friedman test, groups
[p_Friedman, ~, ~] = friedman(KS_stats, 1, 'off'); 
n_models = 4;
num_comparisons = (n_models * (n_models - 1)) / 2;

% Pairwise, Wilcoxon signed-rank test, with corrections
fprintf('\n--- Pairwise Wilcoxon Signed-Rank Tests (Bonferroni Corrected) ---\n');
fprintf('%-25s | %-12s | %-12s | %-10s\n', 'Comparison', 'p-raw', 'p-adj', 'Effect Size');

count=1;
for i = 1:(n_models-1)
    for j = (i+1):n_models
        data1 = KS_stats(:, i);
        data2 = KS_stats(:, j);

        if length(data1) < 15 || length(data2) <15  % no z-value if sample size too small, use approximate method
            [p_raw, ~, stats] = signrank(data1, data2, 'method', 'approximate');
        else
            [p_raw, ~, stats] = signrank(data1, data2);
        end
        zval = stats.zval;

        % Effect size
        sign_eff = abs(zval) / sqrt(length(data1) * 2);
        
        % Bonferroni correction
        p_adj = min(1, p_raw * num_comparisons);
        
        comp_name = sprintf('%s vs %s', group_names{i}, group_names{j});
        fprintf('%-25s | %.4e   | %.4e   | %.3f\n', comp_name, p_raw, p_adj, sign_eff);
        
        % Save results
        results(count).Model_1=group_names{i};
        results(count).Model_2=group_names{j};
        results(count).KS_stats=KS_stats;
        results(count).Stats=stats;
        results(count).Effect=sign_eff;
        results(count).PVal_Raw=p_raw;
        results(count).PVal_Adj=p_adj;
        count=count+1;
    end
end
fprintf('\n');

% Figure: shows KS statistics, boxplot
f1 = figure('Renderer', 'painters', 'Color', 'w', 'Position', [100, 100, 700, 500]);
hold on; box off; grid on;
boxplot(KS_stats, 'Labels', group_names, 'Colors', 'k', 'Symbol', 'ko','Widths', 0.4); 
ylabel('Time-Rescaling KS Statistic', 'FontSize', 12, 'FontWeight', 'bold');
title(['Model Fit Comparison, ' typename varname], 'FontSize', 14);
ylim([0 0.4]);
hold off;

fig_name=[typename varname '_TRTs.fig'];
svg_name=[typename varname '_TRTs.svg'];
eps_name=[typename varname '_TRTs.eps'];
png_name=[typename varname '_TRTs.png'];

saveas(f1, fig_name);
saveas(f1, svg_name);
saveas(f1, png_name);
saveas(f1, eps_name);

save([typename varname '_TRTResults.mat'],'KS_stats','group_names','results');
