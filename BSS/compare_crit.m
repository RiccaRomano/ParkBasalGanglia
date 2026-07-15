%% Richard Foster and Cheng Ly
% Code compares proximity to criticality within neuron type, accros health condition
clear;
close all;

typename1='Cntr';
typename2='Prk';
varname='STN';
filename='stn';

% Group 1
load([typename1 varname '_d2Results.mat']);
groups=cell(2,1);

d2cell=struct2cell(d2_results);
group_temp={d2cell{2,1,:}};
group_temp=group_temp';
group_temp=cell2mat(group_temp);
group_temp=rmoutliers(group_temp);
groups{1}=group_temp;

rmNaN_group1=groups{1}(~isnan([groups{1}]));
med_group1=median(rmNaN_group1);
iqr_group1=iqr(rmNaN_group1);

% Group 2
load([typename2 varname '_d2Results.mat']);

d2cell=struct2cell(d2_results);
group_temp={d2cell{2,1,:}};
group_temp=group_temp';
group_temp=cell2mat(group_temp);
group_temp=rmoutliers(group_temp);
groups{2}=group_temp;

rmNaN_group2=groups{2}(~isnan([groups{2}]));
med_group2=median(rmNaN_group2);
iqr_group2=iqr(rmNaN_group2);

% Wilcoxon Rank Sum
[pValue, h, stats] = ranksum(groups{1}, groups{2});

f1=figure;
groupA_labels = repmat({typename1}, length(groups{1}), 1);
groupB_labels = repmat({typename2}, length(groups{2}), 1);
all_db_values = [groups{1}; groups{2}];
all_labels = [groupA_labels; groupB_labels];
boxplot(all_db_values, all_labels, 'Colors', 'k', 'Symbol', 'ko');
ylabel('Distance to Criticality, d_b (bits/s)', 'FontSize', 12, 'FontWeight', 'bold');
title(['Critical Proximity - ' varname], 'FontSize', 14);

ylim([-0.2 2]);
set(gca, 'FontSize', 12);
if pValue < 0.05
    hold on;
    y_max = max(all_db_values);
    plot([1, 2], [y_max * 1.05, y_max * 1.05], '-k', 'LineWidth', 1.5);
    text(1.5, y_max * 1.08, '*', 'FontSize', 20, 'HorizontalAlignment', 'center');
    hold off;
end

f2=figure;
categorical_labels = categorical(all_labels);
violinplot(categorical_labels, all_db_values);
ylabel('Distance to Criticality, d_b (bits/s)', 'FontSize', 12, 'FontWeight', 'bold');
title(['Critical Proximity - ' varname], 'FontSize', 14);

set(gca, 'FontSize', 12);
if pValue < 0.05
    hold on;
    y_max = max(all_db_values);
    plot([1, 2], [y_max * 1.05, y_max * 1.05], '-k', 'LineWidth', 1.5);
    text(1.5, y_max * 1.08, '*', 'FontSize', 20, 'HorizontalAlignment', 'center');
    hold off;
end