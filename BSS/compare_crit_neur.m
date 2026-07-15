%% Richard Foster and Cheng Ly
% Code compares proximity to criticality results across neuron type
clear;
close all;

typename='Prk';
varname1='GPi';
varname2='STN';

% Group 1
load([typename varname1 '_d2Results.mat']);

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
load([typename varname2 '_d2Results.mat']);

d2cell=struct2cell(d2_results);
group_temp={d2cell{2,1,:}};
group_temp=group_temp';
group_temp=cell2mat(group_temp);
group_temp=rmoutliers(group_temp);
groups{2}=group_temp;

rmNaN_group2=groups{2}(~isnan([groups{2}]));
med_group2=median(rmNaN_group2);
iqr_group2=iqr(rmNaN_group2);

% Wilcoxon Rank-sum test on d2 calculations from the two groups
[pValue, h, stats] = ranksum(groups{1}, groups{2});

% EFfect size
eff=abs(stats.zval)/(sqrt(length(groups{1})+length(groups{2})));