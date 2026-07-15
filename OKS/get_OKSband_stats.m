% Richard Foster and Cheng Ly
% Code loads OKS bandwidths and compares median and IQR across cell type and condition

clear;
close all;

% Create group names for statistical comparisons
group_names = {'CntrGPe', 'PrkGPe', 'CntrGPi', 'PrkGPi', 'CntrSTN', 'PrkSTN'};
label_vec={'A','B','C','D','E','F'};

% Initialize median and IQR vectors of OKS bandwidths
med_h=zeros(length(group_names),1);
iqr_h=zeros(length(group_names),1);

for ii=1:length(group_names)
    load([group_names{ii} '_FinalResults.mat']); % Load data
    width=[results.Width]; % Extract OKS bandwidths

    med_h(ii)=median(width); % Save median
    iqr_h(ii)=iqr(width); % Save IQR
end