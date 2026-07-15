function binnedSpikes = binSpikes(spikeTimes, binSize, startTime, endTime)
%% INPUTS

% spikeTimes: A vector of spike times (in seconds).
% binSize: The size of each time bin (in the same units as spikeTimes).
% startTime: The start time of the first bin.
% endTime: The end time of the last bin.

%% OUTPUTS

% binnedSpikes: vector, elements are the number of spikes within a time bin

% Define the bin edges
binEdges = startTime:binSize:endTime;

% Use histcounts to count spikes in each bin
binnedSpikes = histcounts(spikeTimes, binEdges);

end