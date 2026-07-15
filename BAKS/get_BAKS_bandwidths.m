function h = get_BAKS_bandwidths(spkTms, pars)
%% INPUTS

% spkTms: vector of spike times, in seconds
% pars: vector of BAKS hyperparameter values (a1, a2, b1, b2, p)

%% OUTPUTS

% h: vector of BAKS bandwidth values

a1 = pars(1);
a2 = pars(2);
b1 = pars(3);
b2 = pars(4);
p1 = max(pars(5), eps);
p2 = max(1 - p1, eps);

spikes = spkTms(:);
w1 = exp(log(p1) - a1 * log(b1));
g1 = exp(gammaln(a1 + 0.5) - gammaln(a1));
v1 = w1 * g1;
inv_b1 = 1 / b1;

w2 = exp(log(p2) - a2 * log(b2));
g2 = exp(gammaln(a2 + 0.5) - gammaln(a2));
v2 = w2 * g2;
inv_b2 = 1 / b2;

sq_dist = (spikes - spikes').^2 ./ 2;
base1 = sq_dist + inv_b1;
num1 = w1 .* (base1 .^ (-a1));
denom1 = v1 .* (base1 .^ (-a1 - 0.5));

base2 = sq_dist + inv_b2;
num2 = w2 .* (base2 .^ (-a2));
denom2 = v2 .* (base2 .^ (-a2 - 0.5));
sumnum = sum(num1 + num2, 2);
sumdenom = sum(denom1 + denom2, 2);

h = sumnum ./ sumdenom;
end