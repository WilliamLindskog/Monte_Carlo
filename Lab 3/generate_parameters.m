% GENERATE NECESSARY PARAMETERS
function [alpha, alpha2, beta_array, mu_array, delta] = generate_parameters(samples, significance_lvl)
    alpha = (significance_lvl/2)*samples;
    alpha2  =(1-(significance_lvl/2))*samples;
    beta_array = zeros(samples, 1);
    mu_array = zeros(samples, 1);
    delta = zeros(samples, 2); 
end