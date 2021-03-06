%% 2b) PARAMETRIC BOOTSTRAPPED 95% CONFIDENCE INTERVALS

% CLEAR PREVIOUS EXERCISES
clear

% SET SEED
SEED = 69;
random_nbr_generator = rng(SEED);

% LOAD TEXT FILE
data = load('atlantic.txt');

% USE PROVIDED FUNCTION FOR ESTIMATION OF BETA AND MU
[beta_mean, mu_mean] = est_gumbel(data);

% FIND LENGTH OF SPECIFIED DATA AND OTHER RELEVANT PARAMETERS FOR QUESTION
n = length(data); 
samples = 2000;

% USE SAMPLES TO GENERATE NECESSARY PARAMETERS
significance_lvl = 0.05; 
[alpha, alpha2, beta_array, mu_array, delta] = generate_parameters(samples, significance_lvl);

% CALCULATING DELTA USING GUMBEL DIST. TO ESTIMATE BETA AND MU(LOOP)
for i = 1:samples
    % DRAW u 
    u = rand(n, 1);
    
    % CALCULATE WAVE FROM F INVERSE
    wave = F_inverse_question_2(u, beta_mean, mu_mean);
    
    % ESTIMATE BETA AND MU USING GIVEN FILE(FUNCTION)
    [beta_est, mu_est] = est_gumbel(wave);
    
    % ADD ESTIMATES TO ARRAYS
    beta_array(i) = beta_est;
    mu_array(i) = mu_est;
    
    % CALCULATE DEVIATION FROM MEAN FROM GUMBEL DISTRIBUTION
    delta(i,:) = [beta_mean - beta_est, mu_mean - beta_est];
end

% SORT DELTA FOR COMPARISON PURPOSES
delta = sort(delta);

% CALCULATE CONFIDENCE BETA AND MU INTERVALS USING BOOTSTRAP METHOD
confidence_interval = zeros(3,2);
confidence_interval(1,:) = beta_mean + [delta(floor(alpha)), delta(ceil(alpha2))];
confidence_interval(2,:) = mu_mean + [delta(floor(alpha)), delta(ceil(alpha2))];


%% 2c) ONE-SIDED PARAMETRIC BOOTSTRAPPED 95% CONFIDENCE INTERVAL FOR THE 100-YEAR RETURN VALUE

% OBSERVATIONS 
T = 3*14*100;

% ONE-SIDED ALPHA
one_side_alpha = 0.95*samples;

% SET WAVES IN ARRAY 
wave_array = zeros(samples, 1); 

% SET VALUES IN ARRAY 
value_draw = 1 - 1/T;
for i = 1:samples
    wave_array(i) = F_inverse_question_2(value_draw, beta_array(i), mu_array(i));
end

% CALCULATE MEAN FOR WAVE
wave_mean = F_inverse_question_2(value_draw, beta_mean, mu_mean);

% CALCULATE DEVATION FROM MEAN
wave_delta = wave_mean - wave_array;

% SORT FOR PRACTICAL REASONS
wave_delta = sort(wave_delta);

% CONFIDENCE INTERVAL
confidence_interval(3,:) =[0, wave_mean + wave_delta(ceil(one_side_alpha))]; 

% PLOT
x = linspace(0,1,582);
plot(x, wave);
figure(2)
plot(1:582, data);



