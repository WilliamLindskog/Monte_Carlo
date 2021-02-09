%% Part 1

%% Part 2
load powercurve_V164 %Saved in P
%p = 1.225; %air density
%d = 164;   %rotor diameter
%h = 105;   %tower height

%{
Month     - Number of days
Jan       - 31
Feb       - 28 (29 in leap years)
Mar       - 31
April     - 30
May       - 31
June      - 30
July      - 31
August    - 31
September - 30
October   - 31
November  - 30
December  - 31
%}
A = [10.6 9.7 9.2 8 7.8 8.1 7.8 8.1 9.1 9.9 10.6 10.6]; %lambda
B = [2 2 2 1.9 1.9 1.9 1.9 1.9 2 1.9 2 2]; %k
days = [31 28 31 30 31 30 31 31 30 31 30 31];

%% 2 a) Standard Monte Carlo
tauN_standardMC_final = zeros(1,12);
I_width_standardMC_final = zeros(1,12);
I_valueU = zeros(1,12);
I_valueD = zeros(1,12);
for month = 1:12
N = 6000;

%{
Optimized version below!!!
X = zeros(1,N);
for i = 1:N
    X(i) = wblrnd(A(month),B(month));
end
%}

X = wblrnd(A(month),B(month), N, 1);

% Calculating phi(X)
phi = P(X);

lambda = 1.96;
tauN = zeros(1,N);
I_lower = zeros(1,N);
I_upper = zeros(1,N);

for i = 1:N
    
    % Estimating tauN
    tauN(i) = 1/i * sum(phi(1:i));
    
    % Estimating confidence interval
    sigma2phi = var(phi(1:i));
    I_lower(i) = tauN(i) - lambda*sqrt(sigma2phi/i);
    I_upper(i) = tauN(i) + lambda*sqrt(sigma2phi/i);

end
tauN_standardMC_final(month) = tauN(end);
I_width_standardMC_final(month) = I_upper(end) - I_lower(end);
I_valueU(month) = round(I_upper(end)./1000000, 3); %MW
I_valueD(month) = round(I_lower(end)./1000000, 3); %MW
end
I = [I_valueU ; I_valueD]; %MW
a = plot(tauN./1000000); title('MC convergence'), ylabel('Power (MW)'), xlabel('Sample size');
hold on
b = plot(I_upper./1000000, 'Color', '#A2142F');
plot(I_lower./1000000, 'Color', '#A2142F');
hold off
legend([a,b],'Expected amount of power', 'Confidence interval')

%% 2 a) TRUNCATED Standard Monte Carlo
tauN_truncatedMC_final = zeros(1,12);
I_width_truncatedMC_final = zeros(1,12);
I_valueU = zeros(1,12);
I_valueD = zeros(1,12);
for month = 1:12
N = 6000;
U = rand(1,N);
a = 3.5;
b = 25;
F_X_a = wblcdf( a, A(month), B(month) );
F_X_b = wblcdf( b, A(month), B(month) );

%F_X = @(x) wblcdf(x, A(month), B(month) );
%F_X_cond = @(x) (F_X(x) - F_X_a) / (F_X_b - F_X_a);

F_inv_X = @(x) icdf('wbl', x, A(month), B(month));
F_inv_X_cond = @(x) F_inv_X( x*(F_X_b - F_X_a) + F_X_a );

% F_gen_inv = F_inv_X_cond 
F_gen_inv = @(x) F_inv_X( x*(F_X_b - F_X_a) + F_X_a );

X = F_gen_inv(U);

% Calculating phi(X)
phi = P(X);

lambda = 1.96;
tauN = zeros(1,N);
I_lower = zeros(1,N);
I_upper = zeros(1,N);
p = (F_X_b - F_X_a);
for i = 1:N
    
    % Estimating tauN
    tauN(i) = 1/i *p*sum(phi(1:i)); %FL 4 slide 27
    
    % Estimating confidence interval
    sigma2phi = p^2.*var(phi(1:i)); %FL4 slide 28
    I_lower(i) = tauN(i) - lambda*sqrt(sigma2phi/i);
    I_upper(i) = tauN(i) + lambda*sqrt(sigma2phi/i);
    
end
tauN_truncatedMC_final(month) = tauN(end);
I_width_truncatedMC_final(month) = I_upper(end) - I_lower(end);
I_valueU(month) = round(I_upper(end)./1000000, 3); %MW
I_valueD(month) = round(I_lower(end)./1000000, 3); %MW

end
I = [I_valueU ; I_valueD]; %MW
I_width_truncatedMC_final./I_width_standardMC_final
temp = 1 - I_width_truncatedMC_final./I_width_standardMC_final
mean(temp)
temp2 = round(100*(1 - I_width_truncatedMC_final./I_width_standardMC_final), 1)

a = plot(tauN./1000000); title('Truncated MC convergence'), ylabel('Power (MW)'), xlabel('Sample size');
hold on
b = plot(I_upper./1000000, 'Color', '#A2142F');
plot(I_lower./1000000, 'Color', '#A2142F');
hold off
legend([a,b],'Expected amount of power', 'Confidence interval')

%% 2b) Comparing ϕ(x), pdf f(x) and ϕ(x)*f(x) and determining new instrumental function g(x)

N = 6000;
xs = linspace(0,30, N);

% Jan, Determining suitable new instrumental function g(x)
month = 1;
r = 9.555;
lambda = 1.35;
g = gampdf(xs, r, lambda);

% Feb, Determining suitable new instrumental function g(x)
% month = 2;
% r = 9.9;
% lambda = 1.23;
% g = gampdf(xs, r, lambda);

% Mar, Determining suitable new instrumental function g(x)
% month = 3;
% r = 10.1;
% lambda = 1.15;
% g = gampdf(xs, r, lambda);

% Apr, Determining suitable new instrumental function g(x)
% month = 4;
% r = 10.05;
% lambda = 1.09;
% g = gampdf(xs, r, lambda);

% May, Determining suitable new instrumental function g(x)
% month = 5;
% r = 10.2;
% lambda = 1.05;
% g = gampdf(xs, r, lambda);

% Jun, Determining suitable new instrumental function g(x)
% month = 6;
% r = 10.05;
% lambda = 1.09;
% g = gampdf(xs, r, lambda);

% Jul, Determining suitable new instrumental function g(x)
% month = 7;
% r = 10.2;
% lambda = 1.05;
% g = gampdf(xs, r, lambda);

% Aug, Determining suitable new instrumental function g(x)
% month = 8;
% r = 10.05;
% lambda = 1.09;
% g = gampdf(xs, r, lambda);

% Sep, Determining suitable new instrumental function g(x)
% month = 9;
% r = 10.1;
% lambda = 1.15;
% g = gampdf(xs, r, lambda);

% Oct, Determining suitable new instrumental function g(x)
% month = 10;
% r = 10;
% lambda = 1.25;
% g = gampdf(xs, r, lambda);

% Nov, Determining suitable new instrumental function g(x)
% month = 11;
% r = 9.555;
% lambda = 1.35;
% g = gampdf(xs, r, lambda);

% Dec, Determining suitable new instrumental function g(x)
% month = 12;
% r = 9.555;
% lambda = 1.35;
% g = gampdf(xs, r, lambda);

% Calculating pdf f(x)
f = wblpdf(xs, A(month), B(month));

% Calculating objective function ϕ(X)
phi = P(xs)';
phi = phi./100000000;

% Calculating ϕ(x)*f(x)
phif = phi.*f;
phif = phif.*22;

% Calculating new objective function ϕ(x)*f(x)/g(x)
% We aim to achieve the ratio ϕ(x)*f(x)/g(x) as constant as possible
phifg = phi.*f./g;

% Plot
a = plot(xs, f);
hold on
b = plot(xs, phi); 
c = plot(xs, phif, '--');
d = plot(xs, g, '--');
hold off
legend([a,b,c,d],'f(v)', 'P(v)', 'P(v)*f(v)', 'g(v)')

% Plot
% c = plot(xs, phif, '--');
% hold on
% d = plot(xs, g, '--');
% hold off
% legend([c,d], 'ϕ(x)*f(x)', 'Scaled g(x)')

figure(2)
e = plot(xs, phifg);
legend(e,'P(v)*f(v)/g(v)')

%% 2b) Importance sampling (IS) Monte Carlo
N = 6000;
tauN_IS_final = zeros(1,12);
I_width_IS_final = zeros(1,12);
I_valueU = zeros(1,12);
I_valueD = zeros(1,12);
for month = 1:12

    if month == 1
        % Jan
        r = 9.555;
        lambda = 1.35;
        X = gamrnd(r, lambda, N, 1); %We now sample from g instead of f
    elseif month == 2
        % Feb
        r = 9.9;
        lambda = 1.23;
        X = gamrnd(r, lambda, N, 1); %We now sample from g instead of f
    elseif month == 3;
        % Mar
        r = 10.1;
        lambda = 1.15;
        X = gamrnd(r, lambda, N, 1); %We now sample from g instead of f
    elseif month == 4;
        % Apr
        r = 10.05;
        lambda = 1.09;
        X = gamrnd(r, lambda, N, 1); %We now sample from g instead of f
    elseif month == 5;
        % May
        r = 10.2;
        lambda = 1.05;
        X = gamrnd(r, lambda, N, 1); %We now sample from g instead of f
    elseif month == 6;
        % Jun
        r = 10.05;
        lambda = 1.09;
        X = gamrnd(r, lambda, N, 1); %We now sample from g instead of f
    elseif month == 7;
        % Jul
        r = 10.2;
        lambda = 1.05;
        X = gamrnd(r, lambda, N, 1); %We now sample from g instead of f
    elseif month == 8;
        % Aug
        r = 10.05;
        lambda = 1.09;
        X = gamrnd(r, lambda, N, 1); %We now sample from g instead of f
    elseif month == 9;
        % Sep
        r = 10.1;
        lambda = 1.15;
        X = gamrnd(r, lambda, N, 1); %We now sample from g instead of f
    elseif month == 10;
        % Oct
        r = 10;
        lambda = 1.25;
        X = gamrnd(r, lambda, N, 1); %We now sample from g instead of f
    elseif month == 11;
        % Nov
        r = 9.555;
        lambda = 1.35;
        X = gamrnd(r, lambda, N, 1); %We now sample from g instead of f

    elseif month == 12;
        % Dec
        r = 9.555;
        lambda = 1.35;
        X = gamrnd(r, lambda, N, 1); %We now sample from g instead of f
    end

% Calculating ϕ(X)
phi = P(X);

% Calcualting w(X)
f = wblpdf(X, A(month), B(month));
g = gampdf(X, r, lambda);
w = f./g;

% Calculations and convergence plot
lambda = 1.96;
tauN_IS = zeros(1,N);
I_lower_IS = zeros(1,N);
I_upper_IS = zeros(1,N);
for i = 1:N
    
    phiw = phi(1:i).*w(1:i);
    
    % Estimating Expected Value of tauN
    tauN_IS(i) = 1/i * sum( phiw );
    
    % Estimating confidence interval
    sigma2phiw = var(phiw);
    I_lower(i) = tauN_IS(i) - lambda*sqrt(sigma2phiw/i);
    I_upper(i) = tauN_IS(i) + lambda*sqrt(sigma2phiw/i);
    
end
tauN_IS_final(month) = tauN_IS(end);
I_width_IS_final(month) = I_upper(end) - I_lower(end);
I_valueU(month) = round(I_upper(end)./1000000, 3); %MW
I_valueD(month) = round(I_lower(end)./1000000, 3); %MW
end
I = [I_valueU ; I_valueD]; %MW
temp = 1 - I_width_IS_final./I_width_standardMC_final
mean(temp)
temp2 = round(100*(1 - I_width_IS_final./I_width_standardMC_final), 1)
a = plot(tauN_IS./1000000); title('Importance sampling MC convergence'), ylabel('Power (MW)'), xlabel('Sample size');
hold on
b = plot(I_upper./1000000, 'Color', '#A2142F');
plot(I_lower./1000000, 'Color', '#A2142F');
hold off
legend([a,b],'Expected amount of power', 'Confidence interval')


%% 2 c) Antithetic sampling
%{
The power curve P(v) is monotonously increasing over the interval (3.5, 14) and constant over (14, 25). Use this for
reducing the variance of the estimator in (a) via antithetic sampling. Construct a new 95% confidence interval using
the robustified estimator and compare it to the ones obtained in (a) and (b).
%}
N = 3000;
tauN_AS_final = zeros(1,12);
I_width_AS_final = zeros(1,12);
I_valueU = zeros(1,12);
I_valueD = zeros(1,12);

U = rand(1,N);
T = @(x) 1-x; % Transform
U_bar = T(U);

for month = 1:12
% Wind speed
X = wblinv(U, A(month), B(month));
X_bar = wblinv(U_bar, A(month), B(month));

% Calculating phi(X)
phi_anti = P(X);
phi_anti_bar = P(X_bar);

% Calculating W
W = (phi_anti+phi_anti_bar)/2;

lambda = 1.96;
tauN = zeros(1,N);
I_lower = zeros(1,N);
I_upper = zeros(1,N);
for i = 1:N
    
    % Estimating tauN
    tauN(i) = 1/i * sum(W(1:i));
    
    % Estimating confidence interval
    data_corr = corr(phi_anti(1:i), phi_anti_bar(1:i));
    %sigma2phi = 1/2 * ( var(phi_anti(1:i)) + data_corr);
    sigma2phi = var(W(1:i));
    I_lower(i) = tauN(i) - lambda*sqrt(sigma2phi/i);
    I_upper(i) = tauN(i) + lambda*sqrt(sigma2phi/i);
    
end
tauN_AS_final(month) = tauN(end);
I_width_AS_final(month) = I_upper(end) - I_lower(end);
I_valueU(month) = round(I_upper(end)./1000000, 3); %MW
I_valueD(month) = round(I_lower(end)./1000000, 3); %MW
end
I_width_AS_final./I_width_standardMC_final
I = [I_valueU ; I_valueD]; %MW
temp = 1 - I_width_AS_final./I_width_standardMC_final
mean(temp)
temp2 = round(100*(1 - I_width_AS_final./I_width_standardMC_final), 1)

a = plot(tauN./1000000); title('Antithetic sampling MC convergence'), ylabel('Power (MW)'), xlabel('Sample size');
hold on
b = plot(I_upper./1000000, 'Color', '#A2142F');
c = plot(I_lower./1000000, 'Color', '#A2142F');
hold off
%legend([a,b,c],'Expected amount of power', 'Upper bound', 'Lower bound')
legend([a,b],'Expected amount of power', 'Confidence interval')


%% 2 d)
%{
P[P(v) > 0] = P[3.5<v<25] = F(25) - F(3.5)
%}
prob_power = zeros(1,12);
for month = 1:12
    F_X_a = wblcdf( 3.5, A(month), B(month) );
    F_X_b = wblcdf( 25, A(month), B(month) );
    prob_power(month) =  F_X_b - F_X_a
end
temp = round(100*prob_power,1)
%% 2 e) använd tauN från bästa metoden
% tauN_IS_final(month)
% I_width_IS_final(month)
d = 165;
%h = 105;
p = 1.225; %air density at sea level
%P_total = @(v) (1/2)*p*pi*(d^2)*(v^3)/4; % total wind power

power_actual = zeros(1,12);
power_total = zeros(1,12);
power_coefficient = zeros(1,12);
power_coefficient_I_width = zeros(1,12);
I_upper = zeros(1,12);
I_lower = zeros(1,12);
for month = 1:12
%power_total(month) = P_total(M);
power_total(month) = (3/8)*A(month)*p*pi*(d^2)*B(month)^4*gamcdf(3, 3, 1/B(month));

power_actual(month) = tauN_IS_final(month);

power_coefficient(month) = power_actual(month)/power_total(month);
power_coefficient_I_width(month) = I_width_IS_final(month)/power_total(month);
end
power_coefficient
power_coefficient_I_width
I = [round(100*(power_coefficient+power_coefficient_I_width/2),1) ; round(100*(power_coefficient-power_coefficient_I_width/2),1)]
%% 2 f)

capacity_factor = mean(tauN_IS_final/(9.5*10^6)) %0.3932
availability_factor = mean(prob_power) % 0.8501 < 0.9


%% 3a) one-dim Determining suitable new instrumental function g(x)

N = 6000;
xs = linspace(0,30, N);

% Jan, Determining suitable new instrumental function g(x)
r = 9.85;
lambda = 1.2;
g = gampdf(xs, r, lambda);

% Calculating pdf f(x)
f = wblpdf(xs, 9.13, 1.96);

% Calculating objective function ϕ(X)
phi = P(xs)';
phi = phi./100000000;

% Calculating ϕ(x)*f(x)
phif = phi.*f;
phif = phif.*22;

% Calculating new objective function ϕ(x)*f(x)/g(x)
% We aim to achieve the ratio ϕ(x)*f(x)/g(x) as constant as possible
phifg = phi.*f./g;

% Plot
a = plot(xs, f);
hold on
b = plot(xs, phi); 
c = plot(xs, phif, '--');
d = plot(xs, g, '--');
hold off
legend([a,b,c,d],'f(v)', 'P(v)', 'P(v)*f(v)', 'g(v)')

% Plot
% c = plot(xs, phif, '--');
% hold on
% d = plot(xs, g, '--');
% hold off
% legend([c,d], 'ϕ(x)*f(x)', 'Scaled g(x)')

figure(2)
e = plot(xs, phifg);
legend(e,'p(v)*f(v)/g(v)')


%% 3a) Importance sampling (IS) Monte Carlo
N = 6000;
tauN_IS_final = zeros(1,12);
I_width_IS_final = zeros(1,12);
I_valueU = zeros(1,12);
I_valueD = zeros(1,12);

r = 9.85;
lambda = 1.2;
X = gamrnd(r, lambda, N, 1); %We now sample from g instead of f


% Calculating ϕ(X)
phi = P(X);

% Calcualting w(X)
f = wblpdf(X, 9.13, 1.96);
g = gampdf(X, r, lambda);
w = f./g;

% Calculations and convergence plot
lambda = 1.96;
tauN_IS = zeros(1,N);
I_lower_IS = zeros(1,N);
I_upper_IS = zeros(1,N);
    
phiw = phi.*w;
    
% Estimating Expected Value of tauN
tauN_IS = mean(phiw)
tauN_IS_dual = 2*tauN_IS

%% 3 b) c) Determining suitable new instrumental function g(x)
a = 0.638;
p = 3;
q = 1.5;

N = 100;
xs = linspace(0,30, N);

% PDF f bivariate function
%F_2 = @(v1, v2) wblcdf(v1, 9.13, 1.96).*wblcdf(v2, 9.13, 1.96).*(1 + 0.638*(1-(wblcdf(v1, 9.13, 1.96).^3)).^1.5 .* (1-(wblcdf(v2, 9.13, 1.96).^3)).^1.5);
f = @(v) wblpdf(v, 9.13, 1.96);
F = @(v) wblcdf(v, 9.13, 1.96);
f_2 = @(v1, v2) f(v1).*f(v2).*(1 + 0.638*(1-((F(v1)).^3)).^(1.5-1) .* (1-(F(v2)).^3).^(1.5-1) .* (((F(v1)).^3) .* (1+3*1.5) - 1) .* (((F(v2)).^3) .* (1+3*1.5) - 1));

% Determining suitable new instrumental function g(x)
u = 12.12;
sigma = 17;
g = @(v1,v2) mvnpdf([v1 v2], [u u], [sigma 0 ; 0 sigma]);
g_rand = @(N) mvnrnd([u u], [sigma 0 ; 0 sigma], N);
g_single = @(v) normpdf(v, u, sigma);

% Calculating pdf f(v1,v2)
%f_dual = f_2(xs, xs);


% Objective function ϕ(V)=P(V) ->  ϕ(V1,V2) = ϕ(V1)*ϕ(V2)= P(V1)*P(V2)
%phi = P(xs)';
phi = @(v1,v2) (P(v1)'.*P(v2)'); %obs kanske ta bort "'"

% ϕ(v1,v2)*f_2(v1, v2)
%phif = phi.*f_2(xs,xs);
phif = @(v1,v2) phi(v1,v2).*f_2(v1,v2);

% New objective function ϕ(v1,v2)*f(v1,v2)/g(v1,v2)
% We aim to achieve the ratio ϕ(v1,v2)*f(v1,v2)/g(v1,v2) as constant as possible
%phifg = phi.*f_2(x2,x2)./g_2(xs,xs);
phifg = @(v1,v2) phi(v1,v2).*f_2(v1,v2)./g(v1,v2);


[x,y] = meshgrid(xs,xs);
% Plot f_2(v1,v2)
% figure('Name','PDF f(v1,v2)','NumberTitle','off')
% Z = zeros(length(xs),length(xs));
% for i = 1:length(xs)
%     for j = 1:length(xs)
%         Z(i,j) = f_2(xs(i),xs(j));
%     end
% end
% surf(x,y, Z);

%plot ϕ(v1,v2)*f(v1, v2)
figure('Name','P(v1,v2)*f(v1,v2)','NumberTitle','off')
Z = zeros(length(xs),length(xs));
for i = 1:length(xs)
    for j = 1:length(xs)
        Z(i,j) = phif(xs(i),xs(j));
    end
end
surf(x,y, Z);

% Plot g(v1,v2)
figure('Name','g(v1,v2))','NumberTitle','off')
Z = zeros(length(xs),length(xs));
for i = 1:length(xs)
    for j = 1:length(xs)
        Z(i,j) = g(xs(i),xs(j));
    end
end
surf(x, y, Z);

% Plot ϕ(v1,v2)*f(v1,v2)/g(v1,v2)
figure('Name','P(v1,v2)*f(v1,v2)/g(v1,v2)','NumberTitle','off')
Z = zeros(length(xs),length(xs));
for i = 1:length(xs)
    for j = 1:length(xs)
        Z(i,j) = phifg(xs(i),xs(j));
    end
end
surf(x, y, Z);

%% 3 c) ALWAYS RUN SECTION ABOVE BEFORE
N = 6000;
lambda = 9.13;
k = 1.96;

% Generate samples, now from g instead of f

X = g_rand(N);
X1 = X(:,1);
X2 = X(:,2);


% Calculating ϕ(v1) and ϕ(v2)
% phi1 = P(X1);
% phi2 = P(X2);
% Calculating ϕ(v1,v2) = P(v1,v2) = P(v1)*P(v2)
phi_prod = P(X1).*P(X2);


% Calcualting importance weight w(v1,v2) = f(v1,v2)/g(v1,v2) (= f(v1,v2)/( g1(v1)*g2(v2) )
w_func = @(v1,v2) f_2(v1,v2)./g(v1,v2);
w = w_func(X1,X2);

phiw = phi_prod.*w;

% Estimating Expected Value of tauN
tauN_IS = mean(phiw)
%% 3 b) c) ALWAYS RUN SECTION ABOVE BEFORE
N = 6000;
% E[X + Y] = E[X] + E[Y]
% Var[X + Y] = Var[X] + Var[Y] + 2Cov[X,Y]
lambda = 9.13;
k = 1.96;

% Generate samples, now from g instead of f
% r1 = 10.07;
% lambda1 = 1.16;
% r2= 10.07;
% lambda2 = 1.16;
% X1 = gamrnd(r1, lambda1, N, 1); %We now sample from g instead of f
% X2 = gamrnd(r2, lambda2, N, 1); %We now sample from g instead of f
% OR ------------
%  u = [12.12 12.12];
%  sigma = [17 0 ; 0 17];
% X = mvnrnd(u, sigma, N);
X = g_rand(N);
X1 = X(:,1);
X2 = X(:,2);
% ----------

% Calculating ϕ(v1) and ϕ(v2)
phi1 = P(X1);
phi2 = P(X2);
% Calculating ϕ(v1,v2) = P(v1,v2) = P(v1)*P(v2)
phi_prod = P(X1).*P(X2);

% Calcualting importance weights w1(X1) and w2(X2)
% f1 = f(X1);
% f2 = f(X2);
% g1 = g_single(X1);
% g2 = g_single(X2);
% w1 = f1./g1;
% w2 = f2./g2;
% Calcualting importance weight w(v1,v2) = f(v1,v2)/g(v1,v2) (= f(v1,v2)/( g1(v1)*g2(v2) )
w_func = @(v1,v2) f_2(v1,v2)./g(v1,v2);
w = w_func(X1,X2);


lambda = 1.96;
tauN1 = zeros(1,N);
tauN2 = zeros(1,N);
tauN = zeros(1,N);
tauN_prod = zeros(1,N);
power_cov = zeros(1,N);
power_var = zeros(1,N);
power_std = zeros(1,N);
I_lower = zeros(1,N);
I_upper = zeros(1,N);
for i = 1:N
    
    % Estimating ϕ(v1)*w(v1,2) = P(v1)*w(v1,v2) and ϕ(v2)*w(v1,2) = P(v2)*w(v1,v2)
    phiw1 = phi1(1:i).*w(1:i);
    phiw2 = phi2(1:i).*w(1:i);
    % Estimating ϕ(v1,v2)*w(v1,v2) = P(v1,v2)*w(v1,v2) = P(v1)*P(v2)*w(v1,v2)
    phiw = phi_prod(1:i).*w(1:i);
    
    % 3 a) Estimating E[P(v1)+P(v2)] = E[P(v1)] + E[P(v2)]
    tauN1(i) = 1/i * sum(phiw1(1:i)); % E[P(v1)]
    tauN2(i) = 1/i * sum(phiw2(1:i)); % E[P(v2)]
    tauN(i) = tauN1(i) + tauN2(i); % E[P(v1)+P(v2)] = E[P(v1)] + E[P(v2)]
    % Estimating tauN_prod
    tauN_prod(i) = 1/i * sum(phiw);
    
    % 3 b)
    % Note: ϕ(v) = P(v)
    % Estimating Cov[P(v1),P(v2)] = E[P(v1)*P(v2)] - E[P(v1)]*E[P(v2)] =
    % = E[ϕ(v1)*ϕ(v2)] - E[ϕ(v1)]*E[ϕ(v2)] = E[ϕ(v1,v2)] - E[ϕ(v1)]*E[ϕ(v2)]
    % E[ϕ(v1)] = tauN1
    % E[ϕ(v2)] = tauN2
    % E[ϕ(v1,v2)] = tauN_joint
    power_cov(i) = tauN_prod(i) - tauN1(i).*tauN2(i);
    
    % 3 c) var
    % Estimating Var[P(v1) + P(v2)] = Var[P(v1)] + Var[P(v2)] + 2Cov[P(v1),P(v2)]
    power_var(i) = var(tauN1(1:i)) + var(tauN2(1:i)) + 2*abs(power_cov(i));

    % 3 c) std
    % Estimating standard deviation D[P(v1) + P(v2)]
    power_std(i) = sqrt(power_var(i));
   
    % Estimating confidence interval
    I_lower(i) = tauN(i) - lambda*sqrt(power_var(i)/i);
    I_upper(i) = tauN(i) + lambda*sqrt(power_var(i)/i);
    
end
tauN_dual_final = tauN(end)
I_width_dual_final = I_upper(end) - I_lower(end)
phiw_test = phiw1+phiw2;

a = plot(tauN./1000000); title('Importance sampling MC convergence'), ylabel('Power (MW)'), xlabel('Sample size');
hold on
b = plot(I_upper./1000000, 'Color', '#A2142F');
c = plot(I_lower./1000000, 'Color', '#A2142F');
hold off
%legend([a,b,c],'Expected amount of power', 'Upper bound', 'Lower bound')
legend([a,b],'Expected amount of power', 'Confidence interval')
% legend([a],'Expected amount of power')


%% 3d) one-dim Determining suitable new instrumental function g(x)

N = 6000;
xs = linspace(0,30, N);

% Jan, Determining suitable new instrumental function g(x)
r = 9.85;
lambda = 1.2;
g = gampdf(xs, r, lambda);

% Calculating pdf f(x)
f = wblpdf(xs, 9.13, 1.96);

% Calculating objective function ϕ(X)
phi = P(xs)';
phi = phi./100000000;

% Calculating ϕ(x)*f(x)
phif = phi.*f;
phif = phif.*22;

% Calculating new objective function ϕ(x)*f(x)/g(x)
% We aim to achieve the ratio ϕ(x)*f(x)/g(x) as constant as possible
phifg = phi.*f./g;

% Plot
a = plot(xs, f);
hold on
b = plot(xs, phi); 
c = plot(xs, phif, '--');
d = plot(xs, g, '--');
hold off
legend([a,b,c,d],'PDF, f(x)', 'Scaled old objective function, ϕ(x)', 'ϕ(x)*f(x)', 'Scaled g(x)')

% Plot
% c = plot(xs, phif, '--');
% hold on
% d = plot(xs, g, '--');
% hold off
% legend([c,d], 'ϕ(x)*f(x)', 'Scaled g(x)')

figure(2)
e = plot(xs, phifg);
legend(e,'ϕ(x)*f(x)/g(x)')

%% 3 d) 2-dim Determining suitable new instrumental function g(x)
a = 0.638;
p = 3;
q = 1.5;

N = 100;
xs = linspace(0,30, N);

% PDF bivariate function
f = @(v) wblpdf(v, 9.13, 1.96);
F = @(v) wblcdf(v, 9.13, 1.96);
f_2 = @(v1, v2) f(v1).*f(v2).*(1 + 0.638*(1-((F(v1)).^3)).^(1.5-1) .* (1-(F(v2)).^3).^(1.5-1) .* (((F(v1)).^3) .* (1+3*1.5) - 1) .* (((F(v2)).^3) .* (1+3*1.5) - 1));

% Determining suitable new instrumental function g(x)
r = 9.85;
lambda = 1.2;
g_first = @(v1) gampdf(v1, r, lambda);
g_second = @(v2) gampdf(v2, r, lambda);
g = @(v1, v2) g_first(v1).*g_second(v2);
g_rand = @(N) gamrnd(r, lambda, N, 1);

% Objective function ϕ(V)=P(V) ->  ϕ(V1,V2) = ϕ(V1)*ϕ(V2)= P(V1)*P(V2)
phi = @(v1,v2) (P(v1)'.*P(v2)'); %obs kanske ta bort "'"

% ϕ(v1,v2)*f_2(v1, v2)
phif = @(v1,v2) phi(v1,v2).*f_2(v1,v2);

% New objective function ϕ(v1,v2)*f(v1,v2)/g(v1,v2)
% We aim to achieve the ratio ϕ(v1,v2)*f(v1,v2)/g(v1,v2) as constant as possible
%phifg = phi.*f_2(x2,x2)./g_2(xs,xs);
phifg = @(v1,v2) phi(v1,v2).*f_2(v1,v2)./g(v1,v2);


[x,y] = meshgrid(xs,xs);
% Plot f_2(v1,v2)

% figure('Name','PDF f(v1,v2)','NumberTitle','off')
% Z = zeros(length(xs),length(xs));
% for i = 1:length(xs)
%     for j = 1:length(xs)
%         Z(i,j) = f_2(xs(i),xs(j));
%     end
% end
% surf(x,y, Z);

%plot ϕ(v1,v2)*f(v1, v2)
figure('Name','ϕ(v1,v2)*f(v1,v2)','NumberTitle','off')
Z = zeros(length(xs),length(xs));
for i = 1:length(xs)
    for j = 1:length(xs)
        Z(i,j) = phif(xs(i),xs(j));
    end
end
surf(x,y, Z);

% Plot g(v1,v2)
figure('Name','g(v1,v2))','NumberTitle','off')
Z = zeros(length(xs),length(xs));
for i = 1:length(xs)
    for j = 1:length(xs)
        Z(i,j) = g(xs(i),xs(j));
    end
end
surf(x, y, Z);

% Plot ϕ(v1,v2)*f(v1,v2)/g(v1,v2)
figure('Name','ϕ(v1,v2)*f(v1,v2)/g(v1,v2)','NumberTitle','off')
Z = zeros(length(xs),length(xs));
for i = 1:length(xs)
    for j = 1:length(xs)
        Z(i,j) = phifg(xs(i),xs(j));
    end
end
surf(x, y, Z);

%% 3 d)ALWAYS RUN SECTION ABOVE BEFORE
lambda = 1.96;
N = 6000;

X_1 = g_rand(6000);
X_2 = g_rand(6000);
X_dual = X_1.*X_2;

% Calculating ϕ(v1) and ϕ(v2)
phi1 = P(X1);
phi2 = P(X2);

% Calcualting importance weight w(v1,v2) = f(v1,v2)/g(v1,v2) (= f(v1,v2)/( g1(v1)*g2(v2) )
w_func = @(v1,v2) f_2(v1,v2)./g(v1,v2);
w = w_func(X1,X2);


% Estimating ϕ(v1)*w(v1,2) = P(v1)*w(v1,v2) and ϕ(v2)*w(v1,2) = P(v2)*w(v1,v2)
phiw1 = phi1.*w;
phiw2 = phi2.*w;
phiw = phiw1 + phiw2;

% Estimating E[P(v1)+P(v2)] = E[P(v1)] + E[P(v2)]
%tauN1 = mean(phiw1); % E[P(v1)]
%tauN2 = mean(phiw2); % E[P(v2)]
%tauN = tauN1 + tauN2; % E[P(v1)+P(v2)] = E[P(v1)] + E[P(v2)]


over = zeros(1,N);
under = zeros(1,N);
for i = 1:N
    if phiw(i) > 9.5*10^6
        over(i) = 1;
    elseif phiw(i) < 9.5*10^6
        under(i) = 1;
    end
end
p_over = mean(over)
p_under = mean(under)
p_sum = p_over + p_under
var_over = var(over)
var_under = var(under)
%{
As these stochastic variables either takes the form of a “1” or “0” they
are Bernoulli distributed stochastic variables. Using the fact that the
sum of Bernoulli distributed stochastic variables is a binomial distributed
stochastic variable
%}

F_bino = @(X, P) binocdf(X, N, P);
under_CI_upper = p_under + lambda*sqrt(p_under*(1-p_under)/N)
under_CI_lower = p_under - lambda*sqrt(p_under*(1-p_under)/N)
over_CI_upper = p_over + lambda*sqrt(p_over*(1-p_over)/N)
over_CI_lower = p_over - lambda*sqrt(p_over*(1-p_over)/N)

%% 3 d)  ALWAYS RUN SECTION ABOVE BEFORE
N = 6000;
% E[X + Y] = E[X] + E[Y]
% Var[X + Y] = Var[X] + Var[Y] + 2Cov[X,Y]
% lambda = 9.13;
% k = 1.96;

% Generate samples, now from g instead of f
% r1 = 10.07;
% lambda1 = 1.16;
% r2= 10.07;
% lambda2 = 1.16;
% X1 = gamrnd(r1, lambda1, N, 1); %We now sample from g instead of f
% X2 = gamrnd(r2, lambda2, N, 1); %We now sample from g instead of f
% OR ------------
%  u = [12.12 12.12];
%  sigma = [17 0 ; 0 17];
% X = mvnrnd(u, sigma, N);
X1 = g_rand(N);
X2 = g_rand(N);
% ----------

% Calculating ϕ(v1) and ϕ(v2)
phi1 = P(X1);
phi2 = P(X2);
% Calculating ϕ(v1,v2) = P(v1,v2) = P(v1)*P(v2)
phi_prod = P(X1).*P(X2);

% Calcualting importance weights w1(X1) and w2(X2)
% f1 = f(X1);
% f2 = f(X2);
% g1 = g_single(X1);
% g2 = g_single(X2);
% w1 = f1./g1;
% w2 = f2./g2;
% Calcualting importance weight w(v1,v2) = f(v1,v2)/g(v1,v2) (= f(v1,v2)/( g1(v1)*g2(v2) )
w_func = @(v1,v2) f_2(v1,v2)./g(v1,v2);
w = w_func(X1,X2);


lambda = 1.96;
tauN1 = zeros(1,N);
tauN2 = zeros(1,N);
tauN = zeros(1,N);
tauN_prod = zeros(1,N);
power_cov = zeros(1,N);
power_var = zeros(1,N);
power_std = zeros(1,N);
I_lower = zeros(1,N);
I_upper = zeros(1,N);
for i = 1:N
    
    % Estimating ϕ(v1)*w(v1,2) = P(v1)*w(v1,v2) and ϕ(v2)*w(v1,2) = P(v2)*w(v1,v2)
    phiw1 = phi1(1:i).*w(1:i);
    phiw2 = phi2(1:i).*w(1:i);
    % Estimating ϕ(v1,v2)*w(v1,v2) = P(v1,v2)*w(v1,v2) = P(v1)*P(v2)*w(v1,v2)
    phiw = phi_prod(1:i).*w(1:i);
    
    % 3 a) Estimating E[P(v1)+P(v2)] = E[P(v1)] + E[P(v2)]
    tauN1(i) = 1/i * sum(phiw1(1:i)); % E[P(v1)]
    tauN2(i) = 1/i * sum(phiw2(1:i)); % E[P(v2)]
    tauN(i) = tauN1(i) + tauN2(i); % E[P(v1)+P(v2)] = E[P(v1)] + E[P(v2)]
    % Estimating tauN_prod
    tauN_prod(i) = 1/i * sum(phiw);
    
    % 3 b)
    % Note: ϕ(v) = P(v)
    % Estimating Cov[P(v1),P(v2)] = E[P(v1)*P(v2)] - E[P(v1)]*E[P(v2)] =
    % = E[ϕ(v1)*ϕ(v2)] - E[ϕ(v1)]*E[ϕ(v2)] = E[ϕ(v1,v2)] - E[ϕ(v1)]*E[ϕ(v2)]
    % E[ϕ(v1)] = tauN1
    % E[ϕ(v2)] = tauN2
    % E[ϕ(v1,v2)] = tauN_joint
    power_cov(i) = tauN_prod(i) - tauN1(i).*tauN2(i);
    
    % 3 c) var
    % Estimating Var[P(v1) + P(v2)] = Var[P(v1)] + Var[P(v2)] + 2Cov[P(v1),P(v2)]
    power_var(i) = var(tauN1(1:i)) + var(tauN2(1:i)) + 2*abs(power_cov(i));

    % 3 c) std
    % Estimating standard deviation D[P(v1) + P(v2)]
    power_std(i) = sqrt(power_var(i));
   
    % Estimating confidence interval
    I_lower(i) = tauN(i) - lambda*sqrt(power_var(i)/i);
    I_upper(i) = tauN(i) + lambda*sqrt(power_var(i)/i);
    
end
tauN_dual_final = tauN(end)
I_width_dual_final = I_upper(end) - I_lower(end)
phiw_test = phiw1+phiw2;

a = plot(tauN./1000000); title('Importance sampling MC convergence'), ylabel('Power (MW)'), xlabel('Sample size');
hold on
b = plot(I_upper./1000000, 'Color', '#A2142F');
c = plot(I_lower./1000000, 'Color', '#A2142F');
hold off
%legend([a,b,c],'Expected amount of power', 'Upper bound', 'Lower bound')
legend([a,b],'Expected amount of power', 'Confidence interval')
% legend([a],'Expected amount of power')