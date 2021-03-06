%% 
gamma1 = 1;
gamma2 = 43/32;
gammad5plus = 1;
d = 2; % for problem 3-6

%% Problem 3
clear
d = 2;
N = 10000; % Number of random walks
temp = [0 0];

% n = 5; % Number of of steps
% testa för n = 1, 2, 3... Conclusion?

for n = 1:100

% Create empty cell matrix
for i = 1:N
   for j = 1:n
      x{j,i} = temp; 
   end
end

% Samping N random walks
for i = 1:N
    for j = 2:n
        
        r = randi(2*d,1);
        
        if r  == 1 
            x{j,i} = x{j-1, i} + [1 0];
        elseif r == 2  
            x{j,i} = x{j-1, i} + [-1 0];
        elseif r == 3
            x{j,i} = x{j-1, i} + [0 1];
        elseif r == 4
            x{j,i} = x{j-1, i} + [0 -1];
        end
    end
end


% Check if random walks are self avoiding
z = zeros(1,N);
for i = 1:N

   selfavoid = true;
   for j = 1:n-1
      if selfavoid == false
         break 
      end
      compare = x{j,i};
      
      for k = j+1:n
          if compare == x{k,i}
             selfavoid = false;
             z(i) = 1;
             break
          end
      end
      
   end
   
end

n
c_n2 = sum(z)/N

end


%% Problem 3 version 2
clear
d = 2;
N = 10000; % Number of random walks
n = 10; % Number of of steps, test for n = 1, 2, 3... Conclusion?

% Set initial values
% initialization of all particles
for i = 1:N
    x{i,1} = [0 0]; %All particles/walks start at the origin
end
w2 = ones(N,1);  %initialization of weights
z2 = ones(N,1); % 0 for NOT self avoiding, 1 for self avoiding
c_n = zeros(n,1);
g = 1;

for k = 2:n+1 %n, index 0 = 1, index 1 = 2, hence we start at k = 2 and go to n+1 for n steps
    
    for i = 1:N %N
        
        % Draw and set new values of particles
        r = randi(4,1); % uniform random draw of 1, 2, 3 or 4
        if r  == 1
            x{i,k} = x{i, k-1} + [1 0];
        elseif r == 2
            x{i,k} = x{i, k-1} + [-1 0];
        elseif r == 3
            x{i,k} = x{i, k-1} + [0 1];
        elseif r == 4
            x{i,k} = x{i, k-1} + [0 -1];
        end
        
        % Check if random walks are self avoiding
        z1 = z2; % copy old vector
        if z2(i) == 1 % Only check if we havent previously concluded that the walk isn't self avoiding
            selfavoid = true;
            compare = x{i,k};
            for kk = 1:k-1 % compare new value x{i,k} to all old ones
                if compare == x{i,kk}
                    selfavoid = false;
                    z2(i) = 0;
                    break
                end
            end
        end

    end
    % update weights
    g = 0.25*g; %g_(n+1)( x_(0:n+1 ) = g_(n+1)( x_(n+1)│x_(0:n) )*g_n( x_(0:n )
    w1 = w2; % Save old weights
    w2 = w1.*( z2./(z1*0.25) ); %update weights
    c_n(k-1) = sum(z2)/N; % c_n(2) = N_sa / N
end


%% Problem 3 version 3 FINAL
clear
d = 2;
N = 20000; % Number of random walks
n = 35; % Number of of steps, test for n = 1, 2, 3... Conclusion?

% Set initial values
% initialization of all particles
for i = 1:N
    x{i,1} = [0 0]; %All particles/walks start at the origin
end
z = ones(N,1); % 0 for NOT self avoiding, 1 for self avoiding
c_n = zeros(n,1);

for k = 2:n+1 %n, index 0 = 1, index 1 = 2 etc, hence we start at k = 2 -> n = 1 and go to n+1 for n steps
    
    for i = 1:N %N
        
        if z(i) == 1 %Only continue walks that are self-avoiding
            % Draw and set new values of particles
            r = randi(4,1); % uniform random draw of 1, 2, 3 or 4
            if r  == 1
                x{i,k} = x{i, k-1} + [1 0];
            elseif r == 2
                x{i,k} = x{i, k-1} + [-1 0];
            elseif r == 3
                x{i,k} = x{i, k-1} + [0 1];
            elseif r == 4
                x{i,k} = x{i, k-1} + [0 -1];
            end
            
            
            % Check if still self-avoiding
            selfavoid = true;
            compare = x{i,k};
            for kk = 1:k-1 % compare new value x{i,k} to all old ones
                if compare == x{i,kk}
                    selfavoid = false;
                    z(i) = 0;
                    break
                end
            end
        end
    end
    
    c_n(k-1) = (4^(k-1))*sum(z)/N; % c_n(2) = (4^n)*N_sa / N
end
c_temp = round(c_n, 0);
c_temp2 = [c_temp(1) c_temp(2) c_temp(3) c_temp(4) c_temp(5) c_temp(6) c_temp(7) c_temp(8) c_temp(9) c_temp(10) c_temp(13) c_temp(15) c_temp(20) c_temp(25) c_temp(30) c_temp(35)]';
c_true = [4 12 36 100 284 780 2172 5917 16268 44100 881500 6416596 897697164 123481354908 16741957935348 2252534077759844]';
error = (c_temp2 - c_true);
error2 = 1 - c_temp2./c_true
error_final = round(100.*error2, 2)

%% Problem 4 version 1 (Best version)
clear
d = 2;
N = 20000; % Number of random walks
n = 100; % Number of of steps, test for n = 1, 2, 3... Conclusion?

% Set initial values
% initialization of all particles
for i = 1:N
    x{i,1} = [0 0]; %All particles/walks start at the origin
end
w = ones(N,1);  %initialization of weights
z = ones(N,1); % 0 when walk has terminated, no possible neighbours
c_n = zeros(n,1);

for k = 2:n+1 %n, index 0 = 1, index 1 = 2, hence we start at k = 2 and go to n+1 for n steps
    n_iteration = k-1
    for i = 1:N %N
        
        if z(i) == 0 % If no possible neighbours, set values equal to old position
            x{i,k} = x{i, k-1};
            
        else % Check for free neighbours
            compare1 = x{i, k-1} + [1 0];
            compare2 = x{i, k-1} + [-1 0];
            compare3 = x{i, k-1} + [0 1];
            compare4 = x{i, k-1} + [0 -1];
            c1 = true;
            c2 = true;
            c3 = true;
            c4 = true;
            pos_neighbours = 4;
            success = false;
            for kk = 1:k-1 % compare new possible value x{i,k+1} to all old ones
                if compare1 == x{i,kk} & c1 == true
                    c1 = false;
                    pos_neighbours = pos_neighbours - 1;
                end
                if compare2 == x{i,kk} & c2 == true
                    c2 = false;
                    pos_neighbours = pos_neighbours - 1;
                end
                if compare3 == x{i,kk} & c3 == true
                    c3 = false;
                    pos_neighbours = pos_neighbours - 1;
                end
                if compare4 == x{i,kk} & c4 == true
                    c4 = false;
                    pos_neighbours = pos_neighbours -1;
                end
                
                if c1 == false && c2 == false && c3 == false && c4 == false % if no possible neighbours
                    z(i) = 0;
                    w(i) = 0;
                    x{i,k} = x{i, k-1};
                    success = true;
                    break
                end
            end
            
            while ~success %reaching this point, one of the neighbours must be free
                % Draw and set new values of particles
                r = randi(4,1); % uniform random draw of 1, 2, 3 or 4
                if r  == 1 && c1 == true
                    x{i,k} = compare1;
                    success = true;
                elseif r == 2 && c2 == true
                    x{i,k} = compare2;
                    success = true;
                elseif r == 3 && c3 == true
                    x{i,k} = compare3;
                    success = true;
                elseif r == 4 && c4 == true
                    x{i,k} = compare4;
                    success = true;
                end
            end
            
            w(i) = w(i)/(1/pos_neighbours);
            
        end %%%
        
    end
    c_n(k-1) = sum(w)/N; % c_n(2) = sum(w)/N
end
c_temp = round(c_n, 0);
c_temp2 = [c_temp(1) c_temp(2) c_temp(3) c_temp(4) c_temp(5) c_temp(6) c_temp(7) c_temp(8) c_temp(9) c_temp(10) c_temp(13) c_temp(15) c_temp(20) c_temp(25) c_temp(30) c_temp(35)]';
c_true = [4 12 36 100 284 780 2172 5917 16268 44100 881500 6416596 897697164 123481354908 16741957935348 2252534077759844]';
error = (c_temp2 - c_true);
error2 = 1 - c_temp2./c_true
error_final = round(100.*error2, 2)


%% Problem 4 version 2
clear
d = 2;
N = 20000; % Number of random walks
n = 35; % Number of of steps, test for n = 1, 2, 3... Conclusion?

% Set initial values
% initialization of all particles
for i = 1:N
    x{i,1} = [0 0]; %All particles/walks start at the origin
end
w = ones(N,1);  %initialization of weights
c_n = zeros(n,1);

for k = 2:n+1 %n, index 0 = 1, index 1 = 2, hence we start at k = 2 and go to n+1 for n steps
    n_iteration = k-1
    for i = 1:N %N
        
        if w(i) == 0 % If walk has already gotten stuck -> no possible neighbours, set values equal to old position
            x{i,k} = x{i, k-1};
        else % Check for free neighbours
            compare1 = x{i, k-1} + [1 0];
            compare2 = x{i, k-1} + [-1 0];
            compare3 = x{i, k-1} + [0 1];
            compare4 = x{i, k-1} + [0 -1];
            c1 = true;
            c2 = true;
            c3 = true;
            c4 = true;
            pos_neighbours = 4;
            terminate = false;
            
            for kk = 1:k-1 % compare new possible value x{i,k+1} to all old ones
                if compare1 == x{i,kk} & c1 == true
                    c1 = false;
                    pos_neighbours = pos_neighbours - 1;
                end
                if compare2 == x{i,kk} & c2 == true
                    c2 = false;
                    pos_neighbours = pos_neighbours - 1;
                end
                if compare3 == x{i,kk} & c3 == true
                    c3 = false;
                    pos_neighbours = pos_neighbours - 1;
                end
                if compare4 == x{i,kk} & c4 == true
                    c4 = false;
                    pos_neighbours = pos_neighbours -1;
                end
                
                if c1 == false && c2 == false && c3 == false && c4 == false % if no possible neighbours
                    w(i) = 0;
                    x{i,k} = x{i, k-1};
                    terminate = true;
                    break
                end
            end
            
            if ~terminate
                % Reaching this point, one of the neighbours must be free
                % Draw and set new values of particles
                sampler = [c1 2*c2 3*c3 4*c4];
                sampler = nonzeros(sampler); % Remove already visited neighbours
                r = randsample(sampler,1); % uniform random draw of moving to free neighbours
                if r  == 1
                    x{i,k} = compare1;
                elseif r == 2
                    x{i,k} = compare2;
                elseif r == 3
                    x{i,k} = compare3;
                elseif r == 4
                    x{i,k} = compare4;
                end
                w(i) = w(i)/(1/pos_neighbours);
            end
        end
    end
    c_n(k-1) = sum(w)/N; % c_n(2) = sum(w)/N
end
c_temp = round(c_n, 0);
c_temp2 = [c_temp(1) c_temp(2) c_temp(3) c_temp(4) c_temp(5) c_temp(6) c_temp(7) c_temp(8) c_temp(9) c_temp(10) c_temp(13) c_temp(15) c_temp(20) c_temp(25) c_temp(30) c_temp(35)]';
c_true = [4 12 36 100 284 780 2172 5917 16268 44100 881500 6416596 897697164 123481354908 16741957935348 2252534077759844]';
error = (c_temp2 - c_true);
error2 = 1 - c_temp2./c_true
error_final = round(100.*error2, 2)


%% Problem 5
clear
d = 2;
N = 20000; % Number of random walks
n = 150; % Number of of steps, test for n = 1, 2, 3... Conclusion?

% Set initial values
% initialization of all particles
for i = 1:N
    x{i,1} = [0 0]; %All particles/walks start at the origin
end
w = ones(N,1);  %initialization of weights
c_n = zeros(n,1);

for k = 2:n+1 %n, index 0 = 1, index 1 = 2, hence we start at k = 2 and go to n+1 for n steps
    n_iteration = k-1
    w = ones(N,1); %reset weights, i.e. set equal weights
    for i = 1:N %N
  
        % Check for free neighbours
        compare1 = x{i, k-1} + [1 0];
        compare2 = x{i, k-1} + [-1 0];
        compare3 = x{i, k-1} + [0 1];
        compare4 = x{i, k-1} + [0 -1];
        c1 = true;
        c2 = true;
        c3 = true;
        c4 = true;
        pos_neighbours = 4;
        terminate = false;
        
        for kk = 1:k-1 % compare new possible value x{i,k+1} to all old ones
            if compare1 == x{i,kk} & c1 == true
                c1 = false;
                pos_neighbours = pos_neighbours - 1;
            end
            if compare2 == x{i,kk} & c2 == true
                c2 = false;
                pos_neighbours = pos_neighbours - 1;
            end
            if compare3 == x{i,kk} & c3 == true
                c3 = false;
                pos_neighbours = pos_neighbours - 1;
            end
            if compare4 == x{i,kk} & c4 == true
                c4 = false;
                pos_neighbours = pos_neighbours -1;
            end
            
            if c1 == false && c2 == false && c3 == false && c4 == false % if no possible neighbours
                w(i) = 0;
                x{i,k} = x{i, k-1};
                terminate = true;
                break
            end
        end
        
        if ~terminate
            % Reaching this point, one of the neighbours must be free
            % Draw and set new values of particles
            sampler = [c1 2*c2 3*c3 4*c4];
            sampler = nonzeros(sampler); % Remove already visited neighbours
            r = randsample(sampler,1); % uniform random draw of moving to free neighbours
            if r  == 1
                x{i,k} = compare1;
            elseif r == 2
                x{i,k} = compare2;
            elseif r == 3
                x{i,k} = compare3;
            elseif r == 4
                x{i,k} = compare4;
            end
            w(i) = w(i)/(1/pos_neighbours);
        end
    end
    
    ind = randsample(N,N,true,w); % Selection
    x = x(ind,:);
    
    if k == 2
        c_n(k-1) = sum(w)/N; %we dont include c_0(2) = 1 in our vector, this we modify for this case
    else
        c_n(k-1) = c_n(k-2)*sum(w)/N;
    end
    clear w %toss old weights
end
c_temp = round(c_n, 0);
c_temp2 = [c_temp(1) c_temp(2) c_temp(3) c_temp(4) c_temp(5) c_temp(6) c_temp(7) c_temp(8) c_temp(9) c_temp(10) c_temp(13) c_temp(15) c_temp(20) c_temp(25) c_temp(30) c_temp(35)]';
c_true = [4 12 36 100 284 780 2172 5917 16268 44100 881500 6416596 897697164 123481354908 16741957935348 2252534077759844]';
error = (c_temp2 - c_true);
error2 = 1 - c_temp2./c_true
error_final = round(100.*error2, 2)

c_100 = c_n;
%% problem 6 test
n = 100;

% Forming y
z = zeros(n,1);
for i = 1:n
    z(i) = log(i);
end
y = [log(c_n(1:n))+z]; % 100x1

% Forming X
X1 = ones(n,1);
X2 = zeros(n,1);
for i = 1:n
   X2(i) = i; 
end
X3 = z;
X = [X1 X2 X3]; %100x3,  X': 3x100

% Calculation
%theta = inv(X'*X)*X'*y
theta = (X'*X)\X'*y

% Estimates
A_2 = exp(theta(1))
u_2 = exp(theta(2))
gamma_2 = theta(3)



%% Problem 6 final
N = 20000; % Number of random walks
n = 150; % Number of of steps, test for n = 1, 2, 3... Conclusion?

% Vectors to store estimated values in
A_2v = zeros(1,10);
u_2v = zeros(1,10);
gamma_2v = zeros(1,10);

for q = 1:10
    q
    
    
    % Set initial values
    % initialization of all particles
    for i = 1:N
        x{i,1} = [0 0]; %All particles/walks start at the origin
    end
    w = ones(N,1);  %initialization of weights
    c_n = zeros(n,1);
    
    for k = 2:n+1 %n, index 0 = 1, index 1 = 2, hence we start at k = 2 and go to n+1 for n steps
        %     q
        %     n_iteration = k-1
        w = ones(N,1); %reset weights, i.e. set equal weights
        for i = 1:N %N
            
            % Check for free neighbours
            compare1 = x{i, k-1} + [1 0];
            compare2 = x{i, k-1} + [-1 0];
            compare3 = x{i, k-1} + [0 1];
            compare4 = x{i, k-1} + [0 -1];
            c1 = true;
            c2 = true;
            c3 = true;
            c4 = true;
            pos_neighbours = 4;
            terminate = false;
            
            for kk = 1:k-1 % compare new possible value x{i,k+1} to all old ones
                if compare1 == x{i,kk} & c1 == true
                    c1 = false;
                    pos_neighbours = pos_neighbours - 1;
                end
                if compare2 == x{i,kk} & c2 == true
                    c2 = false;
                    pos_neighbours = pos_neighbours - 1;
                end
                if compare3 == x{i,kk} & c3 == true
                    c3 = false;
                    pos_neighbours = pos_neighbours - 1;
                end
                if compare4 == x{i,kk} & c4 == true
                    c4 = false;
                    pos_neighbours = pos_neighbours -1;
                end
                
                if c1 == false && c2 == false && c3 == false && c4 == false % if no possible neighbours
                    w(i) = 0;
                    x{i,k} = x{i, k-1};
                    terminate = true;
                    break
                end
            end
            
            if ~terminate
                % Reaching this point, one of the neighbours must be free
                % Draw and set new values of particles
                sampler = [c1 2*c2 3*c3 4*c4];
                sampler = nonzeros(sampler); % Remove already visited neighbours
                r = randsample(sampler,1); % uniform random draw of moving to free neighbours
                if r  == 1
                    x{i,k} = compare1;
                elseif r == 2
                    x{i,k} = compare2;
                elseif r == 3
                    x{i,k} = compare3;
                elseif r == 4
                    x{i,k} = compare4;
                end
                w(i) = w(i)/(1/pos_neighbours);
            end
        end
        
        ind = randsample(N,N,true,w); % Selection
        x = x(ind,:);
        
        if k == 2
            c_n(k-1) = sum(w)/N; %we dont include c_0(2) = 1 in our vector, this we modify for this case
        else
            c_n(k-1) = c_n(k-2)*sum(w)/N;
        end
        clear w %toss old weights
    end
    
    % Forming y
    z = zeros(n,1);
    for i = 1:n
        z(i) = log(i);
    end
    y = [log(c_n(1:n))+z]; % 100x1
    
    % Estimate values for problem 6
    
    % Forming X
    X1 = ones(n,1);
    X2 = zeros(n,1);
    for i = 1:n
        X2(i) = i;
    end
    X3 = z;
    X = [X1 X2 X3]; %100x3,  X': 3x100
    
    % Calculation
    %theta = inv(X'*X)*X'*y
    theta = (X'*X)\X'*y
    
    % Estimates
    A_2 = exp(theta(1));
    u_2 = exp(theta(2));
    gamma_2 = theta(3);
    
    % Store estimates in vector
    A_2v(q) = A_2;
    u_2v(q) = u_2;
    gamma_2v(q) = gamma_2;
    
end

final_results = [A_2v' u_2v' gamma_2v']

%% Problem 6 tweak for tables

table = round(final_results, 5)

meanA = round(mean(final_results(:,1)),5)
meanu = round(mean(final_results(:,2)),5)
meangamma = round(mean(final_results(:,3)),5)

varA = round((10^5)*var(final_results(:,1)),5)
varu = round((10^5)*var(final_results(:,2)),5)
vargamma = round((10^5)*var(final_results(:,3)),5)


%% Problem 9, d = 3
N = 20000; % Number of random walks
n = 100; % Number of of steps, test for n = 1, 2, 3... Conclusion?
d = 3;

% Vectors to store estimated values in
A_2v = zeros(1,10);
u_2v = zeros(1,10);
gamma_2v = zeros(1,10);

for q = 1:10
    q
    
    
% Set initial values
% initialization of all particles
for i = 1:N
    x{i,1} = [zeros(1,d)]; %All particles/walks start at the origin
end
w = ones(N,1);  %initialization of weights
c_n = zeros(n,1);

for k = 2:n+1 %n, index 0 = 1, index 1 = 2, hence we start at k = 2 and go to n+1 for n steps
    n_iteration = k-1
    w = ones(N,1); %reset weights, i.e. set equal weights
    for i = 1:N %N
  
        % Check for free neighbours
        compare1 = x{i, k-1} + [1 0 0];
        compare2 = x{i, k-1} + [-1 0 0];
        compare3 = x{i, k-1} + [0 1 0];
        compare4 = x{i, k-1} + [0 -1 0];
        compare5 = x{i, k-1} + [0 0 1];
        compare6 = x{i, k-1} + [0 0 -1];
        c1 = true;
        c2 = true;
        c3 = true;
        c4 = true;
        c5 = true;
        c6 = true;
        pos_neighbours = 2*d;
        terminate = false;
        
        for kk = 1:k-1 % compare new possible value x{i,k+1} to all old ones
            if compare1 == x{i,kk} & c1 == true
                c1 = false;
                pos_neighbours = pos_neighbours - 1;
            end
            if compare2 == x{i,kk} & c2 == true
                c2 = false;
                pos_neighbours = pos_neighbours - 1;
            end
            if compare3 == x{i,kk} & c3 == true
                c3 = false;
                pos_neighbours = pos_neighbours - 1;
            end
            if compare4 == x{i,kk} & c4 == true
                c4 = false;
                pos_neighbours = pos_neighbours -1;
            end
            if compare5 == x{i,kk} & c5 == true
                c5 = false;
                pos_neighbours = pos_neighbours -1;
            end
            if compare6 == x{i,kk} & c6 == true
                c6 = false;
                pos_neighbours = pos_neighbours -1;
            end
            
            if c1 == false && c2 == false && c3 == false && c4 == false && c5 == false && c6 == false % if no possible neighbours
                w(i) = 0;
                x{i,k} = x{i, k-1};
                terminate = true;
                break
            end
        end
        
        if ~terminate
            % Reaching this point, one of the neighbours must be free
            % Draw and set new values of particles
            sampler = [c1 2*c2 3*c3 4*c4 5*c5 6*c6];
            sampler = nonzeros(sampler); % Remove already visited neighbours
            r = randsample(sampler,1); % uniform random draw of moving to free neighbours
            if r  == 1
                x{i,k} = compare1;
            elseif r == 2
                x{i,k} = compare2;
            elseif r == 3
                x{i,k} = compare3;
            elseif r == 4
                x{i,k} = compare4;
            elseif r == 5
                x{i,k} = compare5;
            elseif r == 6
                x{i,k} = compare6;
            end
            w(i) = w(i)/(1/pos_neighbours);
        end
    end
    
    ind = randsample(N,N,true,w); % Selection
    x = x(ind,:);
    
    if k == 2
        c_n(k-1) = sum(w)/N; %we dont include c_0(2) = 1 in our vector, this we modify for this case
    else
        c_n(k-1) = c_n(k-2)*sum(w)/N;
    end
    clear w %toss old weights
end
    
    % Forming y
    z = zeros(n,1);
    for i = 1:n
        z(i) = log(i);
    end
    y = [log(c_n(1:n))+z]; % 100x1
    
    % Estimate values for problem 6
    
    % Forming X
    X1 = ones(n,1);
    X2 = zeros(n,1);
    for i = 1:n
        X2(i) = i;
    end
    X3 = z;
    X = [X1 X2 X3]; %100x3,  X': 3x100
    
    % Calculation
    %theta = inv(X'*X)*X'*y
    theta = (X'*X)\X'*y
    
    % Estimates
    A_2 = exp(theta(1));
    u_2 = exp(theta(2));
    gamma_2 = theta(3);
    
    % Store estimates in vector
    A_2v(q) = A_2;
    u_2v(q) = u_2;
    gamma_2v(q) = gamma_2;
    
end

final_results = [A_2v' u_2v' gamma_2v']
finA3 = final_results;
%% Problem 9 d = 3, tweak for tables

table = round(final_results, 5)

meanA = round(mean(final_results(:,1)),5)
meanu = round(mean(final_results(:,2)),5)
meangamma = round(mean(final_results(:,3)),5)

varA = round((10^5)*var(final_results(:,1)),5)
varu = round((10^5)*var(final_results(:,2)),5)
vargamma = round((10^5)*var(final_results(:,3)),5)


%% Problem 9, d = 5
N = 20000; % Number of random walks
n = 100; % Number of of steps, test for n = 1, 2, 3... Conclusion?
d = 5;

% Vectors to store estimated values in
A_2v = zeros(1,10);
u_2v = zeros(1,10);
gamma_2v = zeros(1,10);

for q = 1:10
    q
    
    
% Set initial values
% initialization of all particles
for i = 1:N
    x{i,1} = [zeros(1,d)]; %All particles/walks start at the origin
end
w = ones(N,1);  %initialization of weights
c_n = zeros(n,1);

for k = 2:n+1 %n, index 0 = 1, index 1 = 2, hence we start at k = 2 and go to n+1 for n steps
    n_iteration = k-1
    w = ones(N,1); %reset weights, i.e. set equal weights
    for i = 1:N %N
  
        % Check for free neighbours
        compare1 = x{i, k-1} + [1 0 0 0 0];
        compare2 = x{i, k-1} + [-1 0 0 0 0];
        compare3 = x{i, k-1} + [0 1 0 0 0];
        compare4 = x{i, k-1} + [0 -1 0 0 0];
        compare5 = x{i, k-1} + [0 0 1 0 0];
        compare6 = x{i, k-1} + [0 0 -1 0 0];
        compare7 = x{i, k-1} + [0 0 0 1 0];
        compare8 = x{i, k-1} + [0 0 0 -1 0];
        compare9 = x{i, k-1} + [0 0 0 0 1];
        compare10 = x{i, k-1} + [0 0 0 0 -1];
        c1 = true;
        c2 = true;
        c3 = true;
        c4 = true;
        c5 = true;
        c6 = true;
        c7 = true;
        c8 = true;
        c9 = true;
        c10 = true;
        pos_neighbours = 2*d;
        terminate = false;
        
        for kk = 1:k-1 % compare new possible value x{i,k+1} to all old ones
            if compare1 == x{i,kk} & c1 == true
                c1 = false;
                pos_neighbours = pos_neighbours - 1;
            end
            if compare2 == x{i,kk} & c2 == true
                c2 = false;
                pos_neighbours = pos_neighbours - 1;
            end
            if compare3 == x{i,kk} & c3 == true
                c3 = false;
                pos_neighbours = pos_neighbours - 1;
            end
            if compare4 == x{i,kk} & c4 == true
                c4 = false;
                pos_neighbours = pos_neighbours -1;
            end
            if compare5 == x{i,kk} & c5 == true
                c5 = false;
                pos_neighbours = pos_neighbours -1;
            end
            if compare6 == x{i,kk} & c6 == true
                c6 = false;
                pos_neighbours = pos_neighbours -1;
            end
            if compare7 == x{i,kk} & c7 == true
                c7 = false;
                pos_neighbours = pos_neighbours -1;
            end
            if compare8 == x{i,kk} & c8 == true
                c8 = false;
                pos_neighbours = pos_neighbours -1;
            end
            if compare9 == x{i,kk} & c9 == true
                c9 = false;
                pos_neighbours = pos_neighbours -1;
            end
            if compare10 == x{i,kk} & c10 == true
                c10 = false;
                pos_neighbours = pos_neighbours -1;
            end
            
            if c1 == false && c2 == false && c3 == false && c4 == false && c5 == false && c6 == false && c7 == false && c8 == false && c9 == false && c10 == false% if no possible neighbours
                w(i) = 0;
                x{i,k} = x{i, k-1};
                terminate = true;
                break
            end
        end
        
        if ~terminate
            % Reaching this point, one of the neighbours must be free
            % Draw and set new values of particles
            sampler = [c1 2*c2 3*c3 4*c4 5*c5 6*c6 7*c7 8*c8 9*c9 10*c10];
            sampler = nonzeros(sampler); % Remove already visited neighbours
            r = randsample(sampler,1); % uniform random draw of moving to free neighbours
            if r  == 1
                x{i,k} = compare1;
            elseif r == 2
                x{i,k} = compare2;
            elseif r == 3
                x{i,k} = compare3;
            elseif r == 4
                x{i,k} = compare4;
            elseif r == 5
                x{i,k} = compare5;
            elseif r == 6
                x{i,k} = compare6;
            elseif r == 7
                x{i,k} = compare7;
            elseif r == 8
                x{i,k} = compare8;
            elseif r == 9
                x{i,k} = compare9;
            elseif r == 10
                x{i,k} = compare10;
            end
            w(i) = w(i)/(1/pos_neighbours);
        end
    end
    
    ind = randsample(N,N,true,w); % Selection
    x = x(ind,:);
    
    if k == 2
        c_n(k-1) = sum(w)/N; %we dont include c_0(2) = 1 in our vector, this we modify for this case
    else
        c_n(k-1) = c_n(k-2)*sum(w)/N;
    end
    clear w %toss old weights
end

% Forming y
z = zeros(n,1);
for i = 1:n
    z(i) = log(i);
end
y = [log(c_n(1:n))+z]; % 100x1

% Estimate values for problem 6

% Forming X
X1 = ones(n,1);
X2 = zeros(n,1);
for i = 1:n
    X2(i) = i;
end
X3 = z;
X = [X1 X2 X3]; %100x3,  X': 3x100

% Calculation
%theta = inv(X'*X)*X'*y
theta = (X'*X)\X'*y

% Estimates
A_2 = exp(theta(1));
u_2 = exp(theta(2));
gamma_2 = theta(3);

% Store estimates in vector
A_2v(q) = A_2;
u_2v(q) = u_2;
gamma_2v(q) = gamma_2;

end

final_results = [A_2v' u_2v' gamma_2v']
finA5 = final_results;
%% Problem 9, d = 5 tweak for tables

table = round(final_results, 5)

meanA = round(mean(final_results(:,1)),5)
meanu = round(mean(final_results(:,2)),5)
meangamma = round(mean(final_results(:,3)),5)

varA = round((10^5)*var(final_results(:,1)),5)
varu = round((10^5)*var(final_results(:,2)),5)
vargamma = round((10^5)*var(final_results(:,3)),5)


%% Problem 9, d = 10
N = 20000; % Number of random walks
n = 100; % Number of of steps, test for n = 1, 2, 3... Conclusion?
d = 10;

% Vectors to store estimated values in
A_2v = zeros(1,10);
u_2v = zeros(1,10);
gamma_2v = zeros(1,10);

for q = 1:10
    q
    
    
    % Set initial values
    % initialization of all particles
    for i = 1:N
        x{i,1} = [zeros(1,d)]; %All particles/walks start at the origin
    end
    w = ones(N,1);  %initialization of weights
    c_n = zeros(n,1);
    
    for k = 2:n+1 %n, index 0 = 1, index 1 = 2, hence we start at k = 2 and go to n+1 for n steps
        n_iteration = k-1
        w = ones(N,1); %reset weights, i.e. set equal weights
        for i = 1:N %N
            
            % Check for free neighbours
            compare1 = x{i, k-1} + [1 0 0 0 0 0 0 0 0 0];
            compare2 = x{i, k-1} + [-1 0 0 0 0 0 0 0 0 0];
            compare3 = x{i, k-1} + [0 1 0 0 0 0 0 0 0 0];
            compare4 = x{i, k-1} + [0 -1 0 0 0 0 0 0 0 0];
            compare5 = x{i, k-1} + [0 0 1 0 0 0 0 0 0 0];
            compare6 = x{i, k-1} + [0 0 -1 0 0 0 0 0 0 0];
            compare7 = x{i, k-1} + [0 0 0 1 0 0 0 0 0 0];
            compare8 = x{i, k-1} + [0 0 0 -1 0 0 0 0 0 0];
            compare9 = x{i, k-1} + [0 0 0 0 1 0 0 0 0 0];
            compare10 = x{i, k-1} + [0 0 0 0 -1 0 0 0 0 0];
            compare11 = x{i, k-1} + [0 0 0 0 0 1 0 0 0 0];
            compare12 = x{i, k-1} + [0 0 0 0 0 -1 0 0 0 0];
            compare13 = x{i, k-1} + [0 0 0 0 0 0 1 0 0 0];
            compare14 = x{i, k-1} + [0 0 0 0 0 0 -1 0 0 0];
            compare15 = x{i, k-1} + [0 0 0 0 0 0 0 1 0 0];
            compare16 = x{i, k-1} + [0 0 0 0 0 0 0 -1 0 0];
            compare17 = x{i, k-1} + [0 0 0 0 0 0 0 0 1 0];
            compare18 = x{i, k-1} + [0 0 0 0 0 0 0 0 -1 0];
            compare19 = x{i, k-1} + [0 0 0 0 0 0 0 0 0 1];
            compare20 = x{i, k-1} + [0 0 0 0 0 0 0 0 0 -1];
            c1 = true;
            c2 = true;
            c3 = true;
            c4 = true;
            c5 = true;
            c6 = true;
            c7 = true;
            c8 = true;
            c9 = true;
            c10 = true;
            c11 = true;
            c12 = true;
            c13 = true;
            c14 = true;
            c15 = true;
            c16 = true;
            c17 = true;
            c18 = true;
            c19 = true;
            c20 = true;
            pos_neighbours = 2*d;
            terminate = false;
            
            for kk = 1:k-1 % compare new possible value x{i,k+1} to all old ones
                if compare1 == x{i,kk} & c1 == true
                    c1 = false;
                    pos_neighbours = pos_neighbours - 1;
                end
                if compare2 == x{i,kk} & c2 == true
                    c2 = false;
                    pos_neighbours = pos_neighbours - 1;
                end
                if compare3 == x{i,kk} & c3 == true
                    c3 = false;
                    pos_neighbours = pos_neighbours - 1;
                end
                if compare4 == x{i,kk} & c4 == true
                    c4 = false;
                    pos_neighbours = pos_neighbours -1;
                end
                if compare5 == x{i,kk} & c5 == true
                    c5 = false;
                    pos_neighbours = pos_neighbours -1;
                end
                if compare6 == x{i,kk} & c6 == true
                    c6 = false;
                    pos_neighbours = pos_neighbours -1;
                end
                if compare7 == x{i,kk} & c7 == true
                    c7 = false;
                    pos_neighbours = pos_neighbours -1;
                end
                if compare8 == x{i,kk} & c8 == true
                    c8 = false;
                    pos_neighbours = pos_neighbours -1;
                end
                if compare9 == x{i,kk} & c9 == true
                    c9 = false;
                    pos_neighbours = pos_neighbours -1;
                end
                if compare10 == x{i,kk} & c10 == true
                    c10 = false;
                    pos_neighbours = pos_neighbours -1;
                end
                if compare11 == x{i,kk} & c11 == true
                    c11 = false;
                    pos_neighbours = pos_neighbours -1;
                end
                if compare12 == x{i,kk} & c12 == true
                    c12 = false;
                    pos_neighbours = pos_neighbours -1;
                end
                if compare13 == x{i,kk} & c13 == true
                    c13 = false;
                    pos_neighbours = pos_neighbours -1;
                end
                if compare14 == x{i,kk} & c14 == true
                    c14 = false;
                    pos_neighbours = pos_neighbours -1;
                end
                if compare15 == x{i,kk} & c15 == true
                    c15 = false;
                    pos_neighbours = pos_neighbours -1;
                end
                if compare16 == x{i,kk} & c16 == true
                    c16 = false;
                    pos_neighbours = pos_neighbours -1;
                end
                if compare17 == x{i,kk} & c17 == true
                    c17 = false;
                    pos_neighbours = pos_neighbours -1;
                end
                if compare18 == x{i,kk} & c18 == true
                    c18 = false;
                    pos_neighbours = pos_neighbours -1;
                end
                if compare19 == x{i,kk} & c19 == true
                    c19 = false;
                    pos_neighbours = pos_neighbours -1;
                end
                if compare20 == x{i,kk} & c20 == true
                    c20 = false;
                    pos_neighbours = pos_neighbours -1;
                end
                
                if c1 == false && c2 == false && c3 == false && c4 == false && c5 == false && c6 == false && c7 == false && c8 == false && c9 == false && c10 == false && c11 == false && c12 == false && c13 == false && c14 == false && c15 == false && c16 == false && c17 == false && c18 == false && c19 == false && c20 == false% if no possible neighbours
                    w(i) = 0;
                    x{i,k} = x{i, k-1};
                    terminate = true;
                    break
                end
            end
            
            if ~terminate
                % Reaching this point, one of the neighbours must be free
                % Draw and set new values of particles
                sampler = [c1 2*c2 3*c3 4*c4 5*c5 6*c6 7*c7 8*c8 9*c9 10*c10 11*c11 12*c12 13*c13 14*c14 15*c15 16*c16 17*c17 18*c18 19*c19 20*c20];
                sampler = nonzeros(sampler); % Remove already visited neighbours
                r = randsample(sampler,1); % uniform random draw of moving to free neighbours
                if r  == 1
                    x{i,k} = compare1;
                elseif r == 2
                    x{i,k} = compare2;
                elseif r == 3
                    x{i,k} = compare3;
                elseif r == 4
                    x{i,k} = compare4;
                elseif r == 5
                    x{i,k} = compare5;
                elseif r == 6
                    x{i,k} = compare6;
                elseif r == 7
                    x{i,k} = compare7;
                elseif r == 8
                    x{i,k} = compare8;
                elseif r == 9
                    x{i,k} = compare9;
                elseif r == 10
                    x{i,k} = compare10;
                elseif r == 11
                    x{i,k} = compare11;
                elseif r == 12
                    x{i,k} = compare12;
                elseif r == 13
                    x{i,k} = compare13;
                elseif r == 14
                    x{i,k} = compare14;
                elseif r == 15
                    x{i,k} = compare15;
                elseif r == 16
                    x{i,k} = compare16;
                elseif r == 17
                    x{i,k} = compare17;
                elseif r == 18
                    x{i,k} = compare18;
                elseif r == 19
                    x{i,k} = compare19;
                elseif r == 20
                    x{i,k} = compare20;
                end
                w(i) = w(i)/(1/pos_neighbours);
            end
        end
        
        ind = randsample(N,N,true,w); % Selection
        x = x(ind,:);
        
        if k == 2
            c_n(k-1) = sum(w)/N; %we dont include c_0(2) = 1 in our vector, this we modify for this case
        else
            c_n(k-1) = c_n(k-2)*sum(w)/N;
        end
        clear w %toss old weights
    end
    
    % Forming y
    z = zeros(n,1);
    for i = 1:n
        z(i) = log(i);
    end
    y = [log(c_n(1:n))+z]; % 100x1
    
    % Estimate values for problem 6
    
    % Forming X
    X1 = ones(n,1);
    X2 = zeros(n,1);
    for i = 1:n
        X2(i) = i;
    end
    X3 = z;
    X = [X1 X2 X3]; %100x3,  X': 3x100
    
    % Calculation
    %theta = inv(X'*X)*X'*y
    theta = (X'*X)\X'*y
    
    % Estimates
    A_2 = exp(theta(1));
    u_2 = exp(theta(2));
    gamma_2 = theta(3);
    
    % Store estimates in vector
    A_2v(q) = A_2;
    u_2v(q) = u_2;
    gamma_2v(q) = gamma_2;
    
end

final_results = [A_2v' u_2v' gamma_2v']
finA10 = final_results;
%% Problem 9, d = 10 tweak for tables

table = round(final_results, 5)

meanA = round(mean(final_results(:,1)),5)
meanu = round(mean(final_results(:,2)),5)
meangamma = round(mean(final_results(:,3)),5)

varA = round((10^5)*var(final_results(:,1)),5)
varu = round((10^5)*var(final_results(:,2)),5)
vargamma = round((10^5)*var(final_results(:,3)),5)