%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MAIN SCRIPT FILE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Estimate Rust(1987)'s Single Agent Dynamic Discrete Choice Model 
%
% Estimate the model using the Nested Fixed Point Algorithm by "fmincon" 
% with first-order analytic derivatives.
%
%yCAUTIONz
% First need to specify the TRUE values of structural parameters 
% as well as the discount factor in the data generating process. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    

% Variable declaration
global beta N M x xt dt nT nBus 
global indices PayoffDiffPrime TransProbPrime CbEVPrime RPrime
global EVold BellEval tol_inner

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load the model primitive and true parameter values listed below
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load truethetaEV_beta975.mat;

% Data description
% beta = 0.975;                                         % discount factor ƒÀ
% nT = 120;                                             % # of periods (months)
% nBus = 50;                                            % # of buses
% N = 175;                                              % # of discretized points in mileage state
% M = 5;                                                % number of mileage transition states
% RC = 11.7257;                                         % Replacement Cost(RC)
% thetaCost = 2.4569;                                   % maintanence cost parameter = ƒÆ11 (c(x) = 0.001*thetaCost*x : linear)
% thetaProbs = [ 0.0937 0.4475 0.4459 0.0127 0.0002 ]'  % mileage transition probabilities = [ƒÆ30, ƒÆ31, ƒÆ32, ƒÆ33, ƒÆ34]
% EV                                                    % 175*1 vector

%yCAUTIONz
% We obtain the value of EV by solving the Bellman equation parameterized by true parameter values.
% If you change the parameter values or primitive of the model, you need to resolve the Bellman equation for the new EV's.

% True parameter values
% = Parameter estimates in Rust(1987) (in Table X, page 1022)
thetatrue = [thetaCost; thetaProbs; RC];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SIMULATE DATA (state, decision) = (xt,dt)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate data on states and decisions for "nBus" buses and "nT" periods

% Discrete mileage state : 175*1 vector
x = (1:N)';

% the probability of performing regular maintenance at state x[i] : 175*1 vector
P0 = 1./ (1 + exp( 0.001*thetaCost*x - beta.*EV - RC - 0.001*thetaCost*x(1)+ beta*EV(1))); 

rand_seed = 100;
rand('seed',rand_seed);

% Arrays for states and decisions   
Rx  = unifrnd(0, 1, nT, nBus);      % nT*nBus uniform random number matrix
Rd  = unifrnd(0, 1, nT, nBus);      % nT*nBus uniform random number matrix
xt = ones(nT, nBus);                % State : nT*nBus one matrix
dx = zeros(nT, nBus);               % Increments of state : nT*nBus zero matrix
dt = zeros(nT, nBus);               % Decision : nT*nBus zero matrix
cumTransProb = cumsum(thetaProbs);  % Cumlative sum of thetaProbs : 5*1 vector

% Generate states and decisions
for t = 1:nT
    dt(t,:) = (Rd(t,:) >= P0(xt(t,:))');
    for i = 1:nBus
        dx(t,i) = find(Rx(t,i) < cumTransProb,1);
        if t < nT
            if dt(t,i) == 1
               xt(t+1,i) = 1 + dx(t,i)-1;
            else 
               xt(t+1,i) = min(xt(t,i) + dx(t,i)-1,N);
            end
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Setup Optimization Problem for NFXP
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

indices = repmat((1:N)',1,M)+repmat((1:M),N,1)-1;              % 175*5 matrix

% For computing the gradient of the likelihood function
PayoffDiffPrime = zeros(7+N,length(x));                        % 182*175 matrix        
PayoffDiffPrime(1,:)=0.001*(x'-repmat(x(1),1,length(x)));      % Row 1      
PayoffDiffPrime(7,:)=-1;                                       % Row 7
PayoffDiffPrime(8,:)= beta;                                    % Row 8 
PayoffDiffPrime= -beta*[zeros(7,N); eye(N)] + PayoffDiffPrime; % 182*175 matrix

CbEVPrime = zeros(length(x),7+N);                              % 175*182 matrix   
CbEVPrime(:,1)=-0.001*x;                                       % Row 1    
CbEVPrime= beta*[zeros(N,7) eye(N)] + CbEVPrime;        
CbEVPrime =  [ CbEVPrime; repmat(CbEVPrime(length(x),:),M,1)]; % 180*182 matrix
                
TransProbPrime = zeros(7+N,M);                                 % 182*5 matrix
TransProbPrime(2:6,:) = eye(M);                                % Rows 2~6      

RPrime = zeros(1,8-1+N);                                       % 1*182 vector
RPrime(:,7)=-1;                                                % Row 7
RPrime(:,8)=beta;                                              % Row 8

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Define upper and lower bounds for structural parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
NFXPlb = zeros(7,1);     % lower bound : 7*1 vector
NFXPub = zeros(7,1);     % upper bound : 7*1 vector

% thetaCost : Row 1 (Cost parameter "theta11" is non-negative
NFXPlb(1)=0;
NFXPub(1)=inf;

% thetaProbs : Rows 2~6 (Transition probability is in [0, 1])
NFXPlb(2:6)=0;
NFXPub(2:6)=1;

% RC : Row 7 (Replacement Cost is non-negative)
NFXPlb(7)=0;
NFXPub(7)=inf;

% The probability parameters in transition process must add to one.
% Linear constraint(sum(thetaProbs) = 1) Ì Aeq * theta = beq
NFXPAeq = zeros(1, length(thetatrue)); % 1*7 vector
NFXPAeq(2:6)=1;
NFXPbeq=1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% IMPLEMENTATION : NFXP/fmincon
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Initial values for structural parameters : 7*1 vector
theta0 = zeros(size(thetatrue,1),1);
theta0(1)= 1;                        % thetaCost
theta0(2:6)=1/M;                     % thetaProbs
theta0(7)= 4;                        % RC
EVold = zeros(N,1);                  % EV : 175*1 vector
BellEval = 0;                        % # of contraction mapping iterations needed to solve the Bellman equation

% Optimization options
tol_inner = 1.e-10;    % scalar
optionsNFXP = optimset('Algorithm','interior-point', 'Display', 'iter', ...
                       'GradObj','on', 'TolCon',1E-6, 'TolFun',1E-6, 'TolX',1E-15);

% Start measuring computation Time
tic

% Optimization
[thetaNFXPfmincon, fvalNFXPfmincon, flagNFXPfmincon, outputNFXPfmincon] = fmincon(@likelihoodNFXP, theta0, [], [], NFXPAeq, NFXPbeq, NFXPlb, NFXPub, [], optionsNFXP);

% Finish measuring computation time
Computation_Time = toc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ESTIMATION RESULT
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Display true and estimated parameters

disp('                                                           ')
disp('***********************************************************')
disp('                     ESTIMATION RESULT                     ')
disp('***********************************************************')
disp('                                                           ')

% Computation Time (Seconds)
disp(['COMPUTATION TIME (seconds) : ' num2str(Computation_Time)])

% Parameter Estimate
horz=['    TRUE      NFXP/fmincon'];
vert=['theta11      ';
      'theta30      ';
      'theta31      ';
      'theta32      ';
      'theta33      ';
      'theta34      ';
      'RC           '];

fprintf('\nPARAMETER ESTIMATE\n');
disp('*****************************')
disp(horz)
disp('*****************************')
disp('  ')
for ii = 1:size(thetatrue,1)
    disp(vert(ii,:))
    disp([thetatrue(ii) thetaNFXPfmincon(ii)])
end
disp('*****************************')

