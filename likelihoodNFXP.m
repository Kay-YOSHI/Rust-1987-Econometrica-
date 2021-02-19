function [f, gNFXP, h] = likelihoodNFXP(theta)

% Compute the value and the gradient of the likelihood function in NFXP

%yMEMOz
% "f" is the value of the likelihood function evaluated at the structural parameter vector "theta".  
% "gNFXP" is the gradient of the likelihood function evaluated at the structural parameter vector "theta".
%
%yCAUTIONz
% In this implementation, we do not supply second-order analytic derivatives. 
% Hence, the hessian of the likelihood function "h" is empty [].

global dt xt nT nBus N M beta
global PayoffDiffPrime TransProbPrime CbEVPrime indices;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Define parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Parameter for cost function : theta11
thetaCost = theta(1); % scalar

% Parameters for transition probability
thetaProbs = theta(2:6); % 5*1 vector
TransProb = thetaProbs;  % 5*1 vector

% Parameter for replacement cost 
RC = theta(7); % scalar

ntheta = length(theta); % scalar(=7)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Solve the Inner-loop Bellman equations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Use constration mapping iteration to solve the integrated Bellman equations
[EV, CbEV] = Bellcontract(thetaCost, TransProb, RC); % EV : 175*1 vectorCCbEV : 180*1 vector

% Let PayoffDiff[i] = -CbEV[i] - RC + CbEV[1] (= Cost[i] - beta*EV[i] - RC - Cost[1] + beta*EV[1])
% where CbEV[i] = - Cost[i] + beta*EV[i] (Expected payoff without replacement)
% = the difference in expected payoff at x[i] between engine replacement(d=1) and regular maintenance(d=0)
PayoffDiff  = -CbEV - RC + CbEV(1); % 180*1 vector              

% Let ProbRegMaint[i] = 1/(1+exp(PayoffDiff[i])); 
% = the probability of performing regular maintenance at state x[i]
ProbRegMaint = 1./(1+exp(PayoffDiff)); % 180*1 vector

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute the value and the gradient of the likelihood function 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%yMEMOz
% The likelihood function contains two elements;
%  1. Choice Probability : the likelihood that the engine is replaced given time t and state in the data
%  2. Transition Probability : the likelihood that the observed transition between t-1 and t would have occurred

% Initialize likelihood function
f = 0;

g = zeros(length(theta)+N,1);  % 182*1 vector
gPayoffPrime = g;              % 182*1 vector
gTransProbPrime = g;           % 182*1 vector

for i = 1:nBus
    
    dtM1Minus = find(dt(1:(nT-1),i)==0); % Substitute the row number with dt=0 for bus i 
    dtM1Plus  = find(dt(1:(nT-1),i)==1); % Substitute the row number with dt=1 for bus i 
    dtMinus   = find(dt((2:nT),i)==0)+1; % dtM1Minus + 1
    dtPlus    = find(dt((2:nT),i)==1)+1; % dtM1Plus + 1
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Compute the value of the likelihood function %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    f = f - ( sum( log( 1-ProbRegMaint(xt(dtPlus,i)) ) )...
          + sum( log( ProbRegMaint(xt(dtMinus,i)) ) ) ...
          +  sum( log( TransProb(xt(dtM1Plus +1,i)) ) )...
          + sum( log( TransProb(xt(dtM1Minus +1,i)-xt(dtM1Minus,i)+1) ) ) );
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Compute the gradient of the likelihood function %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % nargout : Number of output argument
    
    if nargout > 1
        
        % Derivative of choice Probability
        d1 = PayoffDiffPrime(:,xt(dtPlus,i))*ProbRegMaint(xt(dtPlus,i));           % 182*1 vector
        d2 = - PayoffDiffPrime(:,xt(dtMinus,i))*( 1-ProbRegMaint(xt(dtMinus,i)) ); % 182*1 vector
        
        % Derivative of transition Probability
        d3 = TransProbPrime(:, xt( dtM1Plus +1,i ))*(1./TransProb( xt( dtM1Plus +1,i ) ));                                  % 182*1 vector
        d4 = TransProbPrime(:, xt(dtM1Minus+1,i)-xt(dtM1Minus,i)+1)*(1./TransProb( xt(dtM1Minus +1,i)-xt(dtM1Minus,i)+1 )); % 182*1 vector
        
        gPayoffPrime = gPayoffPrime -(d1+d2);       % 182*1 vector
        gTransProbPrime = gTransProbPrime -(d3+d4); % 182*1 vector
    end
    
end 

% Continue to compute the gradient of the likelihood function  
if nargout > 1
        
    gPayoffPrimetheta = gPayoffPrime(1:ntheta);
    gPayoffPrimeEV = gPayoffPrime(ntheta+1:end);
    gTransProbPrimetheta = gTransProbPrime(1:ntheta);
    
    s1 = exp(CbEV(indices));
    s2 = exp(-RC+CbEV(1));
    s =  s1 + s2;
    logs = log(s);
    
    Rprime = zeros(1,ntheta+N);
    Rprime(ntheta)=-1;
    Rprime(ntheta+1)=beta;
            
    d1 = ((CbEVPrime.*repmat(exp(CbEV),1,ntheta + N) + exp(-RC+CbEV(1))*repmat(Rprime,N+M,1)))./(repmat(exp(CbEV)+exp(-RC+CbEV(1)),1,ntheta + N));
       
    sum1 =  reshape(sum(reshape(d1(indices',:) .* repmat(repmat(TransProb,N,1),1,ntheta + N),M,N, ntheta + N )),N,ntheta + N); 
    sum2 = logs*TransProbPrime';
    TPrime = sum1 + sum2;
    
    %EVPrime = [zeros(N,ntheta) eye(N)];
    dTdtheta = TPrime(:,1:ntheta);
    dTdEV = TPrime(:,(ntheta+1):ntheta+N);
    gNFXP = dTdtheta'*(inv(eye(N)-dTdEV))'*gPayoffPrimeEV + gPayoffPrimetheta + gTransProbPrimetheta;
end

% We do not supply second-order analytic derivatives
if nargout > 2  
   h=[];
end