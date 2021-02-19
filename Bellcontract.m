function [EV, CbEV] = Bellcontract(thetaCost, TransProb, RC)

% Solve the integrated Bellman equation using constraction mapping iteration in the NFXP algorithm. 

global x N M beta indices;
global EVold BellEval tol_inner;

% Initial value of EV = EVold
EV0 = EVold; % 175*1 vector
            
% Cost function : c(x) = 0.001*theta_1*x
% x : 175*1 vector
Cost  = 0.001 * thetaCost * x ; % 175*1 vector

ii = 0;
norm = 1;

% Contraction mapping 
while norm > tol_inner 
    
    %  Let CbEV[i] = - Cost[i] + beta*EV[i]; 
    %  = the expected payoff at x[i] if d = 0 (No replacement)
    
    CbEV = - Cost + beta*EV0;     % 175*1 vector
    CbEV(N+1:N+M)=CbEV(N);        % CbEV becomes 180*1 vector
    s1 = exp(CbEV(indices));      % 175Å~5 matrix (indices : 175Å~5 matrix)
    s2 = exp(-RC+CbEV(1));        % scalar (CbEV(1) is EV when xhat_1=0)
    s =  s1 + s2;                 % 175Å~5 matrix
    logs = log(s);                % 175Å~5 matrix
    
    % EV = É∞_{j=0}^{J} log{É∞_d'={0,1} exp[v(x', d'; É∆1, RC)+É¿EV(x', d')]} Å~ p3(x'|xhat_k,d,É∆3)
    EV = logs * TransProb;        % 175Å~1 vector
    
    BellmanViola = abs(EV - EV0); % Absolute difference b/w EV and EV0 (175Å~1 vector)
    norm = max(BellmanViola);     % Max of "BellmanViola"
    
    % Update EV
    EV0 = EV;
    ii = ii+ 1;
end

% "BellEval" is the number of contraction mapping iterations needed to solve the Bellman equation.
EVold = EV0;
BellEval = ii + BellEval;
