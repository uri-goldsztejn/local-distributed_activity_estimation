% This function decomposes the input tensor as described in:
% Goldsztejn, Uri, and Arye Nehorai. "Estimating uterine activity from electrohysterogram measurements via statistical tensor decomposition."
% Biomedical Signal Processing and Control 85 (2023): 104899.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Input parameters:
%       Required
%       - Y_input:                  Input tensor of any number of dimensions (N>2).
%
%       Optional (Name-value pairs)
%       - max_iterations:           Maximal number of iterations to run (default: 50).
%       - tolerance:                Tolerance for convergence of the model evidence (default: 1e-1).
%       - initial_sparse_variance:  Initial variance of the sparse
%               components. Lower values lead to sparser results (default: 1)
%       - rank_reduction_threshold: Percentage power threshold to reduce
%               the muli-tlinear rank of the low-rank component (X). E.g., if
%               rank_reduction_threshold = [0.01 0.1 0.1], then in each iteration,
%               the columns of the first/second/third factor matrix with less than 1%/ 10%/ 10% of the
%               mean power of the columns of that matrix will be removed
%               (default: [1e-2,1e-2,1e-2]).
%       - verbose: 0 - no progress output. 1 - print progress output in
%       command window. 2 - also display progress in plot.
%
%   Output variables:
%       - S:                Sparse tensor S.
%       - X:                Low rank tensor X.
%       - G:                Core tensor of X.
%       - U:                Low rank tensor X.
%       - tau:              Noise precision.
%       - lambdas:          Prior parameters for X.
%       - beta:             Prior parameter for the core tensor G.
%       - D:                Prior parameters for S.
%       - model_evidence:   Model evidence evolution.
%       - Y_rmse:           Root-mean-square-error of the tensor
%                   reconstruction (i.e., sqrt(mean((Y(:) - X(:) - S(:)).^2))).
%
%   Usage:
%           output = tensor_decomposition(Y, 'max_iterations', 100, 'initial_sparse_variance', 1)
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (c) 2023 Uri Goldsztejn
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



function [output] = tensor_decomposition(Y_input,varargin)


% add tensor toolbox - which can be downloaded from
% https://www.tensortoolbox.org/
addpath(genpath('tensor_toolbox'));
addpath(genpath('utils'));

% Set random seed value for reproducibility
rng('default')

%% Checking input parameters

p = inputParser;

validScalar = @(x) isnumeric(x) && isscalar(x) && (x > 0);
addParameter(p,'max_iterations',50,validScalar);
addParameter(p,'tolerance',1e-1,validScalar);
addParameter(p,'initial_sparse_variance',1,validScalar);

validInteger = @(x) isnumeric(x) && isscalar(x);
addParameter(p,'verbose',1,validInteger);
validVector = @(x) isnumeric(x) && isvector(x) && (min(x) > 0);
addParameter(p,'rank_reduction_threshold',[1e-2,1e-2,1e-2],validVector);
validTensor = @(x) isnumeric(x) || isequal(class(x),'tensor');

addRequired(p,'Y_input',validTensor);
parse(p,Y_input,varargin{:});

if  isequal(class(p.Results.Y_input),'tensor')
    Y = Y_input.data;
else
    Y = Y_input;
end

max_iterations = p.Results.max_iterations;
tolerance = p.Results.tolerance;
initial_sparse_variance = p.Results.initial_sparse_variance;
rank_reduction_threshold = p.Results.rank_reduction_threshold;
verbose = p.Results.verbose;

% Set maximal possible initial rank
Y_dimensions = size(Y);
N = ndims(Y);
R_multi = Y_dimensions;
for h = 1:length(Y_dimensions)
    R_multi(h) = min(Y_dimensions(h), prod(Y_dimensions)/Y_dimensions(h) );
    
end


options_optimization = optimoptions('fmincon','Display','none');


%% Initialize model parameters
% the nomenclature follows that of the supplementary materials in the paper.

a_delta_0 = 1e-6;
b_delta_0 = 1e-6;
a_lambda_0 = 1e-6;
b_lambda_0 = 1e-6;
a_beta_0 = 1e-6;
b_beta_0 = 1e-6;
a_tau_0 = 1e-6;
b_tau_0 = 1e-6;


deltas = (initial_sparse_variance.^(-1))*ones(Y_dimensions).*(a_delta_0/b_delta_0);
S = (deltas.^(-0.5)).*randn(Y_dimensions);

lambdas = cell(N,1);
for i = 1:N
    lambdas{i} = a_lambda_0/b_lambda_0.*ones(R_multi(i),1);
end


U = cell(N,1);
sigma_U = cell(N,1);
for n = 1:N
    U{n} = randn(Y_dimensions(n),R_multi(n));
    sigma_U{n} = diag(lambdas{n}.^(-1/2));
end

beta = a_beta_0/b_beta_0;

tmp_kron = kron(sparse(diag(lambdas{2}.^(-1))),sparse(diag(lambdas{1}.^-(0.5))));
for n = 3:N
    tmp_kron = kron(sparse(diag(lambdas{n}.^(-1))),tmp_kron);
end
sigma_G = (beta.^(-1)*tmp_kron);


G_vec = full(diag(sigma_G).*randn(size(sigma_G,1),1));
tau =  a_tau_0/b_tau_0;
for n=1:N
    relevant_idx{n} = repmat(true,1,Y_dimensions(n));
end


if verbose > 0
    disp("Iteration || Model evidence || Multilinear rank || RMSE")
end

if verbose > 1
    figure(1)
end
%% update model parameters

for iteration = 1:max_iterations
    
    
    % Update factor matrices U
    
    G_tensor = reshape(G_vec,R_multi);
        
    for n = 1:N
        
        G_n{n} = tenmat(G_tensor,n);
        k_neq_n = [1:(n-1),(n+1):N];
        U_k_neq_n={};
        U_k_neq_n_2={};
        for m = 1:length(k_neq_n)   
            U_k_neq_n{m} = U{k_neq_n(m)};%'
            U_k_neq_n_2{m} = U{k_neq_n(m)}'*U{k_neq_n(m)}+Y_dimensions(k_neq_n(m))*sigma_U{k_neq_n(m)};
        end
 
        d = length(U_k_neq_n);
        options=struct('transpose',ones(1,d),'ind',fliplr(1:length(k_neq_n)));
        options_U2 = struct('ind',fliplr(1:length(k_neq_n)));      
        GUUG = G_n{n}.data*ckronx(U_k_neq_n_2, G_n{n}.data',options_U2 );
        sigma_U{n} = inv((diag(lambdas{n})+ tau*GUUG));
        Y_minus_S = (tenmat(Y,n)'-tenmat(S,n)');
        mu_T = sigma_U{n}*tau*G_n{n}*ckronx(U_k_neq_n,Y_minus_S.data,options);
        U{n} = mu_T.data';
    end
    
    
    % update core tensor G
    
    for m = 1:N
        U_trans{m} = U{m}';
    end
    V ={};
    D ={};
    for m = 1:N
        
        U_T_T = U{m}'*U{m}+Y_dimensions(m)*sigma_U{m};
        Lambda_n_05{m} = sparse(diag(lambdas{m}.^(-0.5)));
        A = Lambda_n_05{m}*U_T_T*Lambda_n_05{m};
        [V_temp,D_temp] = eig(A);
        V{m} = real(V_temp); % A is symmetric so V and D are real
        D{m} = real(D_temp);
    end
    
    tmp_D = sparse(D{1});
    for m = 2:N
        tmp_D = kron(sparse(D{m}),tmp_D);
    end
    
    a = tau*tmp_D;
    inv_diag = diag((diag(beta*speye(size(tmp_D,1))) + diag(tau*tmp_D)).^(-1));
    
    lambda_v = {};
    for m = 1:N
        lambda_v{m} = Lambda_n_05{m}*V{m};
    end
    
    options_part1=struct('ind',N:-1:1);
        
    d = length(lambda_v);
    options_part2=struct('transpose',ones(1,d),'ind',N:-1:1);  
    
    G_vec_ending = diag(inv_diag).*ckronx(lambda_v,ckronx(U_trans,Y(:) - S(:),options_part1) ,options_part2);
    G_vec = ckronx(lambda_v,tau*G_vec_ending,options_part1);
    
    a_lambda_rn = cell(N,1);
    b_lambda_rn = cell(N,1);
    
    % update lambdas
    for n = 1:N
        
        
        a_lambda_rn{n} = Y_dimensions(n)/2 + a_lambda_0 + 0.5*prod(R_multi)/R_multi(n);
        k_neq_n = [1:(n-1),(n+1):N];
        d =length(k_neq_n);
        options=struct('ind',fliplr(1:length(k_neq_n)));      
        
        tmp_kron_lambda = sparse(diag(lambdas{k_neq_n(1)}));
        for m = 2:length(k_neq_n)
            tmp_kron_lambda = kron(sparse(diag(lambdas{k_neq_n(m)})),tmp_kron_lambda);
        end
          
        G_tensor = reshape(G_vec,R_multi);
        
        for rn = 1:R_multi(n)
            sz = size(G_tensor);
            inds = repmat({':'},1,ndims(G_tensor));
            inds{n} = rn;
            
            G_sliced = G_tensor(inds{:});
            b_lambda_rn{n}(rn) = b_lambda_0 + 0.5*(Y_dimensions(n)*sigma_U{n}(rn,rn) +...
                U{n}(:,rn)'* U{n}(:,rn)) + 0.5*beta*(G_sliced(:).^2)'*diag(tmp_kron_lambda); 
                   
        end
              
        lambdas{n} = a_lambda_rn{n}./b_lambda_rn{n}; 
         
    end
    
    
    
    
    fun = @(x)objective_lambda_0(x,a_lambda_rn,b_lambda_rn,R_multi);
    x0 = [a_lambda_0, b_lambda_0];
    lb = [1e-9 1e-9];
    ub = [1e-4 1e-4];
    x = fmincon(fun,x0,[],[],[],[],lb,ub,[],options_optimization);   
    a_lambda_0 = x(1);
    b_lambda_0 = x(2);
    
    % update beta
    a_beta = a_beta_0 + 0.5*prod(R_multi);
    
    
    lambda_kron_list={};
    
    for n = 1:N
        lambda_kron_list{n} = sparse(diag(lambdas{n}));
    end
    options = struct('ind',N:-1:1);
    
    b_beta = b_beta_0 + 0.5*G_vec'*ckronx(lambda_kron_list,G_vec,options);
      
    beta = a_beta/b_beta;

    f = @(x,a_beta,b_beta)(x(1)-1)*(psi(a_beta) - log(b_beta)) - x(2)*a_beta/b_beta + x(1)*log(x(2)) - log(gamma(x(1)));
    fun = @(x)-f(x,a_beta,b_beta);
    x0 = [a_beta_0, b_beta_0];
    
    
    x = fmincon(fun,x0,[],[],[],[],lb,ub,[],options_optimization);
    a_beta_0 = x(1);
    b_beta_0 = x(2);

    % update sparse tensor S
    
    sigma_2_delta_i = 1./(tau+deltas);
    
    nv   = ndims(S);
    v    = ones(1, nv);
    vLim = size(S);
    ready = false;
    while ~ready
        
        
        tmp_u_in_kron={};
        for n = 1:N
            tmp_u_in_kron{n} = U{n}(v(n),:);%'
        end
        d = length(tmp_u_in_kron);
        options = struct('ind',N:-1:1);
        C = {};
        for t = 1:length(v)
            C{t} = v(t);
        end
        
        
        S(C{:}) = sigma_2_delta_i(C{:})*tau*(Y(C{:}) - ckronx(tmp_u_in_kron,G_vec,options));
           
        
        % Update the index vector:
        ready = true;       
        for k = 1:nv
            v(k) = v(k) + 1;
            if v(k) <= vLim(k)
                ready = false;  
                break;          % v(k) increased successfully, leave "for k" loop
            end
            v(k) = 1;         % Reset v(k), proceed to next k
        end
    end
    
    
    % update deltas
    a_delta = a_delta_0 + 0.5;
    b_delta = b_delta_0 + 0.5*(S.^2 + sigma_2_delta_i);
    
    deltas = a_delta./b_delta; 
    
    f = @(x,a_delta,b_delta) -sum(x(1)*log(x(2))-log(gamma(x(1))) + (x(1)-1)*(psi(a_delta)- log(b_delta)) -  x(2)*a_delta./b_delta ,'all');
    fun = @(x)f(x,a_delta,b_delta);
    
    x0 = [a_delta_0, b_delta_0];
    
    ub= [1e-4 1e-4];
    
    x = fmincon(fun,x0,[],[],[],[],lb,ub,[],options_optimization);
    a_delta_0 = x(1);
    b_delta_0 = x(2);
    
    % update tau
        
    for m = 1:N
        U_trans{m} = U{m};
        U_2{m} = U{m}'*U{m}+Y_dimensions(m)*sigma_U{m};
    end
    
    d=N;
    options_U = struct('ind',N:-1:1);
    options_U_2 =  struct('ind',N:-1:1);
    
    
    for m = 1:N
        U_trans{m} = U{m};
        U_2{m} = U{m}'*U{m}+Y_dimensions(m)*sigma_U{m};
    end
    
    for n=1:N
        trace_U2_n(n) = trace(U{n}'*U{n}) + trace(Y_dimensions(n)*sigma_U{n});
    end
    trace_A = prod(trace_U2_n);
    
    X_tmp = double(ttensor(tensor(G_tensor),U));
    
    
    E_y_x_s_2 = sum(Y.^2,'all') - 2*Y(:)'*ckronx(U_trans,G_vec,options_U) + sum(ckronx(U_2,(G_vec),options_U_2).*G_vec) +   ... %trace_A+
        sum(sigma_2_delta_i + S.^2,'all') - 2*Y(:)'*S(:) + 2*S(:)'*ckronx(U_trans,G_vec,options_U);
    
    
    
    a_tau = a_tau_0 + 0.5*prod(Y_dimensions);
    b_tau = b_tau_0 + 0.5*E_y_x_s_2;
    
    tau = a_tau./b_tau;
    
    
    
    f= @(x,a_tau,b_tau) (a_tau_0-1)*(psi(a_tau) - log(b_tau)) - x(2)*a_tau/b_tau   + x(1)*log(x(2)) - log(gamma(x(1))) ;
    fun = @(x)-f(x,a_tau,b_tau);
    x0 = [1e-6, 1e-6];
    
    x = fmincon(fun,x0,[],[],[],[],lb,ub,[],options_optimization);
    
    a_tau_0 = x(1);
    b_tau_0 = x(2);
    
    %% compute the model evidence
    
    term1(iteration) = 0.5*prod(Y_dimensions)*(psi(a_tau)-safelog(b_tau)) - 0.5*a_tau/b_tau*E_y_x_s_2 -0.5*prod(Y_dimensions)*log(2*pi);
    
    
    aux_1_term2 = 0;
    for n = 1:N
        aux_1_term2 = aux_1_term2 + Y_dimensions(n)/2*sum((psi(a_lambda_rn{n}) - safelog(b_lambda_rn{n})));
    end
    
    aux_2_term2 = 0;
    
    for n = 1:N
        aux_2_term2 = aux_2_term2 + lambdas{n}*diag(sigma_U{n});
        for in = 1:Y_dimensions(n)
            aux_2_term2 = aux_2_term2 + (U{n}(in,:))*sparse(diag(lambdas{n}))*(U{n}(in,:))';
        end
    end
    
    term2(iteration) = -0.5*safelog(2*pi)*R_multi*Y_dimensions' + aux_1_term2 - 0.5*aux_2_term2;
    
    term3(iteration) = 0;
    
    aux_1_term3 = 0;
    aux_2_term3 = 1;
    G_tensor_2 = reshape(G_vec.^2,R_multi);
    nv   = ndims(G_tensor_2);
    v    = ones(1, nv);
    vLim = size(G_tensor_2);
    ready = false;
    while ~ready
        
        aux_2_term3 = 1;
        
        C = {};
        for t = 1:length(v)
            C{t} = v(t);
            %             aux_2_term3 = aux_2_term3*(psi(a_lambda_rn{t}) - safelog(b_lambda_rn{t}(v(t))));
            aux_2_term3 = aux_2_term3*((a_lambda_rn{t})/(b_lambda_rn{t}(v(t))));
        end
        
        aux_1_term3 = aux_1_term3 - 0.5*a_beta/b_beta*G_tensor_2(C{:})*aux_2_term3;
        
        
        
        % Update the index vector:
        ready = true;      
        for k = 1:nv
            v(k) = v(k) + 1;
            if v(k) <= vLim(k)
                ready = false;  
                break;          % v(k) increased successfully, leave "for k" loop
            end
            v(k) = 1;         % Reset v(k), proceed to next k
        end
    end
    
    aux_3_term3 = 0;
    for n = 1:N
        aux_3_term3 = aux_3_term3 + 0.5*prod(R_multi)/R_multi(n)*sum((psi(a_lambda_rn{n}) - safelog(b_lambda_rn{n})));%*sum(safelog(lambdas{n}));
    end
    
    term3(iteration) = aux_3_term3 + aux_1_term3 + 0.5*prod(R_multi)*log(beta/(2*pi)) ;
    
    term4(iteration) = 0;
    nv   = ndims(S);
    v    = ones(1, nv);
    vLim = size(S);
    ready = false;
    while ~ready
        
        C = {};
        for t = 1:length(v)
            C{t} = v(t);
        end
        
        term4(iteration) = term4(iteration) +0.5*( psi(a_delta)-safelog(b_delta(C{:})) - a_delta/b_delta(C{:})*( sigma_2_delta_i(C{:}) + S(C{:}).^2 ) ) - log(2*pi);
        
        % Update the index vector:
        ready = true;       
        for k = 1:nv
            v(k) = v(k) + 1;
            if v(k) <= vLim(k)
                ready = false;  
                break;          % v(k) increased successfully, leave "for k" loop
            end
            v(k) = 1;         % Reset v(k), proceed to next k
        end
    end
    
    term5(iteration) = 0;
    
    for n = 1:N
        term5(iteration) = term5(iteration) + prod(R_multi)*(a_lambda_0*safelog(b_lambda_0) - safelog(gamma(a_lambda_0))) + sum((a_lambda_0-1)*(psi(a_lambda_rn{n}) - safelog(b_lambda_rn{n})) - b_lambda_0*a_lambda_rn{n}./b_lambda_rn{n});
    end
    
    term6(iteration) = (a_beta_0-1)*(psi(a_beta) - log(b_beta)) - b_beta_0*a_beta/b_beta + a_beta_0*log(b_beta_0) - log(gamma(a_beta_0));
    
    term7(iteration) =  sum((a_delta_0-1)*(psi(a_delta) - log(b_delta)) - b_delta_0*a_delta./b_delta  + a_delta_0*log(b_delta_0) - log(gamma(a_delta_0)) ,'all');
    
    term8(iteration) = (a_tau_0-1)*(psi(a_tau) - log(b_tau)) - b_tau_0*a_tau/b_tau   + a_tau_0*log(b_tau_0) - log(gamma(a_tau_0)) ;
    
    % entropy terms
    term9(iteration) = 0;
    for n = 1:N
        term9(iteration) = term9(iteration) + 0.5*Y_dimensions(n)*safelog(det(sigma_U{n})) +  0.5*R_multi*Y_dimensions'*(1+safelog(2*pi)) ;
    end
    
    
    aux_1_term10=0;
    
    for n = 1:N
        aux_1_term10 = aux_1_term10 + prod(R_multi)/R_multi(n)*log((det(lambda_v{n}))^2);
        
    end
    
    term10(iteration) = 0.5* prod(R_multi)*log(2*pi*exp(1)) + aux_1_term10 + safelog(det(inv_diag));
    
    term11(iteration) = 0.5*sum(safelog(sigma_2_delta_i*2*pi*exp(1)),'all');
    
    
    term12(iteration) = 0;
    for n = 1:N
        term12(iteration) = term12(iteration) + sum(safelog(gamma(a_lambda_rn{n})) - (a_lambda_rn{n}-1).*psi(a_lambda_rn{n}) -safelog(b_lambda_rn{n}) + a_lambda_rn{n});
    end
    
    term13(iteration) = safelog(gamma(a_beta)) - (a_beta-1).*psi(a_beta) -safelog(b_beta) + a_beta;
    
    
    term14(iteration) = sum(safelog(gamma(a_delta)) - (a_delta-1).*psi(a_delta) -safelog(b_delta) + a_delta,'all');
    
    term15(iteration) = safelog(gamma(a_tau)) - (a_tau-1).*psi(a_tau) -safelog(b_tau) + a_tau;
    
    model_evidence(iteration) = term1(iteration) + term2(iteration) + term3(iteration) +  term4(iteration) + term5(iteration) + term6(iteration)...
        + term7(iteration) + term8(iteration) + term9(iteration)+ term10(iteration)  + term11(iteration) + term12(iteration) + term13(iteration) + term14(iteration) + term15(iteration);
    
    
    
    % reduce rank
    for n = 1:N
        power = zeros(1,R_multi(n));
        for r = 1:R_multi(n)
            power(r) = sum(U{n}(:,r).^2 );
            
        end
        relevant_idx{n} = power > (mean(power))*rank_reduction_threshold(n);
        
        if sum(relevant_idx{n})<2
            relevant_idx{n} = false(1,R_multi(n));
            [~,idx] = sort(power,'descend');
            relevant_idx{n}(idx(1)) = true;
            relevant_idx{n}(idx(2)) = true;
            
            
        end
        
        if sum(relevant_idx{n}) == 0
            error('R=0')
        end
        
        
        
        U{n} = U{n}(:,relevant_idx{n});
        sigma_U{n} = sigma_U{n}(relevant_idx{n},relevant_idx{n});
        
        lambdas{n} = lambdas{n}(relevant_idx{n});       
        G_tensor = reshape(G_vec,R_multi);    
        idx_G = repmat({':'},1,ndims(G_tensor));
        idx_G{n} = relevant_idx{n};
        G_tensor = G_tensor(idx_G{:});
        G_vec = G_tensor(:);   
        R_multi(n) = sum(relevant_idx{n});      
        
    end
    
    if isequal(R_multi, 2*ones(size(R_multi))) && (iteration>1)
        disp('Rank reduction limit attained')
        
        break
        
        
    end
    
    X_tmp = double(ttensor(tensor(G_tensor),U));
    
    Y_rmse(iteration) = sqrt(mean((Y(:) - X_tmp(:) - S(:)).^2));
    previous_samples_num = 4;
    
    if iteration>previous_samples_num
        
        cum_me_change = 0;
        for h = 0:previous_samples_num-1
            cum_me_change = cum_me_change + abs((model_evidence(iteration-h) - model_evidence(iteration-h-1))/model_evidence(iteration-h-1))*100;
        end
        
        if cum_me_change < previous_samples_num*tolerance
            if verbose>0
                disp('Model evidence converged')
            end
            break
        end
        
        
    end
    
    % Print progress
    if verbose > 0
        
        space_iteration =     strlength("Iteration") - strlength(string(iteration));
        space_model_evidence=     strlength("Model evidence") - strlength(string(int32(model_evidence(iteration))));
        space_R_multi=     strlength("Multilinear rank") - strlength( mat2str(R_multi))+1;       
        
        txt = sprintf([ blanks(space_iteration), '%i ||', blanks(space_model_evidence),' %.0f ||' , blanks(space_R_multi),'%s ||', ' %.2f']...
            ,iteration, model_evidence(iteration),mat2str(R_multi),Y_rmse(iteration) );
        disp(txt)
        
        
    end
    
    if verbose > 1
        axis_font_size = 15;
        figure(1)
        subplot(1,4,1)
        plot(model_evidence(1:end),'*')
        xlabel('Iteration #','FontSize',axis_font_size); ylabel('Model evidence','FontSize',axis_font_size)
        subplot(1,4,2)
        plot(squeeze(X_tmp(4,2,:)));title('Distributed act','FontSize',axis_font_size)
        xlabel({'Time' 'samples'},'FontSize',axis_font_size);
        subplot(1,4,3)
        plot(squeeze(S(4,2,:)));title('Localized act','FontSize',axis_font_size)
        xlabel({'Time' 'samples'},'FontSize',axis_font_size);
        subplot(1,4,4)
        plot(squeeze(Y(4,2,:)) -squeeze(X_tmp(4,2,:))-squeeze(S(4,2,:)));title('Estimated noise','FontSize',axis_font_size)
        xlabel({'Time' 'samples'},'FontSize',axis_font_size);
    end
    
    %% Output parameters
    
    output.X = double(ttensor(tensor(G_tensor),U));
    output.S = S;
    output.G = G_tensor;
    output.tau = tau;
    output.U = U;
    output.lambdas = lambdas;
    output.beta = beta;
    output.D = deltas;
    output.model_evidence = model_evidence;
    output.Y_rmse = Y_rmse;
    
  
end
