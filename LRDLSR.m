function [Q, T, M, value] = LRDLSR(X, c, H, train_num, alpha, beta, gamma, lambda) 

rho = 1.1;
tol = 1e-6
max_mu = 1e8;
mu = 1e-5;
maxIter = 1e3;
[d, n] = size(X);

%% initialize
Q = zeros(c, d);
T = H;
P = H;
M = ones(c, n);
B = 2 * H - ones(c, n);
Y1 = zeros(c, n);

%% starting iterations
iter = 0;
while iter < maxIter
    iter = iter + 1; 
    
    Tk = T;
    Pk = P;
    Qk = Q;
    Mk = M;


   %update T
   R = H + B .* Mk; 
   T = (1 + alpha + gamma + mu) \ (Qk * X + alpha * R + mu * P - Y1);
    
   %update P
   for i = 1 : c
       Ti = T(:, (i - 1) * train_num + 1 : i * train_num);
       Y1i = Y1(:, (i - 1) * train_num + 1 : i * train_num);
       Pitemp = Ti + mu \ Y1i;
       [U,sigma,V] = svd(Pitemp, 'econ');
       sigma = diag(sigma);
       svp = length(find(sigma > beta / mu));
       if svp>=1
         sigma = sigma(1:svp) - beta / mu;
       else
         svp = 1;
         sigma = 0;
       end
       Pi = U(:, 1 : svp) * diag(sigma) * V(:, 1 : svp)';
       P(:, (i - 1) * train_num + 1 : i * train_num) = Pi;
       clear Pi Ti Y1i Pitemp svp sigma
   end
    
    
   %update Q  
   Q = T * X' * inv(X * X' + lambda * eye(d));
   
  
   %update M
   S = T - H;
   Mtemp = B .* S;
   M = max(Mtemp, 0);
   
   value1 = norm(Q * X - T, 'fro') ^ 2;
   value2 = norm(T - B .* M - H, 'fro') ^ 2;
   value3 = 0;
   for i = 1 : c
       Ti = T(:, (i - 1) * train_num + 1 : i * train_num);
       value3 = value3 + sum(svd(Ti));
   end
   value4 = norm(T, 'fro') ^ 2;
   value5 = norm(Q, 'fro') ^ 2;


   value(iter) = 2 \ value1 + 2 \ (alpha * value2) + beta * value3 + 2 \ (gamma * value4) + 2 \ (lambda * value5);
  %% convergence check   
   leq1 = T - P;

   stopC = max(max(abs(leq1)));
   if stopC < tol || iter >= maxIter    
       break;
   else
       Y1 = Y1 + mu * leq1;
       mu = min(max_mu, mu * rho);
   end
   if (iter==1 || mod(iter, 5 )==0 || stopC<tol)
           disp(['iter ' num2str(iter) ',mu=' num2str(mu,'%2.1e') ...
           ',stopALM=' num2str(stopC,'%2.3e') ]);
   end

end

end
