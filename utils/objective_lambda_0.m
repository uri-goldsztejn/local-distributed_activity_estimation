function f = objective_lambda_0(x,a_lambda_rn,b_lambda_rn,R_multi)
    f = 0;
    for n = 1:length(a_lambda_rn)
       f = f + prod(R_multi)*(x(1)*safelog(x(2)) - safelog(gamma(x(1)))) + sum((x(1)-1)*(psi(a_lambda_rn{n}) - safelog(b_lambda_rn{n})) - x(2)*a_lambda_rn{n}./b_lambda_rn{n});
    end
    f=-f;
end