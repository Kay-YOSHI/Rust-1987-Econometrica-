# Rust-1987-Econometrica-

This repository contains MATLAB (ver. R2018b) files that replicate the estimation results in the paper 
"*Optimal Replacement of GMC Bus Engines: An Empirical Model of Harold Zurcher*" written by Rust, J. (1987). 

The code is based on the one provided by Su, C. and Judd, K. L. in the paper 
"*Constrained Optimization Approaches to Estimation of Structural Models*" (2012). 
Although the authors used *KNITRO*, fee-charging constrained optimization solver, to solve a single-agent dynamic discrete choice model, 
I used the function *fmincon*, one of the MATLAB built-in functions and verified that the estimated parameters are very close to the true parameters.

In the code, 
a data set is generated given the parameter estimates in the third column of Table X in Rust(1987) as true parameters. 
After that, the model is estimated by the nested fixed-point algorithm. 

## Reference

Rust, J.: "Optimal Replacement of GMC Bus Engines: An Empirical Model of Harold Zurcher," *Econometrica*, Vol. 55, No. 5, pp. 999-1033 (1987)

Su, C. and Judd, K. L.: "Constrained Optimization Approaches to Estimation of Structural Models," *Econometrica*, Vol. 80, No. 5, pp. 2213-2230 (2012)
