// GENERADOR DE NUMEROS ALEATORIOS

// Al compilar, linkar correctamente con -lgsl -lgslcblas.
// Ejemplo: g++  program.cpp -lgsl -lgslcblas	(c++)
//			gcc  program.c -lgsl -lgslcblas		(c)

#include <time.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#define RANDOM gsl_rng_uniform(gsl_rng_r)
#define RANDOM_INT(A) gsl_rng_uniform_int(gsl_rng_r, A)
#define RANDOM_GAUSS(M,S) ( M + gsl_ran_gaussian(gsl_rng_r, S) )
#define RANDOM_GAUSS_BIVARIATE(sigma1, sigma2 , rho , x1 , x2) gsl_ran_bivariate_gaussian(gsl_rng_r, sigma1, sigma2, rho, x1, x2)
#define LOGISTIC(s)  gsl_ran_logistic(gsl_rng_r,s)
#define RANDOM_GAMMA(A,B) gsl_ran_gamma(gsl_rng_r, A, B)
#define RANDOM_BINOMIAL(p,n) gsl_ran_binomial(gsl_rng_r, p, n)

// Probability density function
#define RANDOM_GAUSS_PDF(x,M,S) gsl_ran_gaussian_pdf(x-M, S)

// semilla del reloj
#define INITIALIZE_RANDOM {gsl_rng_env_setup(); if(!getenv("GSL_RNG_SEED")) gsl_rng_default_seed = time(0); gsl_rng_T=gsl_rng_default;  gsl_rng_r=gsl_rng_alloc(gsl_rng_T);}
// semilla fija
//#define INITIALIZE_RANDOM {gsl_rng_env_setup(); gsl_rng_T=gsl_rng_default;  gsl_rng_r=gsl_rng_alloc(gsl_rng_T);}

#define FREE_RANDOM gsl_rng_free(gsl_rng_r);

const gsl_rng_type * gsl_rng_T;
gsl_rng * gsl_rng_r;

