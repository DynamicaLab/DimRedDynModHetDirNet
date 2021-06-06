
#define _USE_MATH_DEFINES
#include "random.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <sys/stat.h>
#include <sys/types.h>

#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_odeiv2.h>

#include <gsl/gsl_math.h>

#include <gsl/gsl_multiroots.h>
#include <gsl/gsl_roots.h>

#include <gsl/gsl_statistics.h>
#include <gsl/gsl_sort_double.h>

#include <gsl/gsl_blas.h>   // matrix operations
#include <gsl/gsl_linalg.h> // linear algebra


// * * * * * * * * * * * * * * *
//         STRUCTURES
// * * * * * * * * * * * * * * *

struct params_all {
    
    int N;
    int n;
    double tau;
    double d0;
    int reduced_system;
    int correction_red_syst;
    int reduction_method;
    int reduction_correction;
    
    gsl_matrix *dMat_or;
    gsl_matrix *muMat;
    gsl_matrix *lambdaMat;
    gsl_matrix *dMat;
    gsl_matrix *deltaMat;
    gsl_matrix *mMat;
    
    gsl_vector_int *group_indices;
    int *ns;
    
    double t_sim;
    int steps;
    char *directory;
    char *subdirectory;
};

struct params_cond_ini {
    
    double x_ini0;
    double x_ini1;
};

struct params_interaction_matrix {
 
    double h;
    double rho;
    int real_network;
    int n_blocks;
    int weighted;
};

struct params_bif_diag {
    
    int nd;
    double *d_v;
    double *alpha_v;
};


// * * * * * * * * * * * * * * *
//      GLOBAL PARAMETERS
// * * * * * * * * * * * * * * *

int type_fFG;
double tau_fG;
double mu_fG;
double gamma_fG;
double B_fG;
double C_fG;
double K_fG;
double D_fG;
double E_fG;
double H_fG;


// * * * * * * * * * * * * * * *
//         FUNCTIONS
// * * * * * * * * * * * * * * *

void copy_file ( char *, char * );
void convert_time( double, int * );
int * allocate_vector_int ( int, char * );
int ** allocate_matrix_int ( int, int, char * );
double * allocate_vector_double ( int, char * );
double ** allocate_matrix_double ( int, int, char * );
void copy_vector_double( int, double *, double * );

FILE * open_file( char *, char *, char * );
void print_vector( FILE *, double *, int );
void print_matrix_gsl( FILE *, gsl_matrix * );
void print_matrix_gsl_mathematica( FILE *, gsl_matrix * );
void print_vector_gsl_terminal( gsl_vector * );
void print_matrix_gsl_terminal( gsl_matrix * );
void print_matrix_gsl_int_terminal( gsl_matrix_int * );
void print_matrix_double( FILE *, double **, int, int );

void perturb_node_partition ( int, int *, gsl_vector_int *, double );
void reorder_dMat ( gsl_matrix *, int *, int * );

void print_vectors_observables( struct params_all *, char *, char * );
void read_and_create_dMat_or ( int *, gsl_matrix **, char *, char * );
void read_communities ( int *, int **, gsl_vector_int **, char *, char * );

void compute_parameters_homogeneous ( struct params_all * );
void compute_parameters_degree ( struct params_all * );

double vector_norm_1 ( gsl_vector * );
double euc_dist_vectors( int, double *, double * );
double norm_2_matrix ( gsl_matrix * );
double norm_1_matrix ( gsl_matrix *A );

int dominant_eigenv( gsl_matrix *, gsl_vector *, double * );
void compute_cMat ( gsl_matrix *, int, int *, gsl_matrix **, gsl_vector **, double *, int n );
void elimination_Gauss ( gsl_matrix * );
int maximal_li_subset ( gsl_matrix *, int * );
int pseudoinverse( gsl_matrix *, gsl_matrix * );
int solve_linear_system ( gsl_matrix *, gsl_vector *, gsl_vector * );

int generate_cMat ( gsl_matrix **, int *, gsl_matrix **, gsl_vector **, double *, int );
double average_or_maximal_norm( int, gsl_matrix **, double * );
void random_vector_sum1 ( gsl_vector *, double );
double compute_error ( int, gsl_matrix **, double *, gsl_vector * );
void analyze_error( int, gsl_matrix **, double *, int, double *, double, double );
void solve_compatibility_eqs ( struct params_all *, int );

double fF ( double );
double fF1 ( double );
double fG ( double, double );
double fG1 ( double, double );
double fG2 ( double, double );
double fG11 ( double, double );
double fG12 ( double, double );
double fG22 ( double, double );

double derivative_xi_resp_xk ( struct params_all *, int, int, const double [] );
double derivative_xi_resp_xk_corr1 ( struct params_all *, int, int, const double [] );
double derivative_xi_resp_xk_corr2 ( struct params_all *, int, int, const double [] );

int func ( double, const double [], double [], void * );
int jac ( double, const double [], double *, double [], void * );

int print_state (size_t, gsl_multiroot_fsolver * );
void initial_condition( int, double, double, double * );
void perturb_initial_condition( int, double, double * );

void print_step( FILE *, int, double, const double [] );
void print_equilibrium( FILE *, double, int, int, const double [], int *, int );

double observable_average ( int, const double [], int * );
void compute_observables ( struct params_all *, double *, double * );
int integrate_system ( struct params_all *, double *, double * );

void print_params ( struct params_all );

void rescale_dMat ( double, double, gsl_matrix *, gsl_matrix * );

void bifurcation_diagram ( struct params_all *, struct params_cond_ini *, struct params_bif_diag, char *, char *, int, int, int );
void read_data_bif_diagram ( gsl_vector *, gsl_vector *, char *, char *, char * );
void read_complete_data_bif_diag ( gsl_matrix *, char *, char * );
void compute_data_bif_diag ( gsl_matrix *, gsl_matrix *, int *, struct params_all * );

double RMSE ( gsl_vector *, gsl_vector * );
void define_bif_diag_parameters ( struct params_interaction_matrix *, struct params_bif_diag * );
void define_directories ( char *, char *, char *, struct params_interaction_matrix );
void define_network_name ( char *, struct params_interaction_matrix );
void define_method_name ( char *, char *, struct params_all );
    
