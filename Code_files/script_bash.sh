#!/bin/bash

# DO NOT PUT SPACES IN THE EQUALITIES THAT DEFINE VARIABLES !!!

pdirectory="/Users/marina/Desktop/Prova"
real_network=0
n_blocks=2
h=1
rho=0.8

# --------------------------------------------------------------------------
# Create or read a network and generate node partitions
#
# The arguments to the python script are:
#   pdirectory | n_blocks | h | rho | real_network
#
# The network can either be read or created according to the parameter "create_graph"
#
# From a given initial partition, successive refinements are created.
# The initial partition can be either
#    * read from file
#    * computed using an algorithm
#    * defined using a predefined set of groups
# --------------------------------------------------------------------------

ipython network_clustering.py $pdirectory $n_blocks $h -- $rho $real_network

# ----------------
# Compile .c files
# ----------------

gcc main.c functions.c -lm -lgsl -lgslcblas -o executable
gcc main_partition_perturb.c functions.c -lm -lgsl -lgslcblas -o executable_perturb

# -------------------------------------------------------------------------------------
# Bifurcation diagram for different partition refinements
#
# Each time that the executable file is executed, a bif. diag. is computed
# for all methods (spectral, homogeneous and degree if n=1)
# and for all the corrections (0, 1 and 2)
#
# The arguments to the executable file are:
#   pdirectory | n_blocks | h | rho | real_network | read_coms | ref_index
#
# The arguments to the python script for plotting are:
#   pdirectory | n_blocks | h | rho | real_network | read_coms | ref_index |
#   correction | what_to_plot
# -------------------------------------------------------------------------------------

# 1-dim reduction
ref_index=0
read_coms=0
correction=2 # correction that will be plotted
./executable $pdirectory $n_blocks $h $rho $real_network $read_coms $ref_index
ipython make_plots.py $pdirectory $n_blocks $h -- $rho $real_network $read_coms $ref_index $correction 0

# n-dim reduction (n>1):
#   original partition (ref_index = 0)
#   refined partitions (ref_index > 0)
read_coms=1
min_index=0      # min refinement index
max_index=1      # max refinement index

for (( ref_index=min_index; ref_index<=$max_index; ref_index++ ))
do
    ./executable $pdirectory $n_blocks $h $rho $real_network $read_coms $ref_index
    ipython make_plots.py $pdirectory $n_blocks $h -- $rho $real_network $read_coms $ref_index $correction 0
done

# Plot the dynamic error as a function of n
ipython make_plots.py $pdirectory $n_blocks $h -- $rho $real_network $read_coms $ref_index $correction 2

# Plot the error in the compatibility equations
read_coms=1
ref_index=4
ipython make_plots.py $pdirectory $n_blocks $h -- $rho $real_network $read_coms $ref_index $correction 3


# -----------------------------------------------------------------------------------------
# Dynamic error for different perturbations of a given partition (n>1)
#
# The arguments to the executable file are:
#   pdirectory | n_blocks | h | rho | real_network | read_coms | ref_index |
#   reduction_meth | correction
#
# The arguments to the python script for plotting are:
#   pdirectory | n_blocks | h | rho | real_network | read_coms | ref_index |
#   correction | what_to_plot
# -----------------------------------------------------------------------------------------

correction=2    # correction used to define the reduced system
read_coms=1
min_index=0     # min refinement index
max_index=-1     # max refinement index

for (( ref_index=min_index; ref_index<=$max_index; ref_index++ ))
do
    ./executable_perturb $pdirectory $n_blocks $h $rho $real_network $read_coms $ref_index 0 $correction # spectral
    ./executable_perturb $pdirectory $n_blocks $h $rho $real_network $read_coms $ref_index 1 $correction # homogeneous
    ipython make_plots.py $pdirectory $n_blocks $h -- $rho $real_network $read_coms $ref_index $correction 1
done
