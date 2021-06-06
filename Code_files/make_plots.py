# -*- encoding: utf-8 -*-
# Ara podem posar accents als comentaris!

import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import pylab
from matplotlib.ticker import NullFormatter
import sys
from pylab import *
import matplotlib.patches as pts
from scipy.stats import poisson
from scipy.stats import norm
from scipy.stats import gamma

# Format for figures
ftypes = ["jpg", "pdf"]

# Figure dimensions
wfig0 = 30/2.54   # conversion to inches
hfig0 = 20/2.54

# Ad-hoc colors
verd = (0.5,0.7,0)
taronja = (0.9,0.6,0)
taronja_vermellos = (231/256., 127/256., 35/256.)
vermell = (0.8,0.3,0)
vermell_lilos = (0.65,0.25,0.35)
lila = ( 0.5, 0.2, 0.7)
blau_fosc = ( 0.3, 0.35, 0.7)
blau = (0.1,0.5,0.7)
verd_fosc = (0.3, 0.5, 0.2)
verd_blau = (0.2, 0.5, 0.45)
groc = (1, 0.8, 0.2)
ocre = (0.8,0.7,0)
negre = 'black'
gris = (0.5, 0.5, 0.5)
gris_fosc = (0.35, 0.35, 0.35)
gris_clar = (0.65, 0.65, 0.65)
gris_molt_clar = (0.85, 0.85, 0.85)
ocre_fosc = (217/256., 158/256., 13/256.)
blau_gris = (153/256., 196/256., 208/256.)
blau_gris_fosc = (11/256., 135/256., 170/256.)

# Tick and font sizes
tick_size = 35
text_size = 45
legend_size = 39
title_size = 50

# Line width
line_width = 2

# Dot (marker) size
dot_size = 12

# Font type
plt.rc("font", family='DejaVu Sans', size=tick_size, weight='light')

# Use latex fonts
#matplotlib.rcParams['mathtext.fontset'] = 'stix'
#matplotlib.rcParams['font.family'] = 'STIXGeneral'

plt.ion()

#......................................................................
#                             FUNCTIONS
#......................................................................

# ----------------------------------------
# Define N graded colors between c1 and c2
# ----------------------------------------
def define_graded_colors( c1, c2, N ):
    
    colors_N = [c1]
    
    if N > 1:
        
        dr = (c2[0]-c1[0]) / (N-1)
        dg = (c2[1]-c1[1]) / (N-1)
        db = (c2[2]-c1[2]) / (N-1)
        
        cc0 = c1[0]
        cc1 = c1[1]
        cc2 = c1[2]
        
        for j in range(N):
            
            cc0 += dr
            cc1 += dg
            cc2 += db
            
            colors_N.append( (cc0,cc1,cc2) )

    return( colors_N )

# -------------------------------------------------------------------------
# Set distance between ticks and axes and between labels and axes in a plot
# --------------------------------------------------------------------------

def set_tick_and_label_distances ( g1, lx, ly, tx, ty ):
    
    # Distance between labels and axes
    g1.xaxis.labelpad = lx
    g1.yaxis.labelpad = ly
        
    # Distance between ticks and axes
    g1.tick_params( axis = 'x', pad = tx )
    g1.tick_params( axis = 'y', pad = ty )
    
    g1.tick_params(
       axis='x',          # changes apply to the x-axis
       which='both',      # both major and minor ticks are affected
       bottom=True,       # ticks along the bottom edge are off
       top=False,         # ticks along the top edge are off
       labelbottom=True)  # labels along the bottom edge are off

    return

# --------------------------------------
# Define the directory and network names
# --------------------------------------

def define_dirs_and_names ( pdirectory, n, h, rho, real_network, weighted, read_coms, ref_index, type_fFG ):

    if real_network in [1,2,3,4,5,6]:
        
        if real_network == 1:
            net_name = "CElegans"
            weighted = 1
            type_fFG = 1 # WC
        elif real_network == 2:
            net_name = "Ciona"
        elif real_network == 3:
            net_name = "Mouse"
        elif real_network == 4:
            net_name = "Dupont2003"
            type_fFG = 3 # Ecology
        elif real_network == 5:
            net_name = "Clements1923"
            type_fFG = 3 # Ecology
        elif real_network == 6:
            net_name = "Maier2017"
            type_fFG = 2 # SIS
        
        if weighted == 1:
            net_name = net_name + "_weighted"
        
        pdirectory = pdirectory + "/real_network/" + net_name

    else:
        net_name = "h=%.2f_rho=%.2f" %(h,rho)
        pdirectory = pdirectory + "/%d_clusters" %n

    directory_networks = "%s" %pdirectory

    if type_fFG == 1:
        pdirectory = pdirectory + "/WC"
    elif type_fFG == 2:
        pdirectory = pdirectory + "/SIS"
    elif type_fFG == 3:
        pdirectory = pdirectory + "/Ecology"
    else:
        pdirectory = pdirectory + "/Default"

    # Community name
    if read_coms == 1:
        coms_name = "_coms_ref%d" %ref_index
    else:
        coms_name = ""

    return( pdirectory, directory_networks, net_name, coms_name, type_fFG )


# ------------------------------
# Read the reduction's dimension
# ------------------------------
def read_reduction_dimension ( directory, net_name, coms_name ):

    if coms_name != "":
        
        file_name = directory + "/community_sizes_%s%s.txt" %(net_name, coms_name)
        file = open( file_name, "r" )
        line = file.readline()
        line = file.readline()
        line = line.split(" ")
        n_coms = int(line[1])
        file.close()
    
    else:
        n_coms = 1

    return ( n_coms )

# --------------------------------------------------
# Reduce the range of plotted data in the bif. diag.
# --------------------------------------------------

def reduce_range_bd_data ( range_indexes, alpha, XMac, alphaRed, XMacRed ):
    
    nn = len( alpha )
    
    total_range = nn / 2
    range_ini = int( range_indexes[0] * total_range )
    range_fin = int( range_indexes[1] * total_range )
    range_params = range( range_ini, range_fin )
    
    range_params_rev = [ i for i in reversed( range_params ) ]
                
    alpha_plot = [alpha[i] for i in range_params] + [alpha[nn-i-1] for i in range_params_rev]
    XMac_plot = [XMac[i] for i in range_params] + [XMac[nn-i-1] for i in range_params_rev]
    alphaRed_plot = [alphaRed[i] for i in range_params] + [alphaRed[nn-i-1] for i in range_params_rev]
    
    XMacRed_plot = []
    for corr in range(len( XMacRed )):
        XMacRed_plot.append( [XMacRed[corr][i] for i in range_params] + [XMacRed[corr][nn-i-1] for i in range_params_rev] )

    return( alpha_plot, XMac_plot, alphaRed_plot, XMacRed_plot )


# ---------------------------------------------------
# Plot the bifurcation diagrams for a given partition
# ---------------------------------------------------

# redsV:        list that indicates the reduction methods that we want to plot:
#               0 (eigenval/spectral), 1 (naive/homogeneous), 2 (degree-based)
# obs_indiv:    plot or not the individual observables on a new figure
#               (only when the reduced dimension is 2)
# correction0:  correction that will be plotted

def plot_bif_diagram ( directory, directory_networks, h, rho, redsV, net_name, coms_name, obs_indiv, correction0 ):
    
    # Figure size
    wfig = 2.5 * wfig0
    hfig = 0.38 * wfig
    fig = pylab.figure( figsize=(wfig,hfig) )
    
    # Plot location and sizes
    x0 = 0.1
    y0 = 0.17
    mx = 0.23
    my = mx * wfig / hfig
    d = 0.07
    
    # Distance between labels and axes (lx, ly) and between ticks and axes (tx, ty)
    lx = 20
    ly = 55
    tx = 8
    ty = tx
    
    # Colors
    color_exact = taronja_vermellos
    color_red = blau
    color_corr1 = groc
    color_corr2 = verd
        
    # Line styles
    lwidth = 4
    ls_red = "dashed"
    ls_corr1 = "-."
    ls_corr2 = ":"
    alpha_exact = 0.8
    alpha_red = 0.7
    
    # Labels
    label_red = "reduced"
    label_corr1 = "corr 1"
    label_corr2 = "corr 2"

    labelX1 = r"${\cal K}_1$"
    labelY1 = r"${\cal X}_1$"
    labelY2 = r"${\cal X}_\nu$"
    labelXnu = r"$\langle {\cal K} \rangle$"
    labelYnu = r"$\langle {\cal X} \rangle$"

    titles = ["Spectral", "Homogeneous", "Degree-weighted"]
    method_names = ["eigenval", "naive", "degree"]
    corr_nums = [3, 1, 1]
    corrs_to_plot_v = [[correction0], [0], [0]] # Corrections that will be plotted for each reduction method
    text_colors = [ocre_fosc, verd_blau, negre]
    
    # Read the reduction's dimension
    n_coms = read_reduction_dimension ( directory_networks, net_name, coms_name )
    
    if obs_indiv == 1 and n_coms != 2:
        obs_indiv = 0
    else:
        fig2 = pylab.figure( figsize=( 2./3 * wfig, hfig) )
        mx2 = mx * 3./2
        d2 = d * 3./2
        x02 = x0 * 3./2

    # If n_coms > 1, the degree-based reduction does not exist
    if n_coms > 1:
        redsV.remove(2)

    # Number of plots (maximum 3)
    num = len(redsV)
    if num > 3:
        print( "redsV should be a list of at most 3 elements." )
        return
    
    # + + + + + + + + + + + + + + + + + +
    #       BIFURCATION DIAGRAMS
    # + + + + + + + + + + + + + + + + + +
    
    for ii in range( num ):
        
        red = redsV[ii]
        
        method_name = method_names[red]
        title = titles[red]
        text_color = text_colors[red]
        corrs_to_plot = corrs_to_plot_v[red]
        total_corrs = corr_nums[red]
        
        n = n_coms
        
        if n == 1:
            labelX = labelX1
            labelY = labelY1

        else:
            labelX = labelXnu
            labelY = labelYnu

        # -------------
        # Read the data
        # -------------
        
        # Exact data
        try:
            
            file_name = directory + "/bif_diagram_%s_%s_n=%d%s.txt" %(method_name, net_name, n, coms_name)
            
            # Read the average observable
            alpha, XMac = pylab.loadtxt( fname=file_name, usecols=(0,1), unpack=True )

            # Read each observable
            if obs_indiv == 1 and n > 1:
                xv = []
                for jj in range(n):
                    x = pylab.loadtxt( fname=file_name, usecols=(jj+2,), unpack=True )
                    xv.append( x )

        except IOError:
            print("\nI could not read the data from file\n%s\n" %file_name )
            continue

        # Reduced data
        XMacRed = []
        xvRed = []

        # Read all the corrections
        for correction in range( total_corrs ):
            
            if correction == 0:
                corr_name = ""
            else:
                corr_name = "_corr%d" %correction

            try:
                file_name_red = directory + "/bif_diagram_reduced_%s_%s_n=%d%s%s.txt" %(method_name, net_name, n, coms_name, corr_name )

                # Read the average observable
                alphaRed, XMacRed0 = pylab.loadtxt( fname=file_name_red, usecols=(0,1), unpack=True )
                XMacRed.append( XMacRed0 )

                # Read each observable
                if obs_indiv == 1 and n > 1:
                    xvRed0 = []
                    for jj in range(n):
                        xRed = pylab.loadtxt( fname=file_name_red, usecols=(jj+2,), unpack=True )
                        xvRed0.append( xRed )
                    xvRed.append( xvRed0 )

            except IOError:
                print("\nI could not read the data from file\n%s\n" %file_name_red )
                continue

        # -------------
        # Plot the data
        # -------------

        alpha_plot = alpha
        XMac_plot = XMac
        alphaRed_plot = alphaRed
        XMacRed_plot = XMacRed
        
        # If needed, we can reduce the range of the data that is shown in the diagram
        if net_name == "CElegans" and n != 1:
            
            range_indexes = [0, 1] # range of parameters shown, limits must be within [0,1]
            alpha_plot, XMac_plot, alphaRed_plot, XMacRed_plot = reduce_range_bd_data ( range_indexes, alpha, XMac, alphaRed, XMacRed )

        g1 = fig.add_axes( [ x0 + (ii%3)*(mx+d), y0, mx, my ] )
        
        # ···············································
        #      Plot the average observables' data
        # ···············································

        # Exact data
        g1.plot( alpha_plot, XMac_plot, color=color_exact, alpha=alpha_exact, ls="-", marker="", lw = lwidth, ms = dot_size, label = "exact", zorder = 0 )

        # Reduced data
        for corr in corrs_to_plot:
    
            if corr == 0:
                label_corr = label_red
                color_corr = color_red
                ls = ls_red
                zorder_corr = 1
            elif corr == 1:
                label_corr = label_corr1
                color_corr = color_corr1
                ls = ls_corr1
                zorder_corr = 2
            else:
                label_corr = label_corr2
                color_corr = color_corr2
                ls = ls_corr2
                zorder_corr = 3

            if len(corrs_to_plot) == 1:
                label_corr = label_red
                color_corr = color_red
                ls = ls_red
                zorder_corr = 1
            
            g1.plot( alphaRed_plot, XMacRed_plot[corr], color=color_corr, alpha=alpha_red, ls=ls, marker="", lw = lwidth, ms = dot_size+2, label = label_corr, zorder = zorder_corr )
        
        ############ To add a "fill_between"
        '''
        m = int(len(alphaRed_plot)/2)
        y1Red = [XMacRed_plot[0][i] for i in range(m)]
        y1 = [XMac_plot[i] for i in range(m)]
        x1 = [alpha_plot[i] for i in range(m)]

        y2Red = [XMacRed_plot[0][i] for i in range(m, 2*m)]
        y2 = [XMac_plot[i] for i in range(m, 2*m)]
        x2 = [alpha_plot[i] for i in range(m, 2*m)]
        
        color = verd_blau
        alpha=0.08
        g1.fill_between( x1, y1, y1Red, color=color, alpha=alpha )
        g1.fill_between( x2, y2, y2Red, color=color, alpha=alpha )
        '''
        ###########
        
        # Add title
        g1.text( 0.01, 1.04, " %s" %title, transform=g1.transAxes, c = text_color, fontweight="semibold" )
        g1.text( 0.97, 1.04, r"$n = %d$" %n, transform=g1.transAxes, horizontalalignment='right', fontweight="medium" )

        # Ticks and plot limits
        #yt = [0,4,8,12]
        #g1.set_ylim( [-0.3, 9.] )
        #g1.set_yticks( yt )
        
        # Labels and legend
        g1.set_xlabel( labelX, size=text_size )

        if method_name in ["naive", "degree"]:
            g1.legend( frameon = False, prop={'size':tick_size}, loc = (0.03,0.75) )

        if ii == 0:
            g1.set_ylabel( labelY, size=text_size )

        # Set distance between axes and ticks and between axes and labels
        set_tick_and_label_distances ( g1, lx, ly, tx, ty )

        # ···············································
        #      Plot the individual observables' data
        # ···············································
       
        if obs_indiv == 1 and n > 1:

            g2 = fig2.add_axes([ x02 + (ii%3)*(mx2+d2), y0, mx2, my ] )

            color_X1 = verd
            color_X2 = blau_gris_fosc
            
            for k in range(2):
                
                alpha_red2 = 1
                alpha_exact2 = 0.5
                
                if k == 0:
                    color_X = color_X1
                    label_aux = r"${\cal X}_1$"
                else:
                    color_X = color_X2
                    label_aux = r"${\cal X}_2$"
                
                g2.plot( alpha, xv[k], color=color_X, alpha=alpha_exact2, ls="-", marker="", lw = lwidth, ms = dot_size, zorder = 1 )
                
                for corr in corrs_to_plot:
                    
                    if corr == 0:
                        label_corr = label_red
                        color_corr = color_red
                        ls = ls_red
                        zorder_corr = 1
                    elif corr == 1:
                        label_corr = label_corr1
                        color_corr = color_corr1
                        ls = ls_corr1
                        zorder_corr = 2
                    else:
                        label_corr = label_corr2
                        color_corr = color_corr2
                        ls = ls_corr2
                        zorder_corr = 3

                    if len(corrs_to_plot) == 1:
                        label_corr = label_red
                        color_corr = color_X
                        ls = ls_red
                        zorder_corr = 1

                    g2.plot( alphaRed, xvRed[corr][k], color=color_corr, alpha=alpha_red2, ls=ls, marker="", lw = lwidth, ms = dot_size+2, zorder = zorder_corr ) #color=color_X

            # For the legend
            if method_name != "eigenval":
                g2.plot( [], [], color=gris_fosc, alpha=alpha_exact, ls="-", marker="", lw = lwidth, ms = dot_size, label = "exact" )
                g2.plot( [], [], color=gris_fosc, alpha=alpha_red, ls="dashed", marker="", lw = lwidth, ms = dot_size, label = "reduced" )
                
                leg = g2.legend( frameon = False, prop={'size':tick_size}, loc = (0.03,0.75), ncol = 1 )

            # Observable legend
            g2.text( 0.88, 0.49, r"${\cal X}_1$", transform=g2.transAxes, c = color_X1 ) # 0.5
            g2.text( 0.88, 0.74, r"${\cal X}_2$", transform=g2.transAxes, c = color_X2 ) # 0.81

            g2.text( 0.01, 1.04, " %s" %title, transform=g2.transAxes, c = text_color, fontweight="semibold" )
            g2.text( 0.97, 1.04, r"$n = %d$" %n, transform=g2.transAxes, horizontalalignment='right', fontweight="medium" )

            #g2.set_ylim( [-0.03, 1] )
            #g2.set_yticks( [0, 0.5, 1.] )
            g2.set_xlabel( labelX, size=text_size )

            if method_name == "naive":
                g2.set_ylabel( labelY2, size=text_size )

            # Set distance between axes and ticks and between axes and labels
            set_tick_and_label_distances ( g2, lx, ly, tx, ty )

    # Save
    file_name = directory + "/bif_diagram_%s%s" %(net_name, coms_name)
    file_name2 = file_name + "_n=%d_all" %n_coms + "."
    file_name = file_name + "_n=%d" %n_coms + "."

    print("\nFigure saved as:")
    for ftype in ftypes:
        fig.savefig( file_name + ftype )
        print("%s" %(file_name + ftype) )

    if obs_indiv == 1:
        print("\nFigure saved as:")
        for ftype in ftypes:
            fig2.savefig( file_name2 + ftype )
            print("%s" %(file_name2 + ftype) )

    plt.close()
    
    return

# ----------------------------------------------------------------------------------
# Errors in solving the comp. eq. for a given partition according to the group index
# ----------------------------------------------------------------------------------

def plot_errors_comp_eq_fixed_partition ( directory, directory_networks, net_name, coms_name ):

    # Figure size
    wfig = 1. * wfig0
    hfig = 0.8 * wfig
    fig = pylab.figure( figsize=(wfig,hfig) )
    
    # Plot location and sizes
    x0 = 0.28
    y0 = 0.17
    mx = 0.7
    my = 0.65

    # Distance between labels and axes (lx, ly) and between ticks and axes (tx, ty)
    lx = 20
    ly = 55
    tx = 10
    ty = tx
    
    # Colors and styles
    alpha = 0.6
    color_random = gris_molt_clar
    color = blau
    median_color = taronja
    
    # Read the reduction's dimension
    n = read_reduction_dimension ( directory_networks, net_name, coms_name )
    
    # -------------------
    # Read the error data
    # -------------------

    file_name = directory + "/errors_random_observables_n=%d.txt" %n
    mean_errors = pylab.loadtxt( fname=file_name, usecols=(0,), unpack=True )
    
    errors = []
    errors_random = []

    for i in range( n ):
        err = pylab.loadtxt( fname=file_name, usecols=(i+1,), unpack=True )
        errors.append( err[0] ) # the first datum is the observable's error
        errors_random.append( [err[i] for i in range(1, len(err))] )
    nerr = len( errors_random[0] )

    # -------------
    # Plot the data
    # -------------

    g1 = fig.add_axes( [ x0, y0, mx, my ] )

    # Errors for the random vectors
    
    labels = [ i+1 for i in range(n) ]
    flierprops = dict(marker='.', markerfacecolor=color_random, markeredgecolor=color_random, markersize=8, linestyle='', zorder = -1)
    boxprops = dict( facecolor = color_random )
    bplot = g1.boxplot( errors_random,
           vert=True,          # vertical box alignment
           patch_artist=True,  # fill with color
           #labels = labels,    # will be used to label x-ticks
           whis = [1,99],      # whiskers
           medianprops = {"color": median_color, "lw": 1.5},
           boxprops = boxprops,
           flierprops = flierprops
    )

    # Observable vectors' errors
    pl, = g1.plot( labels, errors, color=color, alpha = 1, ls="", marker="D", ms = 8 )

    # Legend
    g1.legend( [bplot["boxes"][0], pl], ["random vectors", "observable vector"], loc='upper right', frameon = False, prop={'size':legend_size-8}, bbox_to_anchor=(0.05, 1.15, 1, .102), ncol = 1 )

    # Ticks and labels
    g1.set_xticks( labels )
    
    if n >= 15:
        m = 3 # number of ticks until the next label
        g1.set_xticks( [m*i+1 for i in range(int(n/m)+1)] )
        g1.set_xticklabels( [m*i+1 for i in range(int(n/m)+1)] )

    #g1.set_yticks( [0, 0.05, 0.1] )
    g1.set_xlabel( "Group index" )
    g1.set_ylabel( "Normalized error" )

    # Set distance between axes and ticks and between axes and labels
    set_tick_and_label_distances ( g1, lx, ly, tx, ty )

    # Delete top and right axes
    g1.spines["top"].set_visible(False)
    g1.spines["right"].set_visible(False)

    # Save
    file_name = directory + "/errors_random_observables_n=%d." %n
    
    print("\nFigure saved as:")
    for ftype in ftypes:
        fig.savefig( file_name + ftype )
        print("%s" %(file_name + ftype) )

    plt.close()

    return


# ---------------------------------------------------------------------
# Mean errors in solving the comp. eq. according to the group number, n
# ---------------------------------------------------------------------

def plot_errors_comp_eq_n ( directory, directory_networks, net_name, range_refs, plot_random ):
    
    # Figure size
    wfig = 1. * wfig0
    hfig = 0.7 * wfig
    fig = pylab.figure( figsize=(wfig,hfig) )
    
    # Plot location and sizes
    x0 = 0.24
    y0 = 0.17
    mx = 0.75
    my = 0.65
    
    # Distance between labels and axes (lx, ly) and between ticks and axes (tx, ty)
    lx = 20
    ly = 50
    tx = 10
    ty = tx
    
    # Colors and styles
    alpha = 0.6
    color_random = gris_molt_clar
    color = blau
    median_color = taronja
    
    # -------------------
    # Read the error data
    # -------------------

    mean_errors = []
    mean_errors_random = []
    ns = []
    
    for ref in range_refs:
        
        if ref == -1:
            coms_name = ""
            n = 1
        
        else:
            coms_name = "_coms_ref%d" %ref
    
            # Read the reduction's dimension
            n = read_reduction_dimension ( directory_networks, net_name, coms_name )

        # If n coincides with the previous one, we skip it
        if len(ns) > 0 and n == ns[-1]:
            continue
    
        ns.append(n)
        
        file_name = directory + "/errors_random_observables_n=%d.txt" %n
        err = pylab.loadtxt( fname=file_name, usecols=(0,), unpack=True )
        mean_errors.append( err[0] ) # the first datum is the observable's error
        mean_errors_random.append( [err[i] for i in range(1, len(err))] )

    # -------------
    # Plot the data
    # -------------

    g1 = fig.add_axes( [ x0, y0, mx, my ] )
    labels = ns
    
    # Errors of the random vectors
    if plot_random == 1:
        
        flierprops = dict(marker='.', markerfacecolor=color_random, markeredgecolor=color_random, markersize=8, linestyle='', zorder = -1)
        boxprops = dict( facecolor = color_random )
        bplot = g1.boxplot( mean_errors_random,
           vert=True,          # vertical box alignment
           patch_artist=True,  # fill with color
           labels = labels,    # will be used to label x-ticks
           whis = [1,99],      # whiskers
           medianprops = {"color": median_color, "lw": 1.5},
           boxprops = boxprops,
           flierprops = flierprops
           )

    # Observable vectors' errors
    pl, = g1.plot( [1+i for i in range(len(labels))], mean_errors, color=color, alpha = 1, ls="", marker="D", ms = 8 )

    # Legend
    if plot_random == 1:
        g1.legend( [bplot["boxes"][0], pl], ["random vectors", "observable vector"], loc='upper right', frameon = False, prop={'size':legend_size-8}, bbox_to_anchor=(0.05, 1.15, 1, .102), ncol = 1 )

    # Ticks and labels
    g1.set_xticks( [1+i for i in range(len(labels))] )
    g1.set_xticklabels( labels )
    
    if len( ns ) >= 8:
        g1.xaxis.set_tick_params(labelsize=20)
    
    #g1.set_yticks( [0, 0.05, 0.1] )
    g1.set_xlabel( "Reduced dimension " + r"$n$" )
    g1.set_ylabel( "Normalized error" )

    # Set distance between axes and ticks and between axes and labels
    set_tick_and_label_distances ( g1, lx, ly, tx, ty )

    # Delete top and right axes
    g1.spines["top"].set_visible(False)
    g1.spines["right"].set_visible(False)

    # Save
    file_name = directory + "/mean_errors_random_observables."
    
    print("\nFigure saved as:")
    for ftype in ftypes:
        fig.savefig( file_name + ftype )
        print("%s" %(file_name + ftype) )

    plt.close()
                       
    return

# --------------------------------------------------------------
# Compare two ways of solving the comp. equations:
#
#   (a) Searching a solution within the subspace spanned by
#       the dominant eigenvectors of the matrices involved
#   (b) Searching a solution within the general space
#       (optimal solution)
#
# The matrices have been generated at random.
# --------------------------------------------------------------
#
# m_r: pairs (m, r) studied, where
#      m: matrix dimensions
#      r: number of matrices in the equations

def compare_sols ( directory, m_r ):
    
    # Figure size
    wfig = 1.7 * wfig0
    hfig = 1.15 * wfig
    fig = pylab.figure( figsize=(wfig,hfig) )
    fig2 = pylab.figure( figsize=(wfig,hfig) )

    text = r"$n$" + ": number of matrices\n" + r"$m$" + ": dimension of space"
    fig.text( 0.5, 0.93, text, horizontalalignment='center', size = title_size-8 )
    fig2.text( 0.5, 0.93, text, horizontalalignment='center', size = title_size-8 )
    
    # Plot location and sizes
    x0 = 0.14
    y0 = 0.94
    mx = 0.72
    my = 0.65
    
    # Distance between labels and axes (lx, ly) and between ticks and axes (tx, ty)
    lx = 30
    ly = 40
    tx = 5
    ty = tx
    
    alpha = 0.6
    color_random = gris_molt_clar
    color = blau
    
    errors = []
    distancies = []
    
    minAbs = 100
    maxAbs = 0
    
    for m, r in m_r:
    
        nomErr = directory + "/sols_compEq_errors_m=%d_r=%d.txt" %(m,r)
        nomDist = directory + "/sols_compEq_distances_m=%d_r=%d.txt" %(m,r)
    
        # Read data
        errSub, err = pylab.loadtxt( fname=nomErr, usecols=(0,1), unpack=True )
        dist = pylab.loadtxt( fname=nomDist, usecols=(0,), unpack=True )
    
        errors.append( [errSub, err] )
        distancies.append( dist )
    
        min = np.min( err )
        max = np.max( errSub )
    
        if min < minAbs:
            minAbs = min
        if max > maxAbs:
            maxAbs = max
    
    num_errors = 9 # number of error plots
    num_per_fila = 3
    dd = int( len(m_r) / num_errors )
    if dd == 0:
        quins = [0]
    else:
        quins = [ i * dd for i in range(num_errors) ]
    quants = len( quins )

    y00 = y0
    mx2 = 0.88 * mx / num_per_fila
    my2 = mx2 * wfig / hfig
    d = 1.15 * ( x0 + mx - mx2*num_per_fila ) / num_per_fila

    c1 = groc
    c2 = blau
    colors = define_graded_colors( c1, c2, len(m_r) )

    mida = legend_size+1
    mida_l = legend_size-2
    
    for ii in range( quants ):

        m = m_r[ quins[ii] ][0]
        r = m_r[ quins[ii] ][1]
        title = r"$n=%d, m=%d$" %(r,m)
        
        if ii % num_per_fila == 0:
            x00 = x0
            y00 -= my2 + d*1.1
        
        else:
            x00 += (mx2+d)

        min = 0
        min2 = 0
        if int (ii / num_per_fila) == 0:
            max = 20
            ticks = [0,10,20]
            max2 = 0.06
        elif int (ii / num_per_fila) == 1:
            max = 60
            ticks = [0,30,60]
            max2 = 0.03
        else:
            max = 200
            ticks = [0, 100, 200]
            max2 = 0.02
        ticks2 = [0, max2]
        
        # Errors
        g1 = fig.add_axes( [ x00, y00, mx2, my2 ] )
        g1.plot( errors[quins[ii]][1], errors[quins[ii]][0], color=colors[quins[ii]], alpha = 1, ls="", marker="o", ms = 6 )
        g1.plot( [min,max], [min,max], ls="dashed", color=gris_clar, marker=".", lw=line_width )
        g1.set_title( title, fontsize=mida_l, y=1.0, pad=25 )
        
        # Error in the "big" space vs distance
        g2 = fig2.add_axes( [ x00, y00, mx2, my2 ] )
        g2.plot( errors[quins[ii]][1], distancies[quins[ii]], color=colors[quins[ii]], alpha = 1, ls="", marker="o", ms = 6 )
        g2.set_title( title, fontsize=mida_l, y=1.0, pad=25 )

        lim = [min - (max-min)*0.05, max + (max-min)*0.05]
        g1.set_xlim( lim )
        g1.set_ylim( lim )
        g1.set_xticks( ticks )
        g1.set_yticks( ticks )
        
        lim2 = [min2 - (max2-min2)*0.05, max2 + (max2-min2)*0.05]
        g2.set_xlim( lim )
        g2.set_ylim( lim2 )
        g2.set_xticks( ticks )
        g2.set_yticks( ticks2 )

        if ii % num_per_fila == 0:
            g1.set_ylabel( r"$Err_{subspace}$", fontsize=mida )
            g2.set_ylabel( r"$\Vert a - a_{subspace} \Vert$", fontsize=mida )
        if ii >= quants - num_per_fila:
            g1.set_xlabel( r"$Err$", fontsize=mida )
            g2.set_xlabel( r"$Err$", fontsize=mida )
        
        # Set distance between axes and ticks and between axes and labels
        set_tick_and_label_distances ( g1, lx, ly, tx, ty )
        set_tick_and_label_distances ( g2, lx, ly, tx, ty )
        
    file_name = directory + "/sols_compEq_error_comparison."

    print("\nFigure saved as:")
    for ftype in ftypes:
        fig.savefig( file_name + ftype )
        print("%s" %(file_name + ftype) )

    file_name = directory + "/sols_compEq_error_vs_distance."

    print("\nFigure saved as:")
    for ftype in ftypes:
        fig2.savefig( file_name + ftype )
        print("%s" %(file_name + ftype) )

    plt.close()
    
    return

# -----------------------------------------------------------------------
# Plot the error in the bifurcation diagram as we vary the group number n
# -----------------------------------------------------------------------

def plot_errors_bif_diag_n ( directory, directory_networks, name_reds, net_name, range_refs, correction ):
    
    # Figure size
    wfig = 1.2 * wfig0
    hfig = 0.7 * wfig

    wfig = 0.95 * wfig0
    hfig = 0.75 * wfig
    
    fig = pylab.figure( figsize=(wfig,hfig) )
    
    # Plot location and sizes
    x0 = 0.21
    y0 = 0.21
    mx = 0.76
    my = 0.65
    
    # Distance between labels and axes (lx, ly) and between ticks and axes (tx, ty)
    lx = 40
    ly = 55
    tx = 10
    ty = tx
    
    # Colors
    color_naive = verd_blau
    color_eigenval = ocre_fosc
    
    alpha = 0.6
    
    # -------------
    # Read the data
    # -------------
    
    errors = [[] for i in range(len(name_reds))]
    ns = []
    
    for ref in range_refs:
        
        if ref == -1:
            coms_name = ""
            n = 1
    
        else:
            coms_name = "_coms_ref%d" %ref
            n = read_reduction_dimension ( directory_networks, net_name, coms_name )

        # If n coincides with the previous one, we skip it
        if len(ns) > 0 and n == ns[-1]:
            continue

        ns.append(n)

        for ii in range(len(name_reds)):
            
            name_red = name_reds[ii]
        
            if correction == 0 or name_red == "naive":
                corr_name = ""
            else:
                corr_name = "_corr%d" %correction

            file_name_red = directory + "/bif_diagram_reduced_%s_%s_n=%d%s%s.txt" %(name_red, net_name, n, coms_name, corr_name )
            file_name = directory + "/bif_diagram_%s_%s_n=%d%s.txt" %(name_red, net_name, n, coms_name )
            
            # Mean observable
            XMacRed = pylab.loadtxt( fname=file_name_red, usecols=(1,), unpack=True )
            XMac = pylab.loadtxt( fname=file_name, usecols=(1,), unpack=True )

            # Error (root-mean-square error, RMSE) between the exact curve and that of the reduced dynamics
            error = sum( [ (XMac[i] - XMacRed[i])**2 for i in range(len(XMac)) ] )
            error /= len(XMac)
            error = np.sqrt( error )
            errors[ii].append( error )

    # -------------
    # Plot the data
    # -------------

    g1 = fig.add_axes( [ x0, y0, mx, my ] )
    
    for ii in range(len(name_reds)):
        
        if name_reds[ii] == "eigenval":
            label = "Spectral"
            color = color_eigenval
            ls = "-"
        elif name_reds[ii] == "naive":
            label = "Homogeneous"
            color = color_naive
            ls = "-"
        
        g1.plot( [1+i for i in range(len(ns))], errors[ii], color=color, alpha = 1, ls=ls, marker=".", ms = dot_size+1, label = label, lw = line_width, zorder = -ii )

    # Ticks
    g1.set_xticks( [1+i for i in range(len(ns))] )
    g1.set_xticklabels( ns )

    if len( ns ) >= 8:
        g1.xaxis.set_tick_params(labelsize=25)
    
    #g1.set_yticks( [0, 0.05, 0.1] )

    # Axes labels
    g1.set_xlabel( "Reduced dimension " + r"$n$" )
    g1.set_ylabel( "RMSE" )

    # Legend
    g1.text( 0.5, 0.95, " Homogeneous", transform=plt.gca().transAxes, c = color_naive, fontweight="semibold", fontsize=tick_size-3.5 ) # l'última instrucció fa que la posició sigui la posició relativa dins de la imatge
    g1.text( 0.5, 0.83, " Spectral", transform=plt.gca().transAxes, c = color_eigenval, fontweight="semibold", fontsize=tick_size-3.5 )

    # Set distance between axes and ticks and between axes and labels
    set_tick_and_label_distances ( g1, lx, ly, tx, ty )

    # Delete top and right axes
    g1.spines["top"].set_visible(False)
    g1.spines["right"].set_visible(False)
    
    # Save
    file_name = directory + "/dynamical_errors_%s." %net_name

    print("\nFigure saved as:")
    for ftype in ftypes:
        fig.savefig( file_name + ftype )
        print("%s" %(file_name + ftype) )

    plt.close()
                   
    return

# -------------------------------------------------------------------------
# Plot the error in the bif. diagram as the original partition is perturbed
# -------------------------------------------------------------------------

def plot_errors_bif_diag_perturb ( directory, name_reds, net_name, corr_name_spec, n, relative_error ):
    
    # Figure size
    wfig = 0.95 * wfig0
    hfig = 0.75 * wfig
    fig = pylab.figure( figsize=(wfig,hfig) )
    
    # Plot location and sizes
    x0 = 0.21
    y0 = 0.2
    mx = 0.74
    my = 0.64
    
    # Distance between labels and axes (lx, ly) and between ticks and axes (tx, ty)
    lx = 30
    ly = 45
    tx = 10
    ty = tx
    
    # Colors
    color_naive = verd_blau
    color_eigenval = ocre_fosc
    
    # -------------
    # Read the data
    # -------------
    
    data_f = []
    data_mean = []
    data_std = []
    labels = [ 0 for i in range(len(name_reds)) ]
    
    for ii in range(len(name_reds)):
        
        name_red = name_reds[ii]
        
        if name_red == "eigenval":
            label = "Spectral"
            corr_name = corr_name_spec
        
        elif name_red == "naive":
            label = "Homogeneous"
            corr_name = ""
        
        labels[ii] = label
        
        file_name = directory + "/RMSE_perturb_%s_%s_n=%d%s.txt" %(name_red, net_name, n, corr_name )
        f, mean, std = pylab.loadtxt( fname=file_name, usecols=(0,1,2), unpack=True )
        
        if name_red == "eigenval":
            mean_spectral = mean[0]
        
        if relative_error == 1: # The errors are relative to the non-perturbed case (mean[0])
            mean0 = mean[0]
            for i in range(len(mean)):
                mean[i] = mean[i] / mean0
                std[i] = std[i] / mean0
        
        data_f.append( f )
        data_mean.append( mean )
        data_std.append( std )

    if relative_error == 2: # The errors are relative to the non-perturbed AND spectral case (mean_spectral)
        for ii in range(len(name_reds)):
            for i in range(len(mean)):
                data_mean[ii][i] /= mean_spectral
                data_std[ii][i] /= mean_spectral

    g1 = fig.add_axes( [ x0, y0, mx, my ] )

    colors = define_graded_colors( ocre_fosc, verd_blau, len(name_reds) )

    for ii in range(len(name_reds)):
        
        # Mean error
        g1.plot( data_f[ii], data_mean[ii], color=colors[ii], alpha = 1, ls="-", lw = line_width-0.5, marker=".", ms = dot_size+1, label = labels[ii] )
    
        # Standard deviation
        g1.fill_between( data_f[ii], data_mean[ii] - data_std[ii], data_mean[ii] + data_std[ii], alpha=0.15, color = colors[ii] )

    g1.text( 0.01, 1.17, " Homogeneous", transform=plt.gca().transAxes, c = color_naive, fontweight="semibold", fontsize=tick_size-3 )
    g1.text( 0.01, 1.05, " Spectral", transform=plt.gca().transAxes, c = color_eigenval, fontweight="semibold", fontsize=tick_size-3 )

    g1.set_xticks( [0, 0.2, 0.4, 0.6, 0.8, 1] )
    
    if n == 13:
        g1.set_yticks( [1, 6, 11, 16] )
    elif n == 5:
        g1.set_yticks( [1, 4, 7, 10, 13] )
    elif n == 2:
        g1.set_yticks( [1, 4, 7, 10] )

    g1.set_xlabel( r"$f$" + " (flips" + r"$/N$)" )
    g1.set_ylabel( "RMSE" )

    if relative_error == 1 or relative_error == 2:
        g1.set_ylabel( "Relative RMSE" )

    # Title
    g1.set_title( r"$n = %d$" %n, fontsize=legend_size, pad=25 ) #y=1.0,

    # Set distance between axes and ticks and between axes and labels
    set_tick_and_label_distances ( g1, lx, ly, tx, ty )

    # Delete top and right axes
    g1.spines["top"].set_visible(False)
    g1.spines["right"].set_visible(False)

    # Save
    file_name = directory + "/RMSE_perturb_%s_n=%d." %(net_name, n)
    
    print("\nFigure saved as:")
    for ftype in ftypes:
        fig.savefig( file_name + ftype )
        print("%s" %(file_name + ftype) )

    plt.close()

    return


#....................................................................................
#                                  PARAMETERS
#....................................................................................

#pdirectory0 = "/Users/marina/Documents/Postdoc/OneDrive - Université Laval/Dimension_reduction"
pdirectory0 = str( sys.argv[1] )

# Number of groups / clusters / communities
n = int( sys.argv[2] )

# Heterogeneity level, 0 <= h <= 1
h = float( sys.argv[3] )

# Correlation coefficient between hidden in/out-degrees within communities
# (only important when h > 0)
rho = float( sys.argv[4] )

# Real network
#   1 -> CElegans
#   2 -> Ciona
#   3 -> Mouse
#   4 -> Dupont2003
#   5 -> Clements1923
#   6 -> Maier2017
# def -> No real net.
real_network = int( sys.argv[5] )

read_coms = int( sys.argv[6] ) # 1->coms, def->no coms
ref_index = int( sys.argv[7] ) # refinement index when read_coms == 1

correction = int( sys.argv[8] )
what_to_plot = int( sys.argv[9] ) # 0->bif. diag., 1->error for perturbed partitions

type_fFG = 1 # 1->WC, 2->SIS, 3->Ecology
weighted = 0

pdirectory, directory_networks, net_name, coms_name, type_fFG = define_dirs_and_names ( pdirectory0, n, h, rho, real_network, weighted, read_coms, ref_index, type_fFG )

#....................................................................................
#                                  MAKE PLOTS
#....................................................................................

# -------------------
# Bifurcation diagram
# -------------------
if what_to_plot == 0:
    
    redsV = [1, 0, 2]   # reduction methods plotted: 0->spectral, 1->homogeneous, 2->degree
    obs_indiv = 1       # plot also the individual observables (only when n=2)
    plot_bif_diagram ( pdirectory, directory_networks, h, rho, redsV, net_name, coms_name, obs_indiv, correction )


# -----------------------------------------------------------------
# Errors in the bif. diagram as the original partition is perturbed
# -----------------------------------------------------------------
if what_to_plot == 1:

    name_reds = ["eigenval", "naive"]
    n = read_reduction_dimension ( directory_networks, net_name, coms_name )

    if correction == 0:
        corr_name_spec = ""
    else:
        corr_name_spec = "_corr%d" %correction

    # Error relative to
    #   1 -> the non-perturbed case for each method
    #   2 -> the non-perturbed case of the spectral method
    #  def -> no relative error
    relative_error = 2

    plot_errors_bif_diag_perturb ( pdirectory, name_reds, net_name, corr_name_spec, n, relative_error )

# --------------------------------------------------------------
# Error in the bifurcation diagram as we vary the group number n
# --------------------------------------------------------------
if what_to_plot == 2:

    name_reds = ["eigenval", "naive"]
    range_refs = [-1] + [i for i in range(10)]  # indices of the refinaments considered
                                               # -1 indicates that the whole net. is taken as a group
    range_refs = [-1, 0, 1, 4, 5, 6, 7, 8, 9]
    plot_errors_bif_diag_n ( pdirectory, directory_networks, name_reds, net_name, range_refs, correction )

# ---------------------------------------------
# Errors in solving the compatibility equations
# ---------------------------------------------
if what_to_plot == 3:

    # Errors for a fixed partition and according to group index
    plot_errors_comp_eq_fixed_partition ( pdirectory, directory_networks, net_name, coms_name )

    # Mean errors for a collection of partitions
    range_refs = [-1] + [i for i in range(6)]  # indices of the refinaments considered
                                                # -1 indicates that the whole net. is taken as a group
    plot_random = 1 # add the errors associated to the collection of random vectors
    plot_errors_comp_eq_n ( pdirectory, directory_networks, net_name, range_refs, plot_random )

# --------------------------------------------------------------
# Compare two ways of solving the comp. equations:
#
#   (a) Searching a solution within the subspace spanned by
#       the dominant eigenvectors of the matrices involved
#   (b) Searching a solution within the general space
#       (optimal solution)
#
# The matrices have been generated at random.
# --------------------------------------------------------------
if what_to_plot == 4:
    
    m_r = [[2,2], [4,2], [6,2], [3,3], [5,3], [7,3], [5,5], [7,5], [9,5]]
    compare_sols ( pdirectory0 + "/Tests", m_r )

print("\n")
