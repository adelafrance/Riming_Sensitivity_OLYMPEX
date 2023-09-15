"""
Andrew DeLaFrance
adelaf@uw.edu
September 2023

Script Function: 
Process and plot model output from 
McSnow simulations as generated in 
DeLaFrance et al., submitted to 
JAMES April 2023
Revised September 2023

Notes: 
*Script is intended to be run in sections
according to user need. Sections are designated 
by #%%. 
*Sections are also controlled by user-defined
list of True/False run options for each section
if running program in entirety is desired 
(i.e., from command line)
*Some user-defined variables need to be specified
Date: DEC3 or NOV13
Simulation: CONTROL, rime_light, rime_heavy, rime_shallow,
rime_deep, rime_off, updraft_increase, updraft_decrease, 
rime_dmsmall, rime_dmbig
"""

#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib import gridspec
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats
from scipy import optimize
from scipy import linspace
from scipy import pi,sqrt,exp
from scipy.special import erf
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit, leastsq
from scipy.ndimage.filters import gaussian_filter1d
import sys, os
dir_path_scripts = os.path.dirname(os.path.realpath(__file__)) + '/'
dir_path = dir_path_scripts + '../'
sys.path.append(dir_path_scripts)
from read_model_output_2D import strip_ascii_data
np.seterr(invalid='ignore')
import warnings
warnings.filterwarnings('ignore')

#%%
"""
SIMULATION SELECTION
"""
simulation = 'CONTROL'
#datestr = 'NOV13'
datestr = 'DEC3'

"""
DEFINED VARIABLES 
"""
t_ind = 24 #time to process data at number of 10 minute intervals
t_ind_stable = 12 #~ 2 hours = steady state
nx = 125
ny = 125
xdim = 200 #km
zmax = 6000 #m
h1 = 2750 #m; option for evaluating statistics between heights h1 and h2
h2 = 3250 #m; option for evaluating statistics between heights h1 and h2
save_fn_lab = 'A' #optional label on saved file name 

write_out_md_prof_data = True
save_Md_fig_output = True
save_Vt_Dm_Ar_Rf_fig_output = True 

plot_histograms = False 
plot_Md_timesteps = False
plot_Vt_Dm_rho_Rf = True
plot_Md_plus_lines = True
plot_LWC_sensitivity = True
plot_LWCdm_sensitivity = True
plot_depth_sensitivity = True
plot_updraft_sensitivity = True
plot_prcp_LWC_sensitivity = True
plot_prcp_depth_sensitivity = True

blackcolor = '#323232'
greycolor = '#bdbdbd'
greencolor = '#1b9e77'
violetcolor = '#7570b3'
orangecolor = '#d95f02'
bluecolor = '#1f78b4'
redcolor = '#ef8a62'
redcolor = '#d6604d'

lightgreencolor = '#66c2a5'
lightvioletcolor = '#8da0cb'
lightorangecolor = '#fc8d62'

#%%
"""
DATA INTAKE
"""
sim_folder = dir_path + 'Simulation_Output'
save_fn_path = dir_path + 'Figures_' + datestr + '/'

if datestr == 'DEC3':
    base_dir_control = sim_folder + '/2d_oly_rev_xi100_nz125_lwc73_iwc349_dtc10_nrp9_nugam2.2_rm13_rt1_mt0_vt1_h2750_h0-4750_break0_wwind_Rimeout_CONTROL_Rev/'
    base_dir_rime_dmsmall = sim_folder + '/2d_oly_rev_xi100_nz125_lwc73_iwc349_dtc10_nrp9_nugam2.2_rm9_rt1_mt0_vt1_h2750_h0-4750_break0_wwind_Rimeout_rime_dmsmall_Rev/'
    base_dir_rime_dmbig = sim_folder + '/2d_oly_rev_xi100_nz125_lwc73_iwc349_dtc10_nrp9_nugam2.2_rm18_rt1_mt0_vt1_h2750_h0-4750_break0_wwind_Rimeout_rime_dmbig_Rev/'
    base_dir_rime_heavy = sim_folder + '/2d_oly_rev_xi100_nz125_lwc103_iwc349_dtc10_nrp9_nugam2.2_rm13_rt1_mt0_vt1_h2750_h0-4750_break0_wwind_Rimeout_rime_heavy_Rev/'
    base_dir_rime_light = sim_folder + '/2d_oly_rev_xi100_nz125_lwc44_iwc349_dtc10_nrp9_nugam2.2_rm13_rt1_mt0_vt1_h2750_h0-4750_break0_wwind_Rimeout_rime_light_Rev/'
    base_dir_rime_shallow = sim_folder + '/2d_oly_rev_xi100_nz125_lwc73_iwc349_dtc10_nrp9_nugam2.2_rm13_rt1_mt0_vt1_h2750_h0-4000_break0_wwind_Rimeout_rime_shallow_Rev/'
    base_dir_rime_deep = sim_folder + '/2d_oly_rev_xi100_nz125_lwc73_iwc349_dtc10_nrp9_nugam2.2_rm13_rt1_mt0_vt1_h2750_h0-5500_break0_wwind_Rimeout_rime_deep_Rev/'
    base_dir_rime_off = sim_folder + '/2d_oly_rev_xi100_nz125_lwc0_iwc349_dtc10_nrp9_nugam2.2_rm13_rt0_mt0_vt1_h2750_h0-4750_break0_wwind_Rimeout_rime_off_Rev/'
    base_dir_updraft_increase = sim_folder + '/2d_oly_rev_xi100_nz125_lwc73_iwc349_dtc10_nrp9_nugam2.2_rm13_rt1_mt0_vt1_h2750_h0-4750_break0_wwind_Rimeout_updraft_increase_Rev/'
    base_dir_updraft_decrease = sim_folder + '/2d_oly_rev_xi100_nz125_lwc73_iwc349_dtc10_nrp9_nugam2.2_rm13_rt1_mt0_vt1_h2750_h0-4750_break0_wwind_Rimeout_updraft_decrease_Rev/'


elif datestr == 'NOV13':
    base_dir_control = sim_folder + '/2d_oly_13Nov_rev_xi100_nz125_lwc62_iwc267_dtc10_nrp22_nugam0.008_rm12_rt1_mt0_vt1_h2200_h0-5750_break0_wwind_Rimeout_CONTROL_Rev/'
    base_dir_rime_dmsmall = sim_folder + '/2d_oly_13Nov_rev_xi100_nz125_lwc62_iwc267_dtc10_nrp22_nugam0.008_rm9_rt1_mt0_vt1_h2200_h0-5750_break0_wwind_Rimeout_rime_dmsmall_Rev/'
    base_dir_rime_dmbig = sim_folder + '/2d_oly_13Nov_rev_xi100_nz125_lwc62_iwc267_dtc10_nrp22_nugam0.008_rm18_rt1_mt0_vt1_h2200_h0-5750_break0_wwind_Rimeout_rime_dmbig_Rev/'
    base_dir_rime_light = sim_folder + '/2d_oly_13Nov_rev_xi100_nz125_lwc17_iwc267_dtc10_nrp22_nugam0.008_rm12_rt1_mt0_vt1_h2200_h0-5750_break0_wwind_Rimeout_rime_light_Rev/'
    base_dir_rime_heavy = sim_folder + '/2d_oly_13Nov_rev_xi100_nz125_lwc171_iwc267_dtc10_nrp22_nugam0.008_rm12_rt1_mt0_vt1_h2200_h0-5750_break0_wwind_Rimeout_rime_heavy_Rev/'
    base_dir_rime_shallow = sim_folder + '/2d_oly_13Nov_rev_xi100_nz125_lwc62_iwc267_dtc10_nrp22_nugam0.008_rm12_rt1_mt0_vt1_h2200_h0-4000_break0_wwind_Rimeout_rime_shallow_Rev/'
    base_dir_rime_off = sim_folder + '/2d_oly_13Nov_rev_xi100_nz125_lwc0_iwc267_dtc10_nrp22_nugam0.008_rm12_rt0_mt0_vt1_h2200_h0-5750_break0_wwind_Rimeout_rime_off_Rev/'
    base_dir_updraft_increase = sim_folder + '/2d_oly_13Nov_rev_xi100_nz125_lwc62_iwc267_dtc10_nrp22_nugam0.008_rm12_rt1_mt0_vt1_h2200_h0-5750_break0_wwind_Rimeout_updraft_increase_Rev/'
    base_dir_updraft_decrease = sim_folder + '/2d_oly_13Nov_rev_xi100_nz125_lwc62_iwc267_dtc10_nrp22_nugam0.008_rm12_rt1_mt0_vt1_h2200_h0-5750_break0_wwind_Rimeout_updraft_decrease_Rev/'


if simulation == 'CONTROL':
    base_dir = base_dir_control
elif simulation == 'rime_light':
    base_dir = base_dir_rime_light
elif simulation == 'rime_heavy':
    base_dir = base_dir_rime_heavy
elif simulation == 'rime_shallow':
    base_dir = base_dir_rime_shallow
elif simulation == 'rime_deep':
    base_dir = base_dir_rime_deep
elif simulation == 'rime_off':
    base_dir = base_dir_rime_off
elif simulation == 'rime_dmsmall':
    base_dir = base_dir_rime_dmsmall
elif simulation == 'rime_dmbig':
    base_dir = base_dir_rime_dmbig
elif simulation == 'updraft_increase':
    base_dir = base_dir_updraft_increase
elif simulation == 'updraft_decrease':
    base_dir = base_dir_updraft_decrease


Md_fn = [base_dir + 'hei2massdens2D_Md.dat', base_dir + 'hei2massdens2D_Md_UnrPris.dat', base_dir + 'hei2massdens2D_Md_UnrAgg.dat', base_dir + 'hei2massdens2D_Md_Liq.dat', base_dir + 'hei2massdens2D_Md_Grp.dat', base_dir + 'hei2massdens2D_Md_Rime.dat']
Nd_fn = [base_dir + 'hei2massdens2D_Nd.dat', base_dir + 'hei2massdens2D_Nd_UnrPris.dat', base_dir + 'hei2massdens2D_Nd_UnrAgg.dat', base_dir + 'hei2massdens2D_Nd_Liq.dat', base_dir + 'hei2massdens2D_Nd_Grp.dat', base_dir + 'hei2massdens2D_Nd_Rime.dat']
Ar_fn = [base_dir + 'hei2massdens2D_Ar.dat'] #Area ratio
Vt_fn = [base_dir + 'hei2massdens2D_Vt.dat'] #Velocity
Dm_fn = [base_dir + 'hei2massdens2D_Dm.dat'] #Diameter-ave.
Rf_fn = [base_dir + 'hei2massdens2D_Rf.dat'] #Riming fraction
Fm_fn = [base_dir + 'hei2massdens2D_Fm.dat'] #Mass flux
Vol_fn = [base_dir + 'hei2massdens2D_Vol.dat'] #Volume (effective)

elev_45deg_fn = dir_path + 'DATA/elev_45deg.npy'
elev_45deg = np.load(elev_45deg_fn) #[dist, elev], elevation in meters
elev_45deg[1, elev_45deg[1,:] < 0] = 0

dist_offset = 0 #km; currently unused option to shift origin point horizontally 
elev_45deg[0, :] = elev_45deg[0, :] + dist_offset


#%%
"""
GENERALIZED FUNCTION FOR PLOTTING MODEL OUTPUT 
"""
def generate_plot(multipanel_flag, var_fn, t_ind, cbar_lab, cmap, vmin, vmax, log_flag, dist_text_flag, type_text, type_text_flag,
    xticks, xlabs, nx, xdim, mode_flag, var_scaling, t_hrs, save_fn_var, save_fn_lab, save_fn_time, save_fn_path):

    if multipanel_flag:
        z_list, var, time_list = strip_ascii_data(var_fn[0], t_ind)
        z_list, var_UnrPris, time_list = strip_ascii_data(var_fn[1], t_ind)
        z_list, var_UnrAgg, time_list = strip_ascii_data(var_fn[2], t_ind)
        z_list, var_Liq, time_list = strip_ascii_data(var_fn[3], t_ind)
        z_list, var_Grp, time_list = strip_ascii_data(var_fn[4], t_ind)
        z_list, var_Rime, time_list = strip_ascii_data(var_fn[5], t_ind)

        nz=len(z_list)
        ztickposind = [min(range(len(z_list)), key=lambda i: abs(z_list[i]-x)) for x in z_list]

        ytickpos = np.arange(0,np.nanmax(z_list)+1000,1000)
        yticks = [min(range(len(z_list)), key=lambda i: abs(z_list[i]-x)) for x in ytickpos]
        ylabs = [int(np.round(z_list[x]/1000,0)) for x in yticks]


        end_var_2D = var[t_ind,:,:]
        var_fmtd = end_var_2D * var_scaling

        end_var_UnrPris_2D = var_UnrPris[t_ind,:,:]
        var_UnrPris_fmtd = end_var_UnrPris_2D * var_scaling

        end_var_UnrAgg_2D = var_UnrAgg[t_ind,:,:]
        var_UnrAgg_fmtd = end_var_UnrAgg_2D * var_scaling

        end_var_Liq_2D = var_Liq[t_ind,:,:]
        var_Liq_fmtd = end_var_Liq_2D * var_scaling

        end_var_Grp_2D = var_Grp[t_ind,:,:]
        var_Grp_fmtd = end_var_Grp_2D * var_scaling

        end_var_Rime_2D = var_Rime[t_ind,:,:]
        var_Rime_fmtd = end_var_Rime_2D * var_scaling

        var_fmtd[var_fmtd == 0] = np.nan
        var_UnrPris_fmtd[var_UnrPris_fmtd == 0] = np.nan
        var_UnrAgg_fmtd[var_UnrAgg_fmtd == 0] = np.nan
        var_Liq_fmtd[var_Liq_fmtd == 0] = np.nan
        var_Grp_fmtd[var_Grp_fmtd == 0] = np.nan
        var_Rime_fmtd[var_Rime_fmtd == 0] = np.nan

        plt.rcParams.update({'font.size': 16})
        fig, ax = plt.subplots(nrows = 2, ncols = 2, figsize = (10,8))

        titles = ['Unrimed Pristine', 'Unrimed Aggregates', 'Rimed', 'Liquid']
        vars = [var_UnrPris_fmtd, var_UnrAgg_fmtd, var_Rime_fmtd, var_Liq_fmtd]

        if mode_flag and t_ind >= t_ind_stable:
            mode_var = [np.where(var_fmtd[i,:] == np.nanmax(var_fmtd[i,:]))[0][0] for i in range(var_fmtd.shape[0])]
            z = np.polyfit(ztickposind,np.array(mode_var), 3)
            f = np.poly1d(z)
            mode_y = np.linspace(ztickposind[0], ztickposind[-1], 50)
            mode_x = f(mode_y)
            ax.plot(mode_x, mode_y, color = 'lightgray', linewidth = 2, linestyle = '--')

        bbox = {'facecolor': 'whitesmoke', 'alpha': 1.0, 'linewidth': 1.0, 'boxstyle': 'round,pad=0.5'}


        n = 0
        for row in range(2):
            for col in range(2):
                axs = ax[row, col]
                var_n = vars[n]
                if log_flag:
                    im = axs.pcolormesh(var_n, cmap = cmap, norm=LogNorm(vmin=vmin, vmax=vmax))
                else:
                    im = axs.pcolormesh(var_n, vmin = vmin, vmax = vmax, cmap = cmap)
                cbar = fig.colorbar(im, ax = axs)
                cbar.set_label(cbar_lab)
                n += 1

        n = 0
        for row in range(2):
            for col in range(2):

                ax[row, col].set_xticks(xticks)
                ax[row, col].set_xticklabels(xlabs)
                ax[row, col].set_xlabel('Distance from Origin (km)')
                if datestr == 'DEC3':
                    ax[row, col].set_xlim(0,nx*0.6)
                elif datestr == 'NOV13':
                    ax[row, col].set_xlim(0,(nx*0.6+dist_offset * nx / 200)) #100 + offset km

                ax[row, col].set_yticks(yticks)
                ax[row, col].set_yticklabels(ylabs)
                ax[row, col].set_ylabel('Height (km)')

                ax[row, col].grid(linestyle = '--')

                if dist_text_flag:
                    dist_text = '\u0394x: ' + str(np.round(max(mode_x*xdim/nx),1)) + ' km'
                    ax[row, col].text(0.04, 0.04, dist_text, transform=ax[row, col].transAxes, color = 'black',
                        verticalalignment='bottom', horizontalalignment='left', bbox=bbox, zorder = 5)

                if type_text_flag:
                    title = titles[n]
                    ax[row, col].text(0.5, 0.96, title, transform=ax[row, col].transAxes, color = 'black',
                        verticalalignment='top', horizontalalignment='center', bbox=bbox, zorder = 5)
                n += 1
        save_fn = save_fn_path + 'test_McSnow_2D_'+save_fn_var+'_' + datestr + '_at'+save_fn_time+'hrs_z'+str(nz)+'_x'+str(nx)+'_grid_w_SoundingWind_'+save_fn_lab+'categorized.png'
        fig.tight_layout()
        plt.savefig(save_fn, dpi = 300)
        plt.show()
        plt.close()

    else:
        z_list, var, time_list = strip_ascii_data(var_fn[0], t_ind)
        nz=len(z_list)
        ztickposind = [min(range(len(z_list)), key=lambda i: abs(z_list[i]-x)) for x in z_list]

        ytickpos = np.arange(0,np.nanmax(z_list)+1000,1000)
        yticks = [min(range(len(z_list)), key=lambda i: abs(z_list[i]-x)) for x in ytickpos]
        ylabs = [int(np.round(z_list[x]/1000,0)) for x in yticks]

        end_var_2D = var[t_ind,:,:]
        end_var_2D[end_var_2D == 0] = np.nan

        var_fmtd = end_var_2D * var_scaling

        z_list, var_prcntl, time_list = strip_ascii_data(Md_fn[0], t_ind)
        var_prcntl_fmtd = var_prcntl[t_ind,:,:]
        var_prcntl_fmtd[var_prcntl_fmtd == 0] = np.nan

        plt.rcParams.update({'font.size': 16})
        fig, ax = plt.subplots(figsize = (6.4, 4.8))
        if log_flag:
            im = ax.pcolormesh(var_fmtd, cmap = cmap, norm=LogNorm(vmin=vmin, vmax=vmax))
        else:
            im = ax.pcolormesh(var_fmtd, vmin = vmin, vmax = vmax, cmap = cmap)

        if save_fn_var == 'Md':
            cbar = fig.colorbar(im, ax = ax, ticks = np.arange(vmin, vmax+0.1, 0.1))
        else:
            cbar = fig.colorbar(im, ax = ax)
        cbar.set_label(cbar_lab)

        if mode_flag and t_ind >= t_ind_stable:
            prcntl = 15
            
            #EXPONENTIAL MODE APPROACH TO FITTING TRAJECTORY
            def func(x, a, b, c):
                x1, a1, b1, c1 = np.array(x, dtype = np.float128), np.array(a, dtype = np.float128), np.array(b, dtype = np.float128), np.array(c, dtype = np.float128)
                r1 = a1 * (1.0-np.exp(-1.0 * b1 * x1)) + c1
                r6 = np.array(r1, dtype = np.float64)
                return r6


            minn = [np.nanmin(np.where(var_prcntl_fmtd[i,:] >= np.nanpercentile(var_prcntl_fmtd[i,:],(100-prcntl)))) for i in range(var_prcntl_fmtd.shape[0])]
            popt, pcov = curve_fit(func, ztickposind,np.array(minn[::-1]))
            mode_y = np.linspace(ztickposind[0], ztickposind[-1], 50)
            mode_x = func(mode_y, *popt)
            mode_y = mode_y[::-1]
            ax.plot(mode_x, mode_y, color = 'lightgray', linewidth = 2, linestyle = '-')

            maxx = [np.nanmax(np.where(var_prcntl_fmtd[i,:] >= np.nanpercentile(var_prcntl_fmtd[i,:],(100-prcntl)))) for i in range(var_prcntl_fmtd.shape[0])]
            popt, pcov = curve_fit(func, ztickposind,np.array(maxx[::-1]))
            mode_y = np.linspace(ztickposind[0], ztickposind[-1], 50)
            mode_x = func(mode_y, *popt)
            mode_y = mode_y[::-1]
            ax.plot(mode_x, mode_y, color = 'lightgray', linewidth = 2, linestyle = '-')

            mode_var = [np.where(var_prcntl_fmtd[i,:] == np.nanmax(var_prcntl_fmtd[i,:]))[0][0] for i in range(var_prcntl_fmtd.shape[0])]
            popt, pcov = curve_fit(func, ztickposind,np.array(mode_var[::-1]))
            mode_y = np.linspace(ztickposind[0], ztickposind[-1], 50)
            mode_x = func(mode_y, *popt)
            mode_y = mode_y[::-1]
            ax.plot(mode_x, mode_y, color = 'lightgray', linewidth = 2, linestyle = '--')

            if save_Vt_Dm_Ar_Rf_fig_output  and save_fn_var == 'Md':
                np.save(dir_path + 'Data/' + save_fn_var+'_McSnow_2D_SIM'+'_' + datestr + '_at'+save_fn_time +
                        'hrs_z'+str(nz)+'_x'+str(nx)+'_grid_'+save_fn_lab+'_mode_x_'+simulation+'.npy', mode_x)
                np.save(dir_path + 'Data/' + save_fn_var+'_McSnow_2D_SIM'+'_' + datestr + '_at' +
                        save_fn_time+'hrs_z'+str(nz)+'_x'+str(nx)+'_grid_'+save_fn_lab+'_mode_y_'+simulation+'.npy', mode_y)


            if save_Vt_Dm_Ar_Rf_fig_output and (save_fn_var == 'Vt' or save_fn_var == 'Ar' or save_fn_var == 'Dm' or save_fn_var == 'Rf' or save_fn_var == 'Vol'):
                lst_mode = [var_fmtd[aa, mode_var[aa]] for aa in range(len(mode_var))]
                lst_minn = [var_fmtd[aa, minn[aa]] for aa in range(len(minn))]
                lst_maxx = [var_fmtd[aa, maxx[aa]] for aa in range(len(maxx))]
                model_md_ht_50_15_85 = np.vstack((np.array(z_list), np.array(lst_mode), np.array(lst_minn), np.array(lst_maxx)))
                np.save(dir_path + 'Data/' + save_fn_var+'_McSnow_2D_SIM'+'_' + datestr + '_at'+save_fn_time +
                        'hrs_z'+str(nz)+'_x'+str(nx)+'_grid_'+save_fn_lab+'_traj_50_15_85_'+simulation+'.npy', model_md_ht_50_15_85)

            if write_out_md_prof_data == True and dist_text_flag:
                lst_mode = [var_scaling * var_prcntl_fmtd[aa, mode_var[aa]] for aa in range(len(mode_var))]
                lst_minn = [var_scaling * var_prcntl_fmtd[aa, minn[aa]] for aa in range(len(minn))]
                lst_maxx = [var_scaling * var_prcntl_fmtd[aa, maxx[aa]] for aa in range(len(maxx))]
                model_md_ht_50_15_85 = np.vstack((np.array(z_list), np.array(lst_mode), np.array(lst_minn), np.array(lst_maxx)))
                np.save(dir_path + 'Data/' + save_fn_var+'_McSnow_2D_SIM'+'_' + datestr + '_at'+save_fn_time +
                        'hrs_z'+str(nz)+'_x'+str(nx)+'_grid_'+save_fn_lab+'_traj_50_15_85_'+simulation+'.npy', model_md_ht_50_15_85)

        ax.plot(dists_conv_2_nx, elevs_conv_2_ny, color = 'gray', linewidth = 2)
        ax.fill_between(dists_conv_2_nx, np.full(len(elevs_conv_2_ny), 0), y2 = elevs_conv_2_ny, color = 'gray', alpha = 0.75)

        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabs)
        ax.set_xlabel('Distance from Origin (km)')
        ax.set_xlim(0,nx*0.6) 

        z6km_i = np.where(np.array(ylabs) == 6)[0][0]
        z6km_lev = yticks[z6km_i]

        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabs)
        ax.set_ylabel('Height (km)')

        ax.grid(linestyle = '--')

        bbox = {'facecolor': 'whitesmoke', 'alpha': 1.0, 'linewidth': 1.0, 'boxstyle': 'round,pad=0.5'}

        if dist_text_flag and t_ind >= t_ind_stable:
            dist_text = '\u0394x: ' + str(np.round(max(mode_x*xdim/nx),1)) + ' km'
            ax.text(0.04, 0.10, dist_text, transform=ax.transAxes, color = 'black', fontsize = 18, fontweight = 'semibold',
                verticalalignment='bottom', horizontalalignment='left', bbox=bbox, zorder = 5)

        if type_text_flag:
            ax.text(0.96, 0.96, type_text, transform=ax.transAxes, color = 'black',
                verticalalignment='top', horizontalalignment='right', bbox=bbox, zorder = 5)

        fig.tight_layout()
        save_fn = save_fn_path + save_fn_var+'_McSnow_2D_SIM'+'_' + datestr + '_at'+save_fn_time+'hrs_z'+str(nz)+'_x'+str(nx)+'_'+save_fn_lab+'_'+simulation+'.png'

        plt.savefig(save_fn, dpi = 300)
        if save_fn_var == 'Md':
            plt.show()
        plt.close()

        if save_Md_fig_output and save_fn_var == 'Md':
            np.save(dir_path + 'Data/' + save_fn_var+'_McSnow_2D_SIM'+'_' + datestr + '_at'+save_fn_time +
                    'hrs_z'+str(nz)+'_x'+str(nx)+'_grid_'+save_fn_lab+'_var_'+simulation+'.npy', var_fmtd)

            np.save(dir_path + 'Data/' + save_fn_var+'_McSnow_2D_SIM'+'_' + datestr + '_z'+str(nz)+'_x'+str(nx)+'_grid_'+save_fn_lab+'_dists.npy', dists_conv_2_nx)
            np.save(dir_path + 'Data/' + save_fn_var+'_McSnow_2D_SIM'+'_' + datestr + '_z'+str(nz)+'_x'+str(nx)+'_grid_'+save_fn_lab+'_elevs.npy', elevs_conv_2_ny)
            np.save(dir_path + 'Data/' + save_fn_var+'_McSnow_2D_SIM'+'_' + datestr + '_z'+str(nz)+'_x'+str(nx)+'_grid_'+save_fn_lab+'_yticks.npy', yticks)
            np.save(dir_path + 'Data/' + save_fn_var+'_McSnow_2D_SIM'+'_' + datestr + '_z'+str(nz)+'_x'+str(nx)+'_grid_'+save_fn_lab+'_ylabs.npy', ylabs)
        
        if save_Vt_Dm_Ar_Rf_fig_output and (save_fn_var == 'Vt' or save_fn_var == 'Ar' or save_fn_var == 'Dm' or save_fn_var == 'Rf' or save_fn_var == 'Vol'):
            np.save(dir_path + 'Data/' + save_fn_var+'_McSnow_2D_SIM'+'_' + datestr + '_at' +
                    save_fn_time+'hrs_z'+str(nz)+'_x'+str(nx)+'_grid_'+save_fn_lab+'_var_'+simulation+'.npy', var_fmtd)




#%%
"""
GENERAL PLOT SETUP
"""
t_hrs = (t_ind*10)/60
if np.floor(t_hrs) == t_hrs:
    t_hrs = np.int(t_hrs)
    t_hrs_sv = str(t_hrs)
else:
    t_hrs = np.round(t_hrs,1)
    t_hrs_sv = str(t_hrs).replace('.','_')

dx = xdim/nx
xticks = np.arange(0,nx+1,(nx/10))
xlabs = [int(x*dx) for x in xticks]

dists_conv_2_nx = elev_45deg[0,:] * nx / 200
elevs_conv_2_ny = elev_45deg[1,:] * ny / (zmax/1000)
x_NPOL = dist_offset * nx / 200
y_NPOL = 0.139 * ny / (zmax/1000)
rho_h2o = 1000
t_ind_endhr = [t_ind - 5, t_ind]
type_text = '' #unused option for test plotting of particle types 

#%%
"""
PLOT Md
"""
multipanel_flag = False
var_fn = Md_fn
cbar_lab = '$\mathregular{Mass\ Concentration\ (g\ m^{-3})}$'
cmap = 'viridis'
vmin, vmax = 0, 0.4
log_flag = False #only used for Nd
mode_flag = True
dist_text_flag = True
type_text_flag = False
var_scaling = 1000 # g/m^3 from kg/m^3
save_fn_var = 'Md'
save_fn_time = t_hrs_sv

generate_plot(multipanel_flag, var_fn, t_ind, cbar_lab, cmap, vmin, vmax, log_flag, dist_text_flag, type_text, type_text_flag,
    xticks, xlabs, nx, xdim, mode_flag, var_scaling, t_hrs, save_fn_var, save_fn_lab, save_fn_time, save_fn_path)

#%%
"""
PLOT HISTOGRAM OF Md
"""
if plot_histograms:
    if t_ind >= t_ind_stable:
        z_list, var, time_list = strip_ascii_data(var_fn[0], t_ind)
        nz=len(z_list)
        end_var_2D = var[t_ind,:,:]
        var_fmtd = end_var_2D * var_scaling
        var_fmtd[var_fmtd == 0] = np.nan

        mode_var = [np.where(var_fmtd[i,:] == np.nanmax(var_fmtd[i,:]))[0][0] for i in range(var_fmtd.shape[0])]

        data = []

        for zi in range(nz):
            z = z_list[zi]
            if h1 < z < h2:
                mode_var_z = mode_var[zi]
                data.append(var_fmtd[zi,mode_var_z])

        plt.rcParams.update({'font.size': 16})
        fig, ax = plt.subplots(figsize = (6.4, 4.8))

        hist1 = ax.hist(data, bins = 20, range = (0, 0.5), color = '#7570b3', edgecolor = '#7570b3', linewidth = 2.2, alpha = 0.7)

        ax.set_xlabel('$\mathregular{g\ m^{-3}}$')

        ax.grid(linestyle = '--', color = 'grey')
        ax.set_ylabel('Counts')

        fig.tight_layout()
        plt.show()
        plt.close()



#%%
"""
PLOT Nd
"""
multipanel_flag = False
var_fn = Nd_fn
cbar_lab = '$\mathregular{Number\ Concentration\ (m^{-3})}$'
cmap = 'cividis'
vmin, vmax = 200, 12000
log_flag = True
mode_flag = True
dist_text_flag = False
type_text_flag = False
var_scaling = 1 # n/m^3
save_fn_var = 'Nd'
save_fn_time = t_hrs_sv

generate_plot(multipanel_flag, var_fn, t_ind, cbar_lab, cmap, vmin, vmax, log_flag, dist_text_flag, type_text, type_text_flag,
    xticks, xlabs, nx, xdim, mode_flag, var_scaling, t_hrs, save_fn_var, save_fn_lab, save_fn_time, save_fn_path)


#%%
"""
PLOT HISTOGRAM OF Nd
"""
if plot_histograms:
    if t_ind >= t_ind_stable:
        z_list, var, time_list = strip_ascii_data(var_fn[0], t_ind)
        nz=len(z_list)
        end_var_2D = var[t_ind,:,:]
        var_fmtd = end_var_2D * var_scaling
        var_fmtd[var_fmtd == 0] = np.nan

        data = []
        for zi in range(nz):
            z = z_list[zi]
            if h1 < z < h2:
                mode_var_z = mode_var[zi]
                data.append(var_fmtd[zi,mode_var_z])

        plt.rcParams.update({'font.size': 16})
        fig, ax = plt.subplots(figsize = (6.4, 4.8))

        hist1 = ax.hist(data, bins=np.logspace(np.log10(200),np.log10(12000), 20), color = '#7570b3', edgecolor = '#7570b3', linewidth = 2.2, alpha = 0.7)
        plt.gca().set_xscale("log")

        ax.set_xlabel('$\mathregular{n\ m^{-3}}$')

        ax.grid(linestyle = '--', color = 'grey')
        ax.set_ylabel('Counts')


        fig.tight_layout()
        plt.show()
        plt.close()



#%%
"""
PLOT Ar
"""
multipanel_flag = False
var_fn = Ar_fn
cbar_lab = 'Area Ratio'
cmap = 'tab20b_r'
vmin, vmax = 0, 1.0
log_flag = False
mode_flag = True
dist_text_flag = False
type_text_flag = False
var_scaling = 1 # 1/1
save_fn_var = 'Ar'
save_fn_time = t_hrs_sv

generate_plot(multipanel_flag, var_fn, t_ind, cbar_lab, cmap, vmin, vmax, log_flag, dist_text_flag, type_text, type_text_flag,
    xticks, xlabs, nx, xdim, mode_flag, var_scaling, t_hrs, save_fn_var, save_fn_lab, save_fn_time, save_fn_path)

#%%
"""
PLOT Vt
"""
multipanel_flag = False
var_fn = Vt_fn
cbar_lab = '$\mathregular{Fall\ Velocity\ (m\ s^{-1})}$'
cmap = 'tab20b_r'
vmin, vmax = 0, 5
log_flag = False
mode_flag = True
dist_text_flag = False
type_text_flag = False
var_scaling = 1 # 1/1
save_fn_var = 'Vt'
save_fn_time = t_hrs_sv

generate_plot(multipanel_flag, var_fn, t_ind, cbar_lab, cmap, vmin, vmax, log_flag, dist_text_flag, type_text, type_text_flag,
    xticks, xlabs, nx, xdim, mode_flag, var_scaling, t_hrs, save_fn_var, save_fn_lab, save_fn_time, save_fn_path)

#%%
"""
PLOT Dm
"""
multipanel_flag = False
var_fn = Dm_fn
cbar_lab = '$\mathregular{Mean\ Diameter\ (mm)}$'
cmap = 'tab20b_r'
vmin, vmax = 0, 8
log_flag = False
mode_flag = True
dist_text_flag = False
type_text_flag = False
var_scaling = 1000 # mm from m
save_fn_var = 'Dm'
save_fn_time = t_hrs_sv

generate_plot(multipanel_flag, var_fn, t_ind, cbar_lab, cmap, vmin, vmax, log_flag, dist_text_flag, type_text, type_text_flag,
    xticks, xlabs, nx, xdim, mode_flag, var_scaling, t_hrs, save_fn_var, save_fn_lab, save_fn_time, save_fn_path)

#%%
"""
PLOT Rf
"""
multipanel_flag = False
var_fn = Rf_fn
cbar_lab = 'Riming Fraction'
cmap = 'tab20b_r'
vmin, vmax = 0, 1
log_flag = False
mode_flag = True
dist_text_flag = False
type_text_flag = False
var_scaling = 1 # 1/1
save_fn_var = 'Rf'
save_fn_time = t_hrs_sv

generate_plot(multipanel_flag, var_fn, t_ind, cbar_lab, cmap, vmin, vmax, log_flag, dist_text_flag, type_text, type_text_flag,
    xticks, xlabs, nx, xdim, mode_flag, var_scaling, t_hrs, save_fn_var, save_fn_lab, save_fn_time, save_fn_path)


#%%
"""PLOT Vol
"""
multipanel_flag = False
var_fn = Vol_fn
cbar_lab = '$\mathregular{Volume\ (cm^{3}\ m^{-3})}$'
cmap = 'tab20b_r'
vmin, vmax = 0, 0.05
log_flag = False
mode_flag = True
dist_text_flag = False
type_text_flag = False
var_scaling = 10**6 # cm3 from m3
save_fn_var = 'Vol'
save_fn_time = t_hrs_sv

generate_plot(multipanel_flag, var_fn, t_ind, cbar_lab, cmap, vmin, vmax, log_flag, dist_text_flag, type_text, type_text_flag,
    xticks, xlabs, nx, xdim, mode_flag, var_scaling, t_hrs, save_fn_var, save_fn_lab, save_fn_time, save_fn_path)


#%%
"""PLOT rho_eff
"""
cbar_lab = 'Particle Effective Density $\mathregular{(g\ cm^{-3})}$'
cmap = 'tab20b_r'
vmin, vmax = 0, 0.2
save_fn_time = t_hrs_sv

save_fn_var = 'Md'
var_Md = np.load(dir_path + 'Data/' + save_fn_var+'_McSnow_2D_SIM'+'_' + datestr +
                 '_at'+save_fn_time+'hrs_z'+str(nz)+'_x'+str(nx)+'_grid_'+save_fn_lab+'_var_'+simulation+'.npy')
save_fn_var = 'Vol'
var_Vol = np.load(dir_path + 'Data/' + save_fn_var+'_McSnow_2D_SIM'+'_' + datestr +
                  '_at'+save_fn_time+'hrs_z'+str(nz)+'_x'+str(nx)+'_grid_'+save_fn_lab+'_var_'+simulation+'.npy')

rhoeff = (var_Md / (var_Vol))

plt.rcParams.update({'font.size': 14})

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))

bbox_let = dict(facecolor='white', edgecolor='none', boxstyle='round')
if datestr == 'DEC3':
    dtxt = ' 03 Dec.'
    ht_temp0 = 2861.2
    ht_temp0_ytickpos = [ht_temp0]
    ht_temp0_yticks = [min(range(len(z_list)), key=lambda i: abs(z_list[i]-x))
                       for x in ht_temp0_ytickpos]
elif datestr == 'NOV13':
    dtxt = ' 13 Nov.'
    ht_temp0 = 2384.5
    ht_temp0_ytickpos = [ht_temp0]
    ht_temp0_yticks = [min(range(len(z_list)), key=lambda i: abs(z_list[i]-x))
                       for x in ht_temp0_ytickpos]

save_fn_var = 'Md'
save_fn_time = t_hrs_sv
t0_str = '4'

terrain_x = np.load(dir_path + 'Data/' + save_fn_var+'_McSnow_2D_SIM'+'_' + datestr +
                    '_z'+str(nz)+'_x'+str(nx)+'_grid_'+save_fn_lab+'_dists.npy')
terrain_y = np.load(dir_path + 'Data/' + save_fn_var+'_McSnow_2D_SIM'+'_' + datestr +
                    '_z'+str(nz)+'_x'+str(nx)+'_grid_'+save_fn_lab+'_elevs.npy')
yticks = np.load(dir_path + 'Data/' + save_fn_var+'_McSnow_2D_SIM'+'_' + datestr +
                 '_z'+str(nz)+'_x'+str(nx)+'_grid_'+save_fn_lab+'_yticks.npy')
ylabs = np.load(dir_path + 'Data/' + save_fn_var+'_McSnow_2D_SIM'+'_' + datestr +
                '_z'+str(nz)+'_x'+str(nx)+'_grid_'+save_fn_lab+'_ylabs.npy')
mode_x = np.load(dir_path + 'Data/' + save_fn_var+'_McSnow_2D_SIM'+'_' + datestr + '_at' +
                 save_fn_time+'hrs_z'+str(nz)+'_x'+str(nx)+'_grid_'+save_fn_lab+'_mode_x_'+simulation+'.npy')
mode_y = np.load(dir_path + 'Data/' + save_fn_var+'_McSnow_2D_SIM'+'_' + datestr + '_at' +
                 save_fn_time+'hrs_z'+str(nz)+'_x'+str(nx)+'_grid_'+save_fn_lab+'_mode_y_'+simulation+'.npy')


im1 = ax.pcolormesh(rhoeff, vmin=vmin, vmax=vmax, cmap=cmap)
cbar1 = fig.colorbar(im1, ax=ax, ticks=np.arange(
    vmin, vmax+0.04, 0.04), extend='max')
cbar1.set_label(cbar_lab)

ax.plot(terrain_x, terrain_y, color='gray', linewidth=2)
ax.fill_between(terrain_x, np.full(len(terrain_y), 0),
                y2=terrain_y, color='gray', alpha=0.75)
ax.set_xticks(xticks[::2])
ax.set_xticklabels(xlabs[::2])
ax.set_xlim(0, nx*0.6)  # 120 km
ax.set_yticks(yticks)
ax.set_yticklabels(ylabs)
ax.plot(mode_x, mode_y, color='lightgray', linewidth=3, linestyle='--')
ax.grid(linestyle='--', linewidth=0.5)
ax.tick_params(which='both', direction='in')
ax.axhline(ht_temp0_yticks, xmin=0, xmax=nx*0.6,
           color='dimgray', linewidth=3, linestyle=':')

ax.set_ylabel('Height (km)')
ax.set_xlabel('Distance (km)')

fig.tight_layout()
save_fn = save_fn_path + 'rhoeff'+'_McSnow_2D_SIM'+'_' + datestr + \
    '_t_step'+str(nz)+'_x'+str(nx)+'_grid_'+save_fn_lab+'_'+simulation+'.png'
plt.savefig(save_fn, dpi=300)
plt.show()
plt.close()

#%%
"""
PLOT Fm AT SURFACE
"""
var_fn = Fm_fn
lab = '$\mathregular{Mass\ Flux\ (g\ m^{-2}\ s^{-1})}$'
var_scaling = 1000 # g/m^2/s
save_fn_var = 'Fm'
save_fn_time = t_hrs_sv

z_list, var, time_list = strip_ascii_data(var_fn[0], t_ind)
nz=len(z_list)
ztickposind = [min(range(len(z_list)), key=lambda i: abs(z_list[i]-x)) for x in z_list]

end_var_2D = var[t_ind,:,:]
end_var_2D[end_var_2D == 0] = np.nan

var_fmtd = end_var_2D * var_scaling
RR = ((end_var_2D*60*60)/rho_h2o)*var_scaling

xlist = np.arange(0,nx)

plt.rcParams.update({'font.size': 16})
fig, ax = plt.subplots(figsize = (6.4, 4.8))
ax.bar(xlist, RR[0,:], color = '#1f78b4')

ax.set_xticks(xticks)
ax.set_xticklabels(xlabs)
ax.set_xlabel('Distance from Origin (km)')
ax.set_xlim(0,nx*0.6)


ax.set_yticks(np.arange(0,5.5,0.5))
ax.set_ylim(0,5)
ax.set_ylabel('$\mathregular{Rain\ Rate\ (mm\ hr^{-1})}$')

ax.grid(linestyle = '--')
fig.tight_layout()

save_fn = save_fn_path+'RR_surf_McSnow_2D_SIM_' + datestr + '_at'+save_fn_time + 'hrs_z'+str(nz)+'_x'+str(nx)+'_grid_'+save_fn_lab+'_'+simulation+'.png'

plt.savefig(save_fn, dpi = 300)
plt.show()
plt.close()


#%%
"""
PLOT Fm ACCUMULATION AT SURFACE (SUMMED OVER FINAL HOUR)
"""
if t_ind >= t_ind_stable:
    var_fn = Fm_fn
    lab = '$\mathregular{Mass\ Flux\ (g\ m^{-2}\ s^{-1})}$'
    var_scaling = 1000 # g/m^2/s
    save_fn_var = 'Fm'
    save_fn_time = t_hrs_sv

    z_list, var, time_list = strip_ascii_data(var_fn[0], t_ind)
    nz=len(z_list)
    ztickposind = [min(range(len(z_list)), key=lambda i: abs(z_list[i]-x)) for x in z_list]

    endhr_var_2D = var[t_ind_endhr[0]:t_ind_endhr[1]+1,:,:]
    endhr_var_2D[endhr_var_2D == 0] = np.nan

    RRhr = ((endhr_var_2D*10*60)/rho_h2o)*var_scaling #10 minute time step

    sum_RRhr = np.nansum(RRhr, axis = 0)

    xlist = np.arange(0,nx)
 
    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots(figsize = (6.4, 4.8))
    ax.bar(xlist, sum_RRhr[0,:], color = '#1f78b4')
    prcp_xy_bar = np.vstack((xlist, sum_RRhr[0,:]))

    xmin = min(np.where(sum_RRhr[0,:] > 0)[0])-1
    xmax = max(np.where(sum_RRhr[0,:] > 0)[0])+2

    xx, yy = xlist, sum_RRhr[0,:]

    mean = np.mean(xx[xmin:xmax])
    sigma = np.std(xx[xmin:xmax])
    maxyy = np.max(yy)

    xfine = np.linspace(min(xx), max(xx), 1000)
    initials = [1, mean, sigma, 0] # initial guess

    def asymGaussian(x, p):
        amp = (p[0] / (p[2] * np.sqrt(2 * np.pi)))
        spread = np.exp((-(x - p[1]) ** 2.0) / (2 * p[2] ** 2.0))
        skew = (1 + erf((p[3] * (x - p[1])) / (p[2] * np.sqrt(2))))
        return amp * spread * skew

    def residuals(p,y,x):
        return y - asymGaussian(x, p)

    # execute least-squares regression analysis to optimize initial parameters
    cnsts = leastsq(
        residuals,
        initials,
        args=(
            yy, # y value
            xx  # x value
            ))[0]

    yfine = asymGaussian(xfine, cnsts)
    ax.plot(xfine, yfine, c = '#e41a1c', linewidth = 2)

    prcp_xy_smooth = np.vstack((xfine, yfine))



    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabs)
    ax.set_xlabel('Distance from Origin (km)')
    ax.set_xlim(0,nx*0.6)

    ax.set_yticks(np.arange(0,6,1))
    ax.set_ylim(0,5)
    ax.set_ylabel('$\mathregular{Final\ Hour\ Rainfall\ (mm)}$')

    ax.grid(linestyle = '--')
    fig.tight_layout()


    save_fn = save_fn_path+'RR_surf_1hr_McSnow_2D_SIM_' + datestr + '_at'+save_fn_time + \
        'hrs_z'+str(nz)+'_x'+str(nx)+'_grid_'+save_fn_lab+'_'+simulation+'.png'

    plt.savefig(save_fn, dpi = 300)

    plt.show()
    plt.close()

    np.save(dir_path+'Data/'+'RR_surf_1hr_BAR_McSnow_2D_SIM'+'_' + datestr + '_at'+save_fn_time +
            'hrs_z'+str(nz)+'_x'+str(nx)+'_grid_'+save_fn_lab+'_'+simulation+'.npy', prcp_xy_bar)
    np.save(dir_path+'Data/'+'RR_surf_1hr_SMOOTH_McSnow_2D_SIM'+'_' + datestr + '_at'+save_fn_time +
            'hrs_z'+str(nz)+'_x'+str(nx)+'_grid_'+save_fn_lab+'_'+simulation+'.npy', prcp_xy_smooth)



#%%
if plot_Md_timesteps:
    plt.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots(nrows = 1, ncols = 3, figsize = (8, 3.))
    if datestr == 'DEC3':
        dtxt = ''
    elif datestr  == 'NOV13':
        dtxt = ''

    bbox_let=dict(facecolor='white', edgecolor='none', boxstyle='round')
    ax[0].text(0.95, 0.95, '(a)'+dtxt, horizontalalignment='right', verticalalignment='top', transform=ax[0].transAxes, bbox = bbox_let, fontweight = 'semibold')
    ax[1].text(0.95, 0.95, '(b)'+dtxt, horizontalalignment='right', verticalalignment='top', transform=ax[1].transAxes, bbox = bbox_let, fontweight = 'semibold')
    ax[2].text(0.95, 0.95, '(c)'+dtxt, horizontalalignment='right', verticalalignment='top', transform=ax[2].transAxes, bbox = bbox_let, fontweight = 'semibold')

    var_fn = Md_fn
    cbar_lab = '$\mathregular{Mass\ Conc.\ (g\ m^{-3})}$'
    cmap = 'viridis'
    vmin, vmax = 0, 0.4
    var_scaling = 1000 # g/m^3 from kg/m^3
    save_fn_var = 'Md'
    save_fn_time = t_hrs_sv

    t0_str = '0_2'
    t1_str = '0_3'
    t2_str = '1'

    terrain_x = np.load(dir_path + 'Data/' + save_fn_var+'_McSnow_2D_SIM'+'_' + datestr + '_z'+str(nz)+'_x'+str(nx)+'_grid_'+save_fn_lab+'_dists.npy')
    terrain_y = np.load(dir_path + 'Data/' + save_fn_var+'_McSnow_2D_SIM'+'_' + datestr + '_z'+str(nz)+'_x'+str(nx)+'_grid_'+save_fn_lab+'_elevs.npy')
    yticks = np.load(dir_path + 'Data/' + save_fn_var+'_McSnow_2D_SIM'+'_' + datestr + '_z'+str(nz)+'_x'+str(nx)+'_grid_'+save_fn_lab+'_yticks.npy')
    ylabs = np.load(dir_path + 'Data/' + save_fn_var+'_McSnow_2D_SIM'+'_' + datestr + '_z'+str(nz)+'_x'+str(nx)+'_grid_'+save_fn_lab+'_ylabs.npy')
    mode_x = np.load(dir_path + 'Data/' + save_fn_var+'_McSnow_2D_SIM'+'_' + datestr + '_at' +
                     save_fn_time+'hrs_z'+str(nz)+'_x'+str(nx)+'_grid_'+save_fn_lab+'_mode_x_'+simulation+'.npy')
    mode_y = np.load(dir_path + 'Data/' + save_fn_var+'_McSnow_2D_SIM'+'_' + datestr + '_at' +
                     save_fn_time+'hrs_z'+str(nz)+'_x'+str(nx)+'_grid_'+save_fn_lab+'_mode_y_'+simulation+'.npy')


    var = np.load(dir_path + 'Data/' + save_fn_var+'_McSnow_2D_SIM'+'_' + datestr + '_at'+t0_str+'hrs_z'+str(nz)+'_x'+str(nx)+'_grid_'+save_fn_lab+'_var_'+simulation+'.npy')
    im0 = ax[0].pcolormesh(var, vmin = vmin, vmax = vmax, cmap = cmap)

    var = np.load(dir_path + 'Data/' + save_fn_var+'_McSnow_2D_SIM'+'_' + datestr + '_at'+t1_str+'hrs_z'+str(nz)+'_x'+str(nx)+'_grid_'+save_fn_lab+'_var_'+simulation+'.npy')
    im1 = ax[1].pcolormesh(var, vmin = vmin, vmax = vmax, cmap = cmap)

    var = np.load(dir_path + 'Data/' + save_fn_var+'_McSnow_2D_SIM'+'_' + datestr + '_at'+t2_str+'hrs_z'+str(nz)+'_x'+str(nx)+'_grid_'+save_fn_lab+'_var_'+simulation+'.npy')
    im2 = ax[2].pcolormesh(var, vmin = vmin, vmax = vmax, cmap = cmap)
    ax[2].plot(mode_x, mode_y, color = 'lightgray', linewidth = 3, linestyle = '--')

    fig.subplots_adjust(bottom = 0.2, top = 0.95, left = 0.1, right=0.99, wspace = 0.15)
    cbar = fig.colorbar(im2, ax=ax.ravel().tolist(), ticks = np.arange(vmin, vmax+0.1, 0.1))
    cbar.set_label(cbar_lab)


    z6km_i = np.where(np.array(ylabs) == 6)[0][0]
    z6km_lev = yticks[z6km_i]



    for axi in ax:
        axi.plot(terrain_x, terrain_y, color = 'gray', linewidth = 2)
        axi.fill_between(terrain_x, np.full(len(terrain_y), 0), y2 = terrain_y, color = 'gray', alpha = 0.75)
        axi.set_xticks(xticks[::2])
        axi.set_xticklabels(xlabs[::2])
        axi.set_xlabel('Distance (km)')
        axi.set_xlim(0,nx*0.4) # 80 km
        axi.set_yticks(yticks)
        axi.set_yticklabels(ylabs)
        axi.grid(linestyle = '--')
        axi.tick_params(which = 'both', direction = 'in')


    ax[0].set_ylabel('Height (km)')

    ax[1].tick_params(labelleft=False)
    ax[2].tick_params(labelleft=False)
    ax[1].yaxis.label.set_visible(False)
    ax[2].yaxis.label.set_visible(False)

    save_fn = save_fn_path + save_fn_var+'_McSnow_2D_SIM'+'_' + datestr + '_t_step_10_20_60_min_z'+str(nz)+'_x'+str(nx)+'_grid_'+save_fn_lab+'_'+simulation+'.png'
    plt.savefig(save_fn, dpi = 300)
    plt.show()
    plt.close()




#%%
if plot_Vt_Dm_rho_Rf:
    plt.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots(nrows = 2, ncols = 2, figsize = (8, 6))

    bbox_let=dict(facecolor='white', edgecolor='none', boxstyle='round')
    if datestr == 'DEC3':
        dtxt = ' 03 Dec.'
        ht_temp0 = 2861.2 #meters; sounding derived 
        ht_temp0_ytickpos = [ht_temp0]
        ht_temp0_yticks = [min(range(len(z_list)), key=lambda i: abs(z_list[i]-x))
                  for x in ht_temp0_ytickpos]
    elif datestr  == 'NOV13':
        dtxt = ' 13 Nov.'
        ht_temp0 = 2384.5 #meters; sounding derived 
        ht_temp0_ytickpos = [ht_temp0]
        ht_temp0_yticks = [min(range(len(z_list)), key=lambda i: abs(z_list[i]-x))
                  for x in ht_temp0_ytickpos]
    ax[0,0].text(0.95, 0.95, '(a)'+dtxt, horizontalalignment='right', verticalalignment='top', transform=ax[0,0].transAxes, bbox = bbox_let, fontweight = 'semibold')
    ax[0,1].text(0.95, 0.95, '(b)'+dtxt, horizontalalignment='right', verticalalignment='top', transform=ax[0,1].transAxes, bbox = bbox_let, fontweight = 'semibold')
    ax[1,0].text(0.95, 0.95, '(c)'+dtxt, horizontalalignment='right', verticalalignment='top', transform=ax[1,0].transAxes, bbox = bbox_let, fontweight = 'semibold')
    ax[1,1].text(0.95, 0.95, '(d)'+dtxt, horizontalalignment='right', verticalalignment='top', transform=ax[1,1].transAxes, bbox = bbox_let, fontweight = 'semibold')

    save_fn_var = 'Md'
    save_fn_time = t_hrs_sv
    t0_str = '4'

    terrain_x = np.load(dir_path + 'Data/' + save_fn_var+'_McSnow_2D_SIM'+'_' + datestr + '_z'+str(nz)+'_x'+str(nx)+'_grid_'+save_fn_lab+'_dists.npy')
    terrain_y = np.load(dir_path + 'Data/' + save_fn_var+'_McSnow_2D_SIM'+'_' + datestr + '_z'+str(nz)+'_x'+str(nx)+'_grid_'+save_fn_lab+'_elevs.npy')
    yticks = np.load(dir_path + 'Data/' + save_fn_var+'_McSnow_2D_SIM'+'_' + datestr + '_z'+str(nz)+'_x'+str(nx)+'_grid_'+save_fn_lab+'_yticks.npy')
    ylabs = np.load(dir_path + 'Data/' + save_fn_var+'_McSnow_2D_SIM'+'_' + datestr + '_z'+str(nz)+'_x'+str(nx)+'_grid_'+save_fn_lab+'_ylabs.npy')
    mode_x = np.load(dir_path + 'Data/' + save_fn_var+'_McSnow_2D_SIM'+'_' + datestr + '_at'+save_fn_time+'hrs_z'+str(nz)+'_x'+str(nx)+'_grid_'+save_fn_lab+'_mode_x_'+simulation+'.npy')
    mode_y = np.load(dir_path + 'Data/' + save_fn_var+'_McSnow_2D_SIM'+'_' + datestr + '_at' +
                     save_fn_time+'hrs_z'+str(nz)+'_x'+str(nx)+'_grid_'+save_fn_lab+'_mode_y_'+simulation+'.npy')



    """
    PLOT A, Dm
    """
    var_fn = Dm_fn
    cbar_lab = '$\mathregular{Mean\ Diameter\ (mm)}$'
    cmap = 'tab20b_r'
    vmin, vmax = 0, 8
    save_fn_var = 'Dm'
    save_fn_time = t_hrs_sv

    var = np.load(dir_path + 'Data/' + save_fn_var+'_McSnow_2D_SIM'+'_' + datestr + '_at' +
                  save_fn_time+'hrs_z'+str(nz)+'_x'+str(nx)+'_grid_'+save_fn_lab+'_var_'+simulation+'.npy')
    im1 = ax[0,0].pcolormesh(var, vmin = vmin, vmax = vmax, cmap = cmap)
    cbar1 = fig.colorbar(im1, ax = ax[0,0], ticks = np.arange(vmin, vmax+1, 1))
    cbar1.set_label(cbar_lab)


    """
    PLOT B, Vt
    """
    var_fn = Vt_fn
    cbar_lab = '$\mathregular{Fall\ Velocity\ (m\ s^{-1})}$'
    cmap = 'tab20b_r'
    vmin, vmax = 0, 5
    save_fn_var = 'Vt'
    save_fn_time = t_hrs_sv

    var = np.load(dir_path + 'Data/' + save_fn_var+'_McSnow_2D_SIM'+'_' + datestr + '_at' +
                  save_fn_time+'hrs_z'+str(nz)+'_x'+str(nx)+'_grid_'+save_fn_lab+'_var_'+simulation+'.npy')
    im0 = ax[0,1].pcolormesh(var, vmin = vmin, vmax = vmax, cmap = cmap)
    cbar0 = fig.colorbar(im0, ax = ax[0,1], ticks = np.arange(vmin, vmax+1, 1))
    cbar0.set_label(cbar_lab)


    """
    PLOT C, rho
    """
    
    var_fn = Ar_fn
    cbar_lab = 'Effective Density $\mathregular{(g\ cm^{-3})}$'
    cmap = 'tab20b_r'
    vmin, vmax = 0, 0.2
    save_fn_time = t_hrs_sv

    save_fn_var = 'Md'
    var_Md = np.load(dir_path + 'Data/' + save_fn_var+'_McSnow_2D_SIM'+'_' + datestr + '_at' +
                     save_fn_time+'hrs_z'+str(nz)+'_x'+str(nx)+'_grid_'+save_fn_lab+'_var_'+simulation+'.npy')
    save_fn_var = 'Vol'
    var_Vol = np.load(dir_path + 'Data/' + save_fn_var+'_McSnow_2D_SIM'+'_' + datestr +
                      '_at'+save_fn_time+'hrs_z'+str(nz)+'_x'+str(nx)+'_grid_'+save_fn_lab+'_var_'+simulation+'.npy')
    var = (var_Md / (var_Vol))

    im2 = ax[1,0].pcolormesh(var, vmin = vmin, vmax = vmax, cmap = cmap)
    cbar2 = fig.colorbar(im2, ax = ax[1,0], ticks = np.arange(vmin, vmax+0.04, 0.04), extend = 'max')
    cbar2.set_label(cbar_lab)


    """
    PLOT D, Rf
    """
    var_fn = Rf_fn
    cbar_lab = 'Rime Fraction'
    cmap = 'tab20b_r'
    vmin, vmax = 0, 1
    save_fn_var = 'Rf'
    save_fn_time = t_hrs_sv

    var = np.load(dir_path + 'Data/' + save_fn_var+'_McSnow_2D_SIM'+'_' + datestr + '_at' +
                  save_fn_time+'hrs_z'+str(nz)+'_x'+str(nx)+'_grid_'+save_fn_lab+'_var_'+simulation+'.npy')
    im3 = ax[1,1].pcolormesh(var, vmin = vmin, vmax = vmax, cmap = cmap)
    cbar3 = fig.colorbar(im3, ax = ax[1,1], ticks = np.arange(vmin, vmax+0.1, 0.2))
    cbar3.set_label(cbar_lab)


    z6km_i = np.where(np.array(ylabs) == 6)[0][0]
    z6km_lev = yticks[z6km_i]



    for r in np.arange(2):
        for c in np.arange(2):
            ax[r,c].plot(terrain_x, terrain_y, color = 'gray', linewidth = 2)
            ax[r,c].fill_between(terrain_x, np.full(len(terrain_y), 0), y2 = terrain_y, color = 'gray', alpha = 0.75)
            ax[r,c].set_xticks(xticks[::2])
            ax[r,c].set_xticklabels(xlabs[::2])
            ax[r,c].set_xlim(0,nx*0.6) # 120 km
            ax[r,c].set_yticks(yticks)
            ax[r,c].set_yticklabels(ylabs)
            ax[r,c].plot(mode_x, mode_y, color = 'lightgray', linewidth = 3, linestyle = '--')
            ax[r,c].grid(linestyle = '--', linewidth = 0.5)
            ax[r,c].tick_params(which = 'both', direction = 'in')
            ax[r,c].axhline(ht_temp0_yticks, xmin = 0, xmax = nx*0.6, color = 'dimgray', linewidth = 3, linestyle = ':')

    ax[0,0].set_ylabel('Height (km)')
    ax[1,0].set_ylabel('Height (km)')
    ax[0,1].tick_params(labelleft=False)
    ax[1,1].tick_params(labelleft=False)

    ax[1,0].set_xlabel('Distance (km)')
    ax[1,1].set_xlabel('Distance (km)')
    ax[0,1].tick_params(labelbottom=False)
    ax[0,0].tick_params(labelbottom=False)


    fig.tight_layout()
    plt.subplots_adjust(wspace = 0.2, hspace = 0.15)

    save_fn = save_fn_path + 'Vt_Dm_rho_Rf'+'_McSnow_2D_SIM'+'_' + datestr + '_t_step'+str(nz)+'_x'+str(nx)+'_grid_'+save_fn_lab+'_'+simulation+'.png'
    plt.savefig(save_fn, dpi = 300)
    plt.show()
    plt.close()


#%%
if plot_Md_plus_lines:

    plt.rcParams.update({'font.size': 14})
    fig = plt.figure(figsize = (8, 4))

    spec = gridspec.GridSpec(nrows = 2, ncols = 2, width_ratios = [1,1], height_ratios = [0.05, 1], \
                            wspace = 0.25, hspace = 0.05, \
                            left=0.08, bottom=0.15, right=0.95, top=0.8)

    spec1 = gridspec.GridSpecFromSubplotSpec(nrows = 1, ncols = 2, subplot_spec=spec[1,1], width_ratios = [1,1], wspace = 0.25)


    ax0 = fig.add_subplot(spec[1,0])
    axcb = fig.add_subplot(spec[0,0])
    ax1 = fig.add_subplot(spec1[0])
    ax3 = fig.add_subplot(spec1[1])

    ax2 = ax1.twiny()
    ax4 = ax3.twiny()

    if datestr == 'DEC3':
        dtxt = ' 03 Dec.'
    elif datestr  == 'NOV13':
        dtxt = ' 13 Nov.'

    bbox_let=dict(facecolor='white', edgecolor='none', boxstyle='round')

    ax0.text(0.95, 0.95, '(a)'+dtxt, horizontalalignment='right', verticalalignment='top', transform=ax0.transAxes, bbox = bbox_let, fontweight = 'semibold')
    ax1.text(0.95, 0.95, '(b)', horizontalalignment='right', verticalalignment='top', transform=ax1.transAxes, bbox = bbox_let, fontweight = 'semibold')
    ax3.text(0.95, 0.95, '(c)', horizontalalignment='right', verticalalignment='top', transform=ax3.transAxes, bbox = bbox_let, fontweight = 'semibold')

    var_fn = Md_fn
    cbar_lab = '$\mathregular{Mass\ Concentration\ (g\ m^{-3})}$'
    cmap = 'viridis'
    vmin, vmax = 0, 0.4
    save_fn_var = 'Md'
    save_fn_time = t_hrs_sv
    t0_str = t_hrs_sv

    terrain_x = np.load(dir_path + 'Data/' + save_fn_var+'_McSnow_2D_SIM'+'_' + datestr + '_z'+str(nz)+'_x'+str(nx)+'_grid_'+save_fn_lab+'_dists.npy')
    terrain_y = np.load(dir_path + 'Data/' + save_fn_var+'_McSnow_2D_SIM'+'_' + datestr + '_z'+str(nz)+'_x'+str(nx)+'_grid_'+save_fn_lab+'_elevs.npy')
    yticks = np.load(dir_path + 'Data/' + save_fn_var+'_McSnow_2D_SIM'+'_' + datestr + '_z'+str(nz)+'_x'+str(nx)+'_grid_'+save_fn_lab+'_yticks.npy')
    ylabs = np.load(dir_path + 'Data/' + save_fn_var+'_McSnow_2D_SIM'+'_' + datestr + '_z'+str(nz)+'_x'+str(nx)+'_grid_'+save_fn_lab+'_ylabs.npy')
    mode_x = np.load(dir_path + 'Data/' + save_fn_var+'_McSnow_2D_SIM'+'_' + datestr + '_at'+save_fn_time+'hrs_z'+str(nz)+'_x'+str(nx)+'_grid_'+save_fn_lab+'_mode_x_'+simulation+'.npy')
    mode_y = np.load(dir_path + 'Data/' + save_fn_var+'_McSnow_2D_SIM'+'_' + datestr + '_at' +
                     save_fn_time+'hrs_z'+str(nz)+'_x'+str(nx)+'_grid_'+save_fn_lab+'_mode_y_'+simulation+'.npy')

    var = np.load(dir_path + 'Data/' + save_fn_var+'_McSnow_2D_SIM'+'_' + datestr + '_at'+ \
                  t0_str+'hrs_z'+str(nz)+'_x'+str(nx)+'_grid_'+save_fn_lab+'_var_'+simulation+'.npy')
    im0 = ax0.pcolormesh(var, vmin=vmin, vmax=vmax, cmap=cmap)

    z6km_i = np.where(np.array(ylabs) == 6)[0][0]
    z6km_lev = yticks[z6km_i]

    ax0.plot(terrain_x, terrain_y, color = 'gray', linewidth = 2)
    ax0.fill_between(terrain_x, np.full(len(terrain_y), 0), y2 = terrain_y, color = 'gray', alpha = 0.75)
    ax0.set_xticks(xticks[::2])
    ax0.set_xticklabels(xlabs[::2])
    ax0.set_xlim(0,nx*0.6) # 120 km
    ax0.set_yticks(yticks)
    ax0.set_yticklabels(ylabs)
    ax0.plot(mode_x, mode_y, color = 'lightgray', linewidth = 2, linestyle = '--')
    ax0.grid(linestyle = '--', linewidth = 0.5)

    ax0.set_ylabel('Height (km)')
    ax0.set_xlabel('Distance (km)')

    cbar0 = fig.colorbar(im0, cax = axcb, ticks = np.arange(vmin, vmax+0.1, 0.1), orientation = 'horizontal')
    cbar0.set_label(cbar_lab)
    cbar0.ax.xaxis.set_ticks_position('top')
    cbar0.ax.xaxis.set_label_position('top')
    cbar0.ax.xaxis.set_ticks([0, 0.1, 0.2, 0.3, 0.4])
    cbar0.ax.xaxis.set_ticklabels(['0', '0.1', '0.2', '0.3', '0.4'])

    """
    PLOT Vt
    """
    var_fn = Vt_fn
    cbar_lab = '$\mathregular{Velocity\ (m\ s^{-1})}$'
    vmin, vmax = 0, 6
    save_fn_var = 'Vt'
    save_fn_time = t_hrs_sv
    color = blackcolor
    ls = '-'

    model_md_ht_50_15_85 = np.load(dir_path + 'Data/' + save_fn_var+'_McSnow_2D_SIM'+'_' + datestr + '_at'+save_fn_time +
                        'hrs_z'+str(nz)+'_x'+str(nx)+'_grid_'+save_fn_lab+'_traj_50_15_85_'+simulation+'.npy')

    xsmoothed = gaussian_filter1d(model_md_ht_50_15_85[1,:], sigma=2)
    ax1.plot(xsmoothed, model_md_ht_50_15_85[0,:]/1, color = color, linewidth = 2, linestyle = ls)
    ax1.set_xlabel(cbar_lab)
    ax1.xaxis.label.set_color(color)
    ax1.tick_params(axis='x', colors=color)


    """
    PLOT Dm
    """
    var_fn = Dm_fn
    cbar_lab = '$\mathregular{Diameter\ (mm)}$'
    cmap = 'tab20b_r'
    save_fn_var = 'Dm'
    save_fn_time = t_hrs_sv
    colorline = greencolor
    color = '#178564'
    ls = '-'

    model_md_ht_50_15_85 = np.load(dir_path + 'Data/' + save_fn_var+'_McSnow_2D_SIM'+'_' + datestr + '_at'+save_fn_time +
                        'hrs_z'+str(nz)+'_x'+str(nx)+'_grid_'+save_fn_lab+'_traj_50_15_85_'+simulation+'.npy')

    xsmoothed = gaussian_filter1d(model_md_ht_50_15_85[1,:], sigma=2)
    ax2.plot(xsmoothed, model_md_ht_50_15_85[0,:]/1, color = colorline, linewidth = 2, linestyle = ls)
    ax2.xaxis.label.set_color(color)
    ax2.tick_params(axis='x', colors=color)
    ax2.set_xlabel(cbar_lab)

    ax1.set_xlim(vmin,vmax)
    ax2.set_xlim(vmin,vmax)

    ax1.set_xticks(np.arange(vmin, vmax+1, 2))
    ax2.set_xticks(np.arange(vmin, vmax+1, 2))

    ax1.set_ylim(0,6000)
    ax2.set_ylim(0,6000)
    ax1.set_yticks(np.arange(0,7000, 1000))
    ax1.set_yticklabels(np.arange(0,7, 1))
    ax2.set_yticks(np.arange(0,7000, 1000))
    ax1.grid(linestyle = '--', linewidth = 0.5)
    ax1.set_ylabel('Height (km)')

    """
    PLOT C, rho
    """
    var_fn = Ar_fn
    cbar_lab = 'Density $\mathregular{(g\ cm^{-3})}$'
    save_fn_var = 'Md'
    save_fn_time = t_hrs_sv

    colorline = bluecolor
    color = '#1b699d'
    ls = '-'

    save_fn_var = 'Md'
    model_md_ht_50_15_85_Md = np.load(dir_path + 'Data/' + save_fn_var+'_McSnow_2D_SIM'+'_' + datestr + '_at'+save_fn_time +
                                      'hrs_z'+str(nz)+'_x'+str(nx)+'_grid_'+save_fn_lab+'_traj_50_15_85_'+simulation+'.npy')
    save_fn_var = 'Vol'
    model_md_ht_50_15_85_Vol = np.load(dir_path + 'Data/' + save_fn_var+'_McSnow_2D_SIM'+'_' + datestr + '_at'+save_fn_time +
                                   'hrs_z'+str(nz)+'_x'+str(nx)+'_grid_'+save_fn_lab+'_traj_50_15_85_'+simulation+'.npy')
    rho_eff = model_md_ht_50_15_85_Md[1,:] / (model_md_ht_50_15_85_Vol[1,:])
    
    xsmoothed = gaussian_filter1d(rho_eff, sigma=2)
    ax3.plot(xsmoothed, model_md_ht_50_15_85_Md[0,:]/1, color = colorline, linewidth = 2, linestyle = ls)
    ax3.xaxis.label.set_color(color)
    ax3.tick_params(axis='x', colors=color)
    ax3.set_xlabel(cbar_lab)


    """
    PLOT D, Rf
    """
    var_fn = Rf_fn
    cbar_lab = 'Rime Fraction'
    vmin, vmax = 0, 1
    save_fn_var = 'Rf'
    save_fn_time = t_hrs_sv
    colorline = redcolor
    color = '#c4422d'
    ls = '-'

    model_md_ht_50_15_85 = np.load(dir_path + 'Data/' + save_fn_var+'_McSnow_2D_SIM'+'_' + datestr + '_at'+save_fn_time +
                                   'hrs_z'+str(nz)+'_x'+str(nx)+'_grid_'+save_fn_lab+'_traj_50_15_85_'+simulation+'.npy')

    ax4.plot(model_md_ht_50_15_85[1,:], model_md_ht_50_15_85[0,:]/1, color = colorline, linewidth = 2, linestyle = ls)
    ax4.xaxis.label.set_color(color)
    ax4.tick_params(axis='x', colors=color)
    ax4.set_xlabel(cbar_lab)

    ax3.set_xlim(vmin,0.2)
    ax4.set_xlim(vmin,vmax)

    ax3.set_xticks([0,0.05,0.1,0.15, 0.2])
    ax3.set_xticklabels(['0', '', '0.1', '', '0.2'])
    ax4.set_xticks([0,0.25,0.5,0.75, 1])
    ax4.set_xticklabels(['0', '', '0.5', '', '1'])



    ax3.set_ylim(0,6000)
    ax4.set_ylim(0,6000)
    ax3.set_yticks(np.arange(0,7000, 1000))
    ax4.set_yticks(np.arange(0,7000, 1000))
    ax3.grid(linestyle = '--', linewidth = 0.5)



    ax0.tick_params(which = 'both', direction = 'in')
    ax1.tick_params(which = 'both', direction = 'in')
    ax2.tick_params(which = 'both', direction = 'in')
    ax3.tick_params(which = 'both', direction = 'in')
    ax4.tick_params(which = 'both', direction = 'in')

    ax2.tick_params(labelleft=False)
    ax3.tick_params(labelleft=False)
    ax4.tick_params(labelleft=False)

    save_fn = save_fn_path + 'Md_plus_var_lines'+'_McSnow_2D_SIM'+'_' + datestr + '_z'+str(nz)+'_x'+str(nx)+'_grid_'+save_fn_lab+'_'+simulation+'.png'
    plt.savefig(save_fn, dpi = 300, facecolor='white', transparent=False)
    plt.show()
    plt.close()



#%%
if plot_LWC_sensitivity:
    plt.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots(nrows = 2, ncols = 4, figsize = (8, 6))

    bbox_let=dict(facecolor='white', edgecolor='none', boxstyle='round')
    ax[0,0].text(0.95, 0.95, '(a)', horizontalalignment='right', verticalalignment='top', transform=ax[0,0].transAxes, bbox = bbox_let, fontweight = 'semibold')
    ax[0,1].text(0.95, 0.95, '(b)', horizontalalignment='right', verticalalignment='top', transform=ax[0,1].transAxes, bbox = bbox_let, fontweight = 'semibold')
    ax[0,2].text(0.95, 0.95, '(c)', horizontalalignment='right', verticalalignment='top', transform=ax[0,2].transAxes, bbox = bbox_let, fontweight = 'semibold')
    ax[0,3].text(0.95, 0.95, '(d)', horizontalalignment='right', verticalalignment='top', transform=ax[0,3].transAxes, bbox = bbox_let, fontweight = 'semibold')
    ax[0,3].text(1.05, 0.5, '03 December', horizontalalignment='left', verticalalignment='center', transform=ax[0,3].transAxes, bbox = bbox_let, fontweight = 'semibold', rotation = 90)


    ax[1,0].text(0.95, 0.95, '(e)', horizontalalignment='right', verticalalignment='top', transform=ax[1,0].transAxes, bbox = bbox_let, fontweight = 'semibold')
    ax[1,1].text(0.95, 0.95, '(f)', horizontalalignment='right', verticalalignment='top', transform=ax[1,1].transAxes, bbox = bbox_let, fontweight = 'semibold')
    ax[1,2].text(0.95, 0.95, '(g)', horizontalalignment='right', verticalalignment='top', transform=ax[1,2].transAxes, bbox = bbox_let, fontweight = 'semibold')
    ax[1,3].text(0.95, 0.95, '(h)', horizontalalignment='right', verticalalignment='top', transform=ax[1,3].transAxes, bbox = bbox_let, fontweight = 'semibold')
    ax[1,3].text(1.05, 0.5, '13 November', horizontalalignment='left', verticalalignment='center', transform=ax[1,3].transAxes, bbox = bbox_let, fontweight = 'semibold', rotation = 90)

    datelist = ['DEC3SIMULATIONS', 'NOV13SIMULATIONS']
    datelist_short = ['DEC3', 'NOV13']
    simlist = ['rime_off', 'rime_light', 'rime_heavy', 'CONTROL']
    colors = [redcolor, bluecolor, greencolor, blackcolor]
    leg_colors = [blackcolor, greencolor, bluecolor, redcolor]
    lines = ['-', '--', ':']
    leg_lines = [Line2D([0], [0], color=c, linewidth=3, linestyle='-') for c in leg_colors]
    leg_labels = ['control', 'SLW_heavy', 'SLW_light', 'SLW_none']


    """
    PLOT A, Rf
    """
    var_fn = Rf_fn
    cbar_lab = 'Rime Fraction'
    vmin, vmax = 0, 1
    save_fn_var = 'Rf'
    save_fn_time = t_hrs_sv

    for d in range(len(datelist)):
        dstr = datelist[d]
        dstr_short = datelist_short[d]
        for s in range(len(simlist)):
            sstr = simlist[s]
            model_md_ht_50_15_85 = np.load(dir_path + 'Data/' + save_fn_var+'_McSnow_2D_SIM'+'_' + dstr_short + '_at'+save_fn_time +
                        'hrs_z'+str(nz)+'_x'+str(nx)+'_grid_'+save_fn_lab+'_traj_50_15_85_'+sstr+'.npy')

            ax[d, 0].plot(model_md_ht_50_15_85[1, :], model_md_ht_50_15_85[0, :]/1, color=colors[s], linewidth=2, linestyle=lines[0], path_effects=[pe.withStroke(linewidth=3.5, foreground='white'), pe.Normal()])

        if d == 1:
            ax[d,0].set_xlabel(cbar_lab)
        ax[d,0].set_xlim(vmin, vmax)
        ax[d,0].set_xticks([0,0.25,0.5,0.75, 1])
        ax[d,0].set_xticklabels(['0', '', '0.5', '', '1'])

    """
    PLOT B, Vt
    """
    var_fn = Vt_fn
    cbar_lab = 'Fall Velocity\n $\mathregular{(m\ s^{-1})}$'
    vmin, vmax = 0, 6
    save_fn_var = 'Vt'
    save_fn_time = t_hrs_sv

    for d in range(len(datelist)):
        dstr = datelist[d]
        dstr_short = datelist_short[d]
        for s in range(len(simlist)):
            sstr = simlist[s]
            model_md_ht_50_15_85 = np.load(dir_path + 'Data/' + save_fn_var+'_McSnow_2D_SIM'+'_' + dstr_short + '_at'+save_fn_time +
                        'hrs_z'+str(nz)+'_x'+str(nx)+'_grid_'+save_fn_lab+'_traj_50_15_85_'+sstr+'.npy')
            xsmoothed = gaussian_filter1d(model_md_ht_50_15_85[1,:], sigma=2)
            ax[d,1].plot(xsmoothed, model_md_ht_50_15_85[0,:]/1, color = colors[s], linewidth = 2, linestyle = lines[0], path_effects=[pe.withStroke(linewidth=3.5, foreground='white'), pe.Normal()])

        if d == 1:
            ax[d,1].set_xlabel(cbar_lab)
        ax[d,1].set_xlim(vmin, vmax)
        ax[d,1].set_xticks(np.arange(vmin, vmax+2,2))


    """
    PLOT C, Dm
    """
    var_fn = Dm_fn
    cbar_lab = 'Mean Diameter\n $\mathregular{(mm)}$'
    vmin, vmax = 0, 4
    save_fn_var = 'Dm'
    save_fn_time = t_hrs_sv

    for d in range(len(datelist)):
        dstr = datelist[d]
        dstr_short = datelist_short[d]
        for s in range(len(simlist)):
            sstr = simlist[s]
            model_md_ht_50_15_85 = np.load(dir_path + 'Data/' + save_fn_var+'_McSnow_2D_SIM'+'_' + dstr_short + '_at'+save_fn_time +
                        'hrs_z'+str(nz)+'_x'+str(nx)+'_grid_'+save_fn_lab+'_traj_50_15_85_'+sstr+'.npy')
            xsmoothed = gaussian_filter1d(model_md_ht_50_15_85[1,:], sigma=2)
            ax[d,2].plot(xsmoothed, model_md_ht_50_15_85[0,:]/1, color = colors[s], linewidth = 2, linestyle = lines[0], path_effects=[pe.withStroke(linewidth=3.5, foreground='white'), pe.Normal()])

        if d == 1:
            ax[d,2].set_xlabel(cbar_lab)
        ax[d,2].set_xlim(vmin, vmax)
        ax[d,2].set_xticks(np.arange(vmin, vmax+1,1))

    """
    PLOT D, rhoeff
    """
    
    var_fn = Md_fn
    cbar_lab = 'Eff. Density\n $\mathregular{(g\ cm^{-3})}$'
    vmin, vmax = 0, 0.3
    save_fn_var = 'Md'
    save_fn_time = t_hrs_sv

    for d in range(len(datelist)):
        dstr = datelist[d]
        dstr_short = datelist_short[d]
        for s in range(len(simlist)):
            sstr = simlist[s]
            path = '/'.join(save_fn_path.split('/')[0:8]) + '/' + dstr + '/' + sstr + '/'
            save_fn_var = 'Md'
            model_md_ht_50_15_85_Md = np.load(dir_path + 'Data/' + save_fn_var+'_McSnow_2D_SIM'+'_' + dstr_short + '_at'+save_fn_time +
                                              'hrs_z'+str(nz)+'_x'+str(nx)+'_grid_'+save_fn_lab+'_traj_50_15_85_'+sstr+'.npy')
            save_fn_var = 'Vol'
            model_md_ht_50_15_85_Vol = np.load(dir_path + 'Data/' + save_fn_var+'_McSnow_2D_SIM'+'_' + dstr_short + '_at'+save_fn_time +
                                               'hrs_z'+str(nz)+'_x'+str(nx)+'_grid_'+save_fn_lab+'_traj_50_15_85_'+sstr+'.npy')
            rho_eff = model_md_ht_50_15_85_Md[1,:] / (model_md_ht_50_15_85_Vol[1,:])
            xsmoothed = gaussian_filter1d(rho_eff, sigma=2)
            ax[d,3].plot(xsmoothed, model_md_ht_50_15_85[0,:]/1, color = colors[s], linewidth = 2, linestyle = lines[0], path_effects=[pe.withStroke(linewidth=3.5, foreground='white'), pe.Normal()])

        if d == 1:
            ax[d,3].set_xlabel(cbar_lab)
        ax[d,3].set_xlim(vmin, vmax)
        ax[d,3].set_xticks([0, 0.1, 0.2, 0.3])
        ax[d,3].set_xticklabels(['0','0.1', '0.2', '0.3'])



    plt.legend(leg_lines, leg_labels, loc = 'lower center', bbox_to_anchor = (0, 0.9, 1, 1),\
           bbox_transform = plt.gcf().transFigure, ncol = 4)

    for r in range(2):
        for c in range(4):
            ax[r,c].tick_params(axis='x', colors='black')
            ax[r,c].set_ylim(0,6000)
            ax[r,c].set_yticks(np.arange(0,7000, 1000))
            ax[r,c].grid(linestyle = '--', linewidth = 0.5)
            ax[r,c].tick_params(which = 'both', direction = 'in')

            if c == 0:
                ax[r,c].set_yticklabels(np.arange(0,7,1))
                ax[r,c].set_ylabel('Height (km)')
            else:
                ax[r,c].tick_params(labelleft = False)
            if r == 0:
                ax[r,c].tick_params(labelbottom = False)


    fig.tight_layout()
    fig.subplots_adjust(bottom = 0.15)

    save_fn = dir_path + 'Figures_Combined/' + 'LWC_sensitivity' + \
        '_McSnow_2D_SIM_'+str(nz)+'_x'+str(nx)+'_grid'+'.png'
    plt.savefig(save_fn, dpi = 300)

    plt.show()
    plt.close()


#%%
if plot_updraft_sensitivity:

    plt.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots(nrows = 2, ncols = 4, figsize = (8, 6))

    bbox_let=dict(facecolor='white', edgecolor='none', boxstyle='round')
    ax[0,0].text(0.95, 0.95, '(a)', horizontalalignment='right', verticalalignment='top', transform=ax[0,0].transAxes, bbox = bbox_let, fontweight = 'semibold')
    ax[0,1].text(0.95, 0.95, '(b)', horizontalalignment='right', verticalalignment='top', transform=ax[0,1].transAxes, bbox = bbox_let, fontweight = 'semibold')
    ax[0,2].text(0.95, 0.95, '(c)', horizontalalignment='right', verticalalignment='top', transform=ax[0,2].transAxes, bbox = bbox_let, fontweight = 'semibold')
    ax[0,3].text(0.95, 0.95, '(d)', horizontalalignment='right', verticalalignment='top', transform=ax[0,3].transAxes, bbox = bbox_let, fontweight = 'semibold')
    ax[0,3].text(1.05, 0.5, '03 December', horizontalalignment='left', verticalalignment='center', transform=ax[0,3].transAxes, bbox = bbox_let, fontweight = 'regular', rotation = 90)


    ax[1,0].text(0.95, 0.95, '(e)', horizontalalignment='right', verticalalignment='top', transform=ax[1,0].transAxes, bbox = bbox_let, fontweight = 'semibold')
    ax[1,1].text(0.95, 0.95, '(f)', horizontalalignment='right', verticalalignment='top', transform=ax[1,1].transAxes, bbox = bbox_let, fontweight = 'semibold')
    ax[1,2].text(0.95, 0.95, '(g)', horizontalalignment='right', verticalalignment='top', transform=ax[1,2].transAxes, bbox = bbox_let, fontweight = 'semibold')
    ax[1,3].text(0.95, 0.95, '(h)', horizontalalignment='right', verticalalignment='top', transform=ax[1,3].transAxes, bbox = bbox_let, fontweight = 'semibold')
    ax[1,3].text(1.05, 0.5, '13 November', horizontalalignment='left', verticalalignment='center', transform=ax[1,3].transAxes, bbox = bbox_let, fontweight = 'regular', rotation = 90)

    datelist = ['DEC3SIMULATIONS', 'NOV13SIMULATIONS']
    datelist_short = ['DEC3', 'NOV13']
    simlist = ['updraft_decrease', 'updraft_increase', 'CONTROL']
    colors = ['#e41a1c', '#377eb8', 'black']

    leg_colors = ['black', '#377eb8', '#e41a1c']
    lines = ['-', '--', ':']
    leg_lines = [Line2D([0], [0], color=c, linewidth=3, linestyle='-') for c in leg_colors]
    leg_labels = ['control', 'updraft_increase', 'updraft_decrease']


    """
    PLOT A, Rf
    """
    var_fn = Rf_fn
    cbar_lab = 'Rime Fraction'
    vmin, vmax = 0, 1
    save_fn_var = 'Rf'
    save_fn_time = t_hrs_sv

    for d in range(len(datelist)):
        dstr = datelist[d]
        dstr_short = datelist_short[d]
        for s in range(len(simlist)):
            sstr = simlist[s]
            model_md_ht_50_15_85 = np.load(dir_path + 'Data/' + save_fn_var+'_McSnow_2D_SIM'+'_' + dstr_short + '_at'+save_fn_time +
                        'hrs_z'+str(nz)+'_x'+str(nx)+'_grid_'+save_fn_lab+'_traj_50_15_85_'+sstr+'.npy')

            ax[d, 0].plot(model_md_ht_50_15_85[1, :], model_md_ht_50_15_85[0, :]/1, color=colors[s], linewidth=2, linestyle=lines[0], path_effects=[pe.withStroke(linewidth=3.5, foreground='white'), pe.Normal()])

        if d == 1:
            ax[d,0].set_xlabel(cbar_lab)
        ax[d,0].set_xlim(vmin, vmax)
        ax[d,0].set_xticks([0,0.25,0.5,0.75, 1])
        ax[d,0].set_xticklabels(['0', '', '0.5', '', '1'])

    """
    PLOT B, Vt
    """
    var_fn = Vt_fn
    cbar_lab = 'Fall Velocity\n $\mathregular{(m\ s^{-1})}$'
    vmin, vmax = 0, 6
    save_fn_var = 'Vt'
    save_fn_time = t_hrs_sv

    for d in range(len(datelist)):
        dstr = datelist[d]
        dstr_short = datelist_short[d]
        for s in range(len(simlist)):
            sstr = simlist[s]
            model_md_ht_50_15_85 = np.load(dir_path + 'Data/' + save_fn_var+'_McSnow_2D_SIM'+'_' + dstr_short + '_at'+save_fn_time +
                        'hrs_z'+str(nz)+'_x'+str(nx)+'_grid_'+save_fn_lab+'_traj_50_15_85_'+sstr+'.npy')
            xsmoothed = gaussian_filter1d(model_md_ht_50_15_85[1,:], sigma=2)
            ax[d,1].plot(xsmoothed, model_md_ht_50_15_85[0,:]/1, color = colors[s], linewidth = 2, linestyle = lines[0], path_effects=[pe.withStroke(linewidth=3.5, foreground='white'), pe.Normal()])

        if d == 1:
            ax[d,1].set_xlabel(cbar_lab)
        ax[d,1].set_xlim(vmin, vmax)
        ax[d,1].set_xticks(np.arange(vmin, vmax+2,2))


    """
    PLOT C, Dm
    """
    var_fn = Dm_fn
    cbar_lab = 'Mean Diameter\n $\mathregular{(mm)}$'
    vmin, vmax = 0, 4
    save_fn_var = 'Dm'
    save_fn_time = t_hrs_sv

    for d in range(len(datelist)):
        dstr = datelist[d]
        dstr_short = datelist_short[d]
        for s in range(len(simlist)):
            sstr = simlist[s]
            model_md_ht_50_15_85 = np.load(dir_path + 'Data/' + save_fn_var+'_McSnow_2D_SIM'+'_' + dstr_short + '_at'+save_fn_time +
                        'hrs_z'+str(nz)+'_x'+str(nx)+'_grid_'+save_fn_lab+'_traj_50_15_85_'+sstr+'.npy')
            xsmoothed = gaussian_filter1d(model_md_ht_50_15_85[1,:], sigma=2)
            ax[d,2].plot(xsmoothed, model_md_ht_50_15_85[0,:]/1, color = colors[s], linewidth = 2, linestyle = lines[0], path_effects=[pe.withStroke(linewidth=3.5, foreground='white'), pe.Normal()])

        if d == 1:
            ax[d,2].set_xlabel(cbar_lab)
        ax[d,2].set_xlim(vmin, vmax)
        ax[d,2].set_xticks(np.arange(vmin, vmax+1,1))

    """
    PLOT D, rhoeff
    """
    
    var_fn = Md_fn
    cbar_lab = 'Eff. Density\n $\mathregular{(g\ cm^{-3})}$'
    vmin, vmax = 0, 0.3
    save_fn_var = 'Md'
    save_fn_time = t_hrs_sv

    for d in range(len(datelist)):
        dstr = datelist[d]
        dstr_short = datelist_short[d]
        for s in range(len(simlist)):
            sstr = simlist[s]
            path = '/'.join(save_fn_path.split('/')[0:8]) + '/' + dstr + '/' + sstr + '/'
            save_fn_var = 'Md'
            model_md_ht_50_15_85_Md = np.load(dir_path + 'Data/' + save_fn_var+'_McSnow_2D_SIM'+'_' + dstr_short + '_at'+save_fn_time +
                                              'hrs_z'+str(nz)+'_x'+str(nx)+'_grid_'+save_fn_lab+'_traj_50_15_85_'+sstr+'.npy')
            save_fn_var = 'Vol'
            model_md_ht_50_15_85_Vol = np.load(dir_path + 'Data/' + save_fn_var+'_McSnow_2D_SIM'+'_' + dstr_short + '_at'+save_fn_time +
                                               'hrs_z'+str(nz)+'_x'+str(nx)+'_grid_'+save_fn_lab+'_traj_50_15_85_'+sstr+'.npy')
            rho_eff = model_md_ht_50_15_85_Md[1,:] / (model_md_ht_50_15_85_Vol[1,:])
            xsmoothed = gaussian_filter1d(rho_eff, sigma=2)
            ax[d,3].plot(xsmoothed, model_md_ht_50_15_85[0,:]/1, color = colors[s], linewidth = 2, linestyle = lines[0], path_effects=[pe.withStroke(linewidth=3.5, foreground='white'), pe.Normal()])

        if d == 1:
            ax[d,3].set_xlabel(cbar_lab)
        ax[d,3].set_xlim(vmin, vmax)
        ax[d,3].set_xticks([0, 0.1, 0.2, 0.3])
        ax[d,3].set_xticklabels(['0','0.1', '0.2', '0.3'])


    plt.legend(leg_lines, leg_labels, loc = 'lower center', bbox_to_anchor = (0, 0.9, 1, 1),\
           bbox_transform = plt.gcf().transFigure, ncol = 4)


    for r in range(2):
        for c in range(4):
            ax[r,c].tick_params(axis='x', colors='black')
            ax[r,c].set_ylim(0,6000)
            ax[r,c].set_yticks(np.arange(0,7000, 1000))
            ax[r,c].grid(linestyle = '--', linewidth = 0.5)
            ax[r,c].tick_params(which = 'both', direction = 'in')

            if c == 0:
                ax[r,c].set_yticklabels(np.arange(0,7,1))
                ax[r,c].set_ylabel('Height (km)')
            else:
                ax[r,c].tick_params(labelleft = False)
            if r == 0:
                ax[r,c].tick_params(labelbottom = False)


    fig.tight_layout()
    fig.subplots_adjust(bottom = 0.15)

    save_fn = dir_path + 'Figures_Combined/' + 'Updraft_sensitivity' + \
        '_McSnow_2D_SIM_'+str(nz)+'_x'+str(nx)+'_grid'+'.png'
    plt.savefig(save_fn, dpi = 300)

    plt.show()
    plt.close()

#%%
"""LWC droplet diameter sensitvity 
"""
if plot_LWCdm_sensitivity:
    plt.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(8, 6))

    bbox_let = dict(facecolor='white', edgecolor='none', boxstyle='round')
    ax[0, 0].text(0.95, 0.95, '(a)', horizontalalignment='right', verticalalignment='top',
                  transform=ax[0, 0].transAxes, bbox=bbox_let, fontweight='semibold')
    ax[0, 1].text(0.95, 0.95, '(b)', horizontalalignment='right', verticalalignment='top',
                  transform=ax[0, 1].transAxes, bbox=bbox_let, fontweight='semibold')
    ax[0, 2].text(0.95, 0.95, '(c)', horizontalalignment='right', verticalalignment='top',
                  transform=ax[0, 2].transAxes, bbox=bbox_let, fontweight='semibold')
    ax[0, 3].text(0.95, 0.95, '(d)', horizontalalignment='right', verticalalignment='top',
                  transform=ax[0, 3].transAxes, bbox=bbox_let, fontweight='semibold')
    #ax[0,3].text(1.05, 0.5, '03 December', horizontalalignment='left', verticalalignment='center', transform=ax[0,3].transAxes, bbox = bbox_let, fontweight = 'regular', rotation = 90)
    ax[0, 3].text(1.05, 0.5, '03 December', horizontalalignment='left', verticalalignment='center',
                  transform=ax[0, 3].transAxes, bbox=bbox_let, fontweight='semibold', rotation=90)

    ax[1, 0].text(0.95, 0.95, '(e)', horizontalalignment='right', verticalalignment='top',
                  transform=ax[1, 0].transAxes, bbox=bbox_let, fontweight='semibold')
    ax[1, 1].text(0.95, 0.95, '(f)', horizontalalignment='right', verticalalignment='top',
                  transform=ax[1, 1].transAxes, bbox=bbox_let, fontweight='semibold')
    ax[1, 2].text(0.95, 0.95, '(g)', horizontalalignment='right', verticalalignment='top',
                  transform=ax[1, 2].transAxes, bbox=bbox_let, fontweight='semibold')
    ax[1, 3].text(0.95, 0.95, '(h)', horizontalalignment='right', verticalalignment='top',
                  transform=ax[1, 3].transAxes, bbox=bbox_let, fontweight='semibold')
    ax[1, 3].text(1.05, 0.5, '13 November', horizontalalignment='left', verticalalignment='center',
                  transform=ax[1, 3].transAxes, bbox=bbox_let, fontweight='semibold', rotation=90)

    datelist = ['DEC3SIMULATIONS', 'NOV13SIMULATIONS']
    datelist_short = ['DEC3', 'NOV13']
    simlist = ['rime_dmsmall', 'rime_dmbig', 'CONTROL']
    colors = ['#e41a1c', '#377eb8', 'black']

    leg_colors = ['black', '#377eb8', '#e41a1c']
    lines = ['-', '--', ':']
    leg_lines = [Line2D([0], [0], color=c, linewidth=3, linestyle='-')
                 for c in leg_colors]
    #leg_labels = ['control', 'SLW_diam_$\mathregular{P_{85}}$', 'SLW_diam_$\mathregular{P_{15}}$']
    leg_labels = ['control', 'SLW_diam_big', 'SLW_diam_small']

    """
    PLOT A, Rf
    """
    var_fn = Rf_fn
    cbar_lab = 'Rime Fraction'
    vmin, vmax = 0, 1
    save_fn_var = 'Rf'
    save_fn_time = t_hrs_sv

    for d in range(len(datelist)):
        dstr = datelist[d]
        dstr_short = datelist_short[d]
        for s in range(len(simlist)):
            sstr = simlist[s]
            path = '/'.join(save_fn_path.split('/')
                            [0:8]) + '/' + dstr + '/' + sstr + '/'
            model_md_ht_50_15_85 = np.load(dir_path + 'Data/' + save_fn_var+'_McSnow_2D_SIM'+'_' + dstr_short + '_at'+save_fn_time +
                        'hrs_z'+str(nz)+'_x'+str(nx)+'_grid_'+save_fn_lab+'_traj_50_15_85_'+sstr+'.npy')
            ax[d, 0].plot(model_md_ht_50_15_85[1, :], model_md_ht_50_15_85[0,
                                                                           :]/1, color=colors[s], linewidth=2, linestyle=lines[0])

        if d == 1:
            ax[d, 0].set_xlabel(cbar_lab)
        ax[d, 0].set_xlim(vmin, vmax)
        ax[d, 0].set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax[d, 0].set_xticklabels(['0', '', '0.5', '', '1'])

    """
    PLOT B, Vt
    """
    var_fn = Vt_fn
    cbar_lab = 'Fall Velocity\n $\mathregular{(m\ s^{-1})}$'
    vmin, vmax = 0, 6
    save_fn_var = 'Vt'
    save_fn_time = t_hrs_sv

    for d in range(len(datelist)):
        dstr = datelist[d]
        dstr_short = datelist_short[d]
        for s in range(len(simlist)):
            sstr = simlist[s]
            print(dstr, sstr, colors[s])
            path = '/'.join(save_fn_path.split('/')
                            [0:8]) + '/' + dstr + '/' + sstr + '/'
            print(path)
            model_md_ht_50_15_85 = np.load(dir_path + 'Data/' + save_fn_var+'_McSnow_2D_SIM'+'_' + dstr_short + '_at'+save_fn_time +
                                           'hrs_z'+str(nz)+'_x'+str(nx)+'_grid_'+save_fn_lab+'_traj_50_15_85_'+sstr+'.npy')
            xsmoothed = gaussian_filter1d(model_md_ht_50_15_85[1, :], sigma=2)
            ax[d, 1].plot(xsmoothed, model_md_ht_50_15_85[0, :]/1,
                          color=colors[s], linewidth=2, linestyle=lines[0])

        if d == 1:
            ax[d, 1].set_xlabel(cbar_lab)
        ax[d, 1].set_xlim(vmin, vmax)
        ax[d, 1].set_xticks(np.arange(vmin, vmax+2, 2))

    """
    PLOT C, Dm
    """
    var_fn = Dm_fn
    cbar_lab = 'Mean Diameter\n $\mathregular{(mm)}$'
    vmin, vmax = 0, 4
    save_fn_var = 'Dm'
    save_fn_time = t_hrs_sv

    for d in range(len(datelist)):
        dstr = datelist[d]
        dstr_short = datelist_short[d]
        for s in range(len(simlist)):
            sstr = simlist[s]
            path = '/'.join(save_fn_path.split('/')
                            [0:8]) + '/' + dstr + '/' + sstr + '/'
            model_md_ht_50_15_85 = np.load(dir_path + 'Data/' + save_fn_var+'_McSnow_2D_SIM'+'_' + dstr_short + '_at'+save_fn_time +
                                           'hrs_z'+str(nz)+'_x'+str(nx)+'_grid_'+save_fn_lab+'_traj_50_15_85_'+sstr+'.npy')
            xsmoothed = gaussian_filter1d(model_md_ht_50_15_85[1, :], sigma=2)
            ax[d, 2].plot(xsmoothed, model_md_ht_50_15_85[0, :]/1,
                          color=colors[s], linewidth=2, linestyle=lines[0])

        if d == 1:
            ax[d, 2].set_xlabel(cbar_lab)
        ax[d, 2].set_xlim(vmin, vmax)
        ax[d, 2].set_xticks(np.arange(vmin, vmax+1, 1))

    """
    PLOT D, rhoeff
    """

    var_fn = Md_fn
    cbar_lab = 'Eff. Density\n $\mathregular{(g\ cm^{-3})}$'
    vmin, vmax = 0, 0.3
    save_fn_var = 'Md'
    save_fn_time = t_hrs_sv

    for d in range(len(datelist)):
        #for d in range(1):
        dstr = datelist[d]
        dstr_short = datelist_short[d]
        for s in range(len(simlist)):
            sstr = simlist[s]
            path = '/'.join(save_fn_path.split('/')
                            [0:8]) + '/' + dstr + '/' + sstr + '/'
            save_fn_var = 'Md'
            model_md_ht_50_15_85_Md = np.load(dir_path + 'Data/' + save_fn_var+'_McSnow_2D_SIM'+'_' + dstr_short + '_at'+save_fn_time +
                        'hrs_z'+str(nz)+'_x'+str(nx)+'_grid_'+save_fn_lab+'_traj_50_15_85_'+sstr+'.npy')
            save_fn_var = 'Vol'
            model_md_ht_50_15_85_Vol = np.load(dir_path + 'Data/' + save_fn_var+'_McSnow_2D_SIM'+'_' + dstr_short + '_at'+save_fn_time +
                                               'hrs_z'+str(nz)+'_x'+str(nx)+'_grid_'+save_fn_lab+'_traj_50_15_85_'+sstr+'.npy')
            rho_eff = model_md_ht_50_15_85_Md[1,
                                              :] / (model_md_ht_50_15_85_Vol[1, :])
            xsmoothed = gaussian_filter1d(rho_eff, sigma=2)
            ax[d, 3].plot(xsmoothed, model_md_ht_50_15_85[0, :]/1,
                          color=colors[s], linewidth=2, linestyle=lines[0])

        if d == 1:
            ax[d, 3].set_xlabel(cbar_lab)
        ax[d, 3].set_xlim(vmin, vmax)
        ax[d, 3].set_xticks([0, 0.1, 0.2, 0.3])
        ax[d, 3].set_xticklabels(['0', '0.1', '0.2', '0.3'])

    plt.legend(leg_lines, leg_labels, loc='lower center', bbox_to_anchor=(0, 0.9, 1, 1),
               bbox_transform=plt.gcf().transFigure, ncol=4)

    for r in range(2):
        for c in range(4):
            ax[r, c].tick_params(axis='x', colors='black')
            ax[r, c].set_ylim(0, 6000)
            ax[r, c].set_yticks(np.arange(0, 7000, 1000))
            ax[r, c].grid(linestyle='--', linewidth=0.5)
            ax[r, c].tick_params(which='both', direction='in')

            if c == 0:
                ax[r, c].set_yticklabels(np.arange(0, 7, 1))
                ax[r, c].set_ylabel('Height (km)')
            else:
                ax[r, c].tick_params(labelleft=False)
            if r == 0:
                ax[r, c].tick_params(labelbottom=False)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.15)

    save_fn = dir_path + 'Figures_Combined/' + 'dm_sensitivity' + \
        '_McSnow_2D_SIM_'+str(nz)+'_x'+str(nx)+'_grid'+'.png'
    plt.savefig(save_fn, dpi=300)

    plt.show()
    plt.close()

#%%
if plot_depth_sensitivity:

    plt.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots(nrows = 2, ncols = 4, figsize = (8, 6))

    bbox_let=dict(facecolor='white', edgecolor='none', boxstyle='round')
    ax[0,0].text(0.95, 0.95, '(a)', horizontalalignment='right', verticalalignment='top', transform=ax[0,0].transAxes, bbox = bbox_let, fontweight = 'semibold')
    ax[0,1].text(0.95, 0.95, '(b)', horizontalalignment='right', verticalalignment='top', transform=ax[0,1].transAxes, bbox = bbox_let, fontweight = 'semibold')
    ax[0,2].text(0.95, 0.95, '(c)', horizontalalignment='right', verticalalignment='top', transform=ax[0,2].transAxes, bbox = bbox_let, fontweight = 'semibold')
    ax[0,3].text(0.95, 0.95, '(d)', horizontalalignment='right', verticalalignment='top', transform=ax[0,3].transAxes, bbox = bbox_let, fontweight = 'semibold')
    ax[0,3].text(1.05, 0.5, '03 December', horizontalalignment='left', verticalalignment='center', transform=ax[0,3].transAxes, bbox = bbox_let, fontweight = 'semibold', rotation = 90)


    ax[1,0].text(0.95, 0.95, '(e)', horizontalalignment='right', verticalalignment='top', transform=ax[1,0].transAxes, bbox = bbox_let, fontweight = 'semibold')
    ax[1,1].text(0.95, 0.95, '(f)', horizontalalignment='right', verticalalignment='top', transform=ax[1,1].transAxes, bbox = bbox_let, fontweight = 'semibold')
    ax[1,2].text(0.95, 0.95, '(g)', horizontalalignment='right', verticalalignment='top', transform=ax[1,2].transAxes, bbox = bbox_let, fontweight = 'semibold')
    ax[1,3].text(0.95, 0.95, '(h)', horizontalalignment='right', verticalalignment='top', transform=ax[1,3].transAxes, bbox = bbox_let, fontweight = 'semibold')
    ax[1,3].text(1.05, 0.5, '13 November', horizontalalignment='left', verticalalignment='center', transform=ax[1,3].transAxes, bbox = bbox_let, fontweight = 'semibold', rotation = 90)

    datelist = ['DEC3SIMULATIONS', 'NOV13SIMULATIONS']
    datelist_short = ['DEC3', 'NOV13']
    simlist = ['rime_shallow', 'rime_deep', 'CONTROL']
    colors = [bluecolor, greencolor, blackcolor]

    leg_colors = ['black', '#377eb8', '#e41a1c']
    leg_colors = [blackcolor, greencolor, bluecolor]
    lines = ['-', '--', ':']
    leg_lines = [Line2D([0], [0], color=c, linewidth=3, linestyle='-') for c in leg_colors]
    leg_labels = ['control', 'rime_deep', 'rime_shallow']


    """
    PLOT A, Rf
    """
    var_fn = Rf_fn
    cbar_lab = 'Rime Fraction'
    vmin, vmax = 0, 1
    save_fn_var = 'Rf'
    save_fn_time = t_hrs_sv

    for d in range(len(datelist)):
        dstr = datelist[d]
        dstr_short = datelist_short[d]
        if d == 1: #Nov 13
            simlist = ['rime_shallow', 'CONTROL']
            colors = [bluecolor, blackcolor]
        else:
            simlist = ['rime_shallow', 'rime_deep', 'CONTROL']
            colors = [bluecolor, greencolor, blackcolor]
        for s in range(len(simlist)):
            sstr = simlist[s]
            model_md_ht_50_15_85 = np.load(dir_path + 'Data/' + save_fn_var+'_McSnow_2D_SIM'+'_' + dstr_short + '_at'+save_fn_time +
                        'hrs_z'+str(nz)+'_x'+str(nx)+'_grid_'+save_fn_lab+'_traj_50_15_85_'+sstr+'.npy')
            ax[d,0].plot(model_md_ht_50_15_85[1,:], model_md_ht_50_15_85[0,:]/1, color = colors[s], linewidth = 2, linestyle = lines[0], path_effects=[pe.withStroke(linewidth=3.5, foreground='white'), pe.Normal()])

        if d == 1:
            ax[d,0].set_xlabel(cbar_lab)
        ax[d,0].set_xlim(vmin, vmax)
        ax[d,0].set_xticks([0,0.25,0.5,0.75, 1])
        ax[d,0].set_xticklabels(['0', '', '0.5', '', '1'])


    """
    PLOT B, Vt
    """
    var_fn = Vt_fn
    cbar_lab = 'Fall Velocity\n $\mathregular{(m\ s^{-1})}$'
    vmin, vmax = 0, 6
    save_fn_var = 'Vt'
    save_fn_time = t_hrs_sv

    for d in range(len(datelist)):
        dstr = datelist[d]
        dstr_short = datelist_short[d]
        if d == 1: #Nov 13
            simlist = ['rime_shallow', 'CONTROL']
            colors = [bluecolor, blackcolor]
        else:
            simlist = ['rime_shallow', 'rime_deep', 'CONTROL']
            colors = [bluecolor, greencolor, blackcolor]
        for s in range(len(simlist)):
            sstr = simlist[s]
            model_md_ht_50_15_85 = np.load(dir_path + 'Data/' + save_fn_var+'_McSnow_2D_SIM'+'_' + dstr_short + '_at'+save_fn_time +
                        'hrs_z'+str(nz)+'_x'+str(nx)+'_grid_'+save_fn_lab+'_traj_50_15_85_'+sstr+'.npy')
            xsmoothed = gaussian_filter1d(model_md_ht_50_15_85[1,:], sigma=2)
            ax[d, 1].plot(xsmoothed, model_md_ht_50_15_85[0, :]/1, color=colors[s], linewidth=2,
                          linestyle=lines[0], path_effects=[pe.withStroke(linewidth=3.5, foreground='white'), pe.Normal()])

        if d == 1:
            ax[d,1].set_xlabel(cbar_lab)
        ax[d,1].set_xlim(vmin, vmax)
        ax[d,1].set_xticks(np.arange(vmin, vmax+2,2))


    """
    PLOT C, Dm
    """
    var_fn = Dm_fn
    cbar_lab = 'Mean Diameter\n $\mathregular{(mm)}$'
    vmin, vmax = 0, 4
    save_fn_var = 'Dm'
    save_fn_time = t_hrs_sv

    for d in range(len(datelist)):
        dstr = datelist[d]
        dstr_short = datelist_short[d]
        if d == 1:  # Nov 13
            simlist = ['rime_shallow', 'CONTROL']
            colors = [bluecolor, blackcolor]
        else:
            simlist = ['rime_shallow', 'rime_deep', 'CONTROL']
            colors = [bluecolor, greencolor, blackcolor]
        for s in range(len(simlist)):
            sstr = simlist[s]
            model_md_ht_50_15_85 = np.load(dir_path + 'Data/' + save_fn_var+'_McSnow_2D_SIM'+'_' + dstr_short + '_at'+save_fn_time +
                        'hrs_z'+str(nz)+'_x'+str(nx)+'_grid_'+save_fn_lab+'_traj_50_15_85_'+sstr+'.npy')
            xsmoothed = gaussian_filter1d(model_md_ht_50_15_85[1,:], sigma=2)
            ax[d,2].plot(xsmoothed, model_md_ht_50_15_85[0,:]/1, color = colors[s], linewidth = 2, linestyle = lines[0], path_effects=[pe.withStroke(linewidth=3.5, foreground='white'), pe.Normal()])

        if d == 1:
            ax[d,2].set_xlabel(cbar_lab)
        ax[d,2].set_xlim(vmin, vmax)
        ax[d,2].set_xticks(np.arange(vmin, vmax+1,1))


    """
    PLOT D, rhoeff
    """
    
    var_fn = Md_fn
    cbar_lab = 'Eff. Density\n $\mathregular{(g\ cm^{-3})}$'
    vmin, vmax = 0, 0.3
    save_fn_var = 'Md'
    save_fn_time = t_hrs_sv

    for d in range(len(datelist)):
        dstr = datelist[d]
        dstr_short = datelist_short[d]
        for s in range(len(simlist)):
            sstr = simlist[s]
            path = '/'.join(save_fn_path.split('/')[0:8]) + '/' + dstr + '/' + sstr + '/'
            save_fn_var = 'Md'
            model_md_ht_50_15_85_Md = np.load(dir_path + 'Data/' + save_fn_var+'_McSnow_2D_SIM'+'_' + dstr_short + '_at'+save_fn_time +
                                              'hrs_z'+str(nz)+'_x'+str(nx)+'_grid_'+save_fn_lab+'_traj_50_15_85_'+sstr+'.npy')
            save_fn_var = 'Vol'
            model_md_ht_50_15_85_Vol = np.load(dir_path + 'Data/' + save_fn_var+'_McSnow_2D_SIM'+'_' + dstr_short + '_at'+save_fn_time +
                        'hrs_z'+str(nz)+'_x'+str(nx)+'_grid_'+save_fn_lab+'_traj_50_15_85_'+sstr+'.npy')
            rho_eff = model_md_ht_50_15_85_Md[1,:] / (model_md_ht_50_15_85_Vol[1,:])
            xsmoothed = gaussian_filter1d(rho_eff, sigma=2)

            ax[d,3].plot(xsmoothed, model_md_ht_50_15_85[0,:]/1, color = colors[s], linewidth = 2, linestyle = lines[0], path_effects=[pe.withStroke(linewidth=3.5, foreground='white'), pe.Normal()])

        if d == 1:
            ax[d,3].set_xlabel(cbar_lab)
        ax[d,3].set_xlim(vmin, vmax)
        ax[d,3].set_xticks([0, 0.1, 0.2, 0.3])
        ax[d,3].set_xticklabels(['0', '0.1',  '0.2', '0.3'])


    plt.legend(leg_lines, leg_labels, loc = 'lower center', bbox_to_anchor = (0, 0.9, 1, 1),\
           bbox_transform = plt.gcf().transFigure, ncol = 4)


    for r in range(2):
        for c in range(4):
            ax[r,c].tick_params(axis='x', colors='black')
            ax[r,c].set_ylim(0,6000)
            ax[r,c].set_yticks(np.arange(0,7000, 1000))
            ax[r,c].grid(linestyle = '--', linewidth = 0.5)
            ax[r,c].tick_params(which = 'both', direction = 'in')

            if c == 0:
                ax[r,c].set_yticklabels(np.arange(0,7,1))
                ax[r,c].set_ylabel('Height (km)')
            else:
                ax[r,c].tick_params(labelleft = False)
            if r == 0:
                ax[r,c].tick_params(labelbottom = False)


    fig.tight_layout()
    fig.subplots_adjust(bottom = 0.15)

    save_fn = dir_path + 'Figures_Combined/' + 'Depth_sensitivity'+'_McSnow_2D_SIM_'+str(nz)+'_x'+str(nx)+'_grid'+'.png'
    plt.savefig(save_fn, dpi = 300)

    plt.show()
    plt.close()

#%%
if plot_prcp_LWC_sensitivity:
    plt.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots(nrows = 2, ncols = 1, figsize = (6, 5))

    bbox_let=dict(facecolor='white', edgecolor='none', boxstyle='round')
    ax[0].text(0.95, 0.95, '(a) 03 Dec.', horizontalalignment='right', verticalalignment='top', transform=ax[0].transAxes, bbox = bbox_let, fontweight = 'semibold')
    ax[1].text(0.95, 0.95, '(b) 13 Nov.', horizontalalignment='right', verticalalignment='top', transform=ax[1].transAxes, bbox = bbox_let, fontweight = 'semibold')

    datelist = ['DEC3SIMULATIONS', 'NOV13SIMULATIONS']
    datelist_short = ['DEC3', 'NOV13']

    simlist = ['rime_heavy', 'CONTROL', 'rime_light','rime_off']
    colors = [greencolor, blackcolor, bluecolor, redcolor]

    leg_colors = ['black', '#377eb8', '#e41a1c', '#5e3c99']
    leg_colors = [blackcolor, greencolor, bluecolor, redcolor]
    lines = ['-', '--', ':']
    leg_lines = [Line2D([0], [0], color=c, linewidth=3, linestyle='-') for c in leg_colors]
    leg_labels = ['control', 'SLW_heavy', 'SLW_light', 'SLW_none']

    for d in range(len(datelist)):
        dstr = datelist[d]
        dstr_short = datelist_short[d]
        for s in range(len(simlist)):
            sstr = simlist[s]
            prcp_xy_bar = np.load(dir_path+'Data/'+'RR_surf_1hr_BAR_McSnow_2D_SIM'+'_' + dstr_short + '_at'+save_fn_time +
                'hrs_z'+str(nz)+'_x'+str(nx)+'_grid_'+save_fn_lab+'_'+sstr+'.npy')
            prcp_xy_smooth = np.load(dir_path+'Data/'+'RR_surf_1hr_SMOOTH_McSnow_2D_SIM'+'_' + dstr_short + '_at'+save_fn_time +
                'hrs_z'+str(nz)+'_x'+str(nx)+'_grid_'+save_fn_lab+'_'+sstr+'.npy')

            ax[d].bar(prcp_xy_bar[0,:], prcp_xy_bar[1,:], color = colors[s], alpha = 0.75)
            ax[d].plot(prcp_xy_smooth[0,:], prcp_xy_smooth[1,:], color = colors[s], linewidth = 2, path_effects=[pe.withStroke(linewidth=3.5, foreground='white'), pe.Normal()])


    for axi in ax:
        axi.set_xticks(xticks)
        axi.set_xlim(nx*0.2,nx*0.7)
        axi.set_yticks(np.arange(0,6,1))
        axi.set_ylim(0,5)
        axi.set_ylabel('$\mathregular{Rain\ Rate\ (mm\ hr^{-1})}$')

        axi.grid(linestyle = '--', linewidth = 0.5)
        axi.tick_params(which = 'both', direction = 'in')

    ax[0].tick_params(labelbottom = False)
    ax[1].set_xticklabels(xlabs)
    ax[1].set_xlabel('Distance from Origin (km)')

    ax[0].legend(leg_lines, leg_labels, ncol = 1, loc = 'lower right')
    fig.tight_layout()

    save_fn = dir_path + 'Figures_Combined/' + 'prcp_LWC_sensitivity' + \
        '_McSnow_2D_SIM_'+str(nz)+'_x'+str(nx)+'_grid'+'.png'
    plt.savefig(save_fn, dpi = 300)

    plt.show()
    plt.close()

#%%
if plot_prcp_depth_sensitivity:
    plt.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots(nrows = 2, ncols = 1, figsize = (6, 5))

    bbox_let=dict(facecolor='white', edgecolor='none', boxstyle='round')
    ax[0].text(0.95, 0.95, '(a) 03 Dec.', horizontalalignment='right', verticalalignment='top', transform=ax[0].transAxes, bbox = bbox_let, fontweight = 'semibold')
    ax[1].text(0.95, 0.95, '(b) 13 Nov.', horizontalalignment='right', verticalalignment='top', transform=ax[1].transAxes, bbox = bbox_let, fontweight = 'semibold')

    datelist = ['DEC3SIMULATIONS', 'NOV13SIMULATIONS']
    datelist_short = ['DEC3', 'NOV13']

    simlist = ['rime_deep', 'CONTROL', 'rime_shallow']
    colors = [greencolor, blackcolor, bluecolor]

    leg_colors = [blackcolor, greencolor, bluecolor]
    lines = ['-', '--', ':']
    leg_lines = [Line2D([0], [0], color=c, linewidth=3, linestyle='-') for c in leg_colors]
    leg_labels = ['control', 'rime_deep', 'rime_shallow']

    for d in range(len(datelist)):
        dstr = datelist[d]
        dstr_short = datelist_short[d]
        if d == 1:  # Nov 13
            simlist = ['CONTROL', 'rime_shallow',]
            colors = [blackcolor, bluecolor]
        else:
            simlist = ['rime_deep', 'CONTROL', 'rime_shallow']
            colors = [greencolor, blackcolor, bluecolor]
        for s in range(len(simlist)):
            sstr = simlist[s]

            prcp_xy_bar = np.load(dir_path+'Data/'+'RR_surf_1hr_BAR_McSnow_2D_SIM'+'_' + dstr_short + '_at'+save_fn_time +
                'hrs_z'+str(nz)+'_x'+str(nx)+'_grid_'+save_fn_lab+'_'+sstr+'.npy')
            prcp_xy_smooth = np.load(dir_path+'Data/'+'RR_surf_1hr_SMOOTH_McSnow_2D_SIM'+'_' + dstr_short + '_at'+save_fn_time +
                'hrs_z'+str(nz)+'_x'+str(nx)+'_grid_'+save_fn_lab+'_'+sstr+'.npy')
           

            ax[d].bar(prcp_xy_bar[0,:], prcp_xy_bar[1,:], color = colors[s], alpha = 0.75)
            ax[d].plot(prcp_xy_smooth[0,:], prcp_xy_smooth[1,:], color = colors[s], linewidth = 2, path_effects=[pe.withStroke(linewidth=3.5, foreground='white'), pe.Normal()])


    for axi in ax:
        axi.set_xticks(xticks)
        axi.set_xlim(nx*0.2,nx*0.7)
        axi.set_yticks(np.arange(0,6,1))
        axi.set_ylim(0,5)
        axi.set_ylabel('$\mathregular{Rain\ Rate\ (mm\ hr^{-1})}$')

        axi.grid(linestyle = '--', linewidth = 0.5)
        axi.tick_params(which = 'both', direction = 'in')

    ax[0].tick_params(labelbottom = False)
    ax[1].set_xticklabels(xlabs)
    ax[1].set_xlabel('Distance from Origin (km)')

    ax[0].legend(leg_lines, leg_labels, ncol = 1, loc = 'lower right')
    fig.tight_layout()

    save_fn = dir_path + 'Figures_Combined/' + 'prcp_depth_sensitivity'+'_McSnow_2D_SIM_'+str(nz)+'_x'+str(nx)+'_grid'+'.png'
    plt.savefig(save_fn, dpi = 300)

    plt.show()
    plt.close()

# %%
