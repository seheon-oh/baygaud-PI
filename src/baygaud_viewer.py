#!/usr/bin/env python3
# -*- coding: utf-8 -*-   

#|-----------------------------------------|
#| baygaud_viewer.py
#|-----------------------------------------|
#| by Se-Heon Oh
#| Dept. of Physics and Astronomy
#| Sejong University, Seoul, South Korea
#|-----------------------------------------|

import glob
import os
import sys
from tkinter import *
from tkinter import filedialog, messagebox, ttk

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from spectral_cube import SpectralCube
from _baygaud_params import default_params, read_configfile

title = 'baygaud-PI viewer'

dict_params = {'cursor_xy':(-1,-1), 'multiplier_cube':1000.0, 'unit_cube':r'mJy$\,$beam$^{-1}$', 'multiplier_spectral_axis':0.001}
dict_data = {}
dict_plot = {'fix_cursor':False}
plt.rcParams["hatch.linewidth"] = 4
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"

colors = ['tab:blue', 'tab:orange', 'tab:red', 'tab:green', 'tab:purple', 'tab:yellow', 'tab:black', 'tab:magenta', 'tab:cyan', \
        'tab:blue', 'tab:orange', 'tab:red', 'tab:green', 'tab:purple', 'tab:yellow', 'tab:black', 'tab:magenta', 'tab:cyan']


#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def gauss_model(x, amp, vel, disp):
    return amp * np.exp(-((x - vel) ** 2) / (2 * disp ** 2))


#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def colorbar(img, spacing=0, cbarwidth=0.01, orientation='vertical', pos='right', label='', ticks=[0], fontsize=12):

    ax = img.axes
    fig = ax.figure

    if orientation == 'vertical':
        if pos == 'right':
            cax = fig.add_axes([ax.get_position().x1 + spacing, ax.get_position().y0, cbarwidth, ax.get_position().height])

        elif pos == 'left':
            cax = fig.add_axes([ax.get_position().x0 - spacing - cbarwidth, ax.get_position().y0, cbarwidth, ax.get_position().height])
            cax.yaxis.set_ticks_position('left')

    elif orientation == 'horizontal':
        if pos == 'top':
            cax = fig.add_axes([ax.get_position().x0, ax.get_position().y1 + spacing, ax.get_position().width, cbarwidth])
            cax.tick_params(axis='x', labelbottom=False, labeltop=True)

        elif pos == 'bottom':
            cax = fig.add_axes([ax.get_position().x0, ax.get_position().y0 - spacing - cbarwidth, ax.get_position().width, cbarwidth])

    cbar = plt.colorbar(img, cax=cax, orientation=orientation, ticks=ticks) if len(ticks) != 1 else plt.colorbar(img, cax=cax, orientation=orientation)
    cbar.set_label(label=label, fontsize=fontsize)

    return cbar, cax

#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def panel_label(ax, text, xpos=0.05, ypos=0.95, color='black', fontsize=10, inside_box=False, pad=5.0):

    font_props = {'fontsize': fontsize, 'color': color, 'verticalalignment': 'top'}
    bbox = {'facecolor': 'none', 'edgecolor': color if inside_box else 'none', 'pad': pad}
    ax.text(xpos, ypos, text, transform=ax.transAxes, bbox=bbox, **font_props)


#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def fillentry(entry, content):

    entry['state'] = 'normal'
    entry.delete(0, "end")
    entry.insert(0, content)

    if entry['state'] == 'readonly':
        entry['state'] = 'readonly'

#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def makelabelentry(frame, array, title=[], startcol=0, widthlabel=10, widthentry=10):

    if not title:
        title = array

    for i, content in enumerate(array):
        label = Label(frame, text=title[i], width=widthlabel, anchor='e')
        label.grid(row=i + startcol, column=0, padx=5)
        entry = Entry(frame, width=widthentry, justify='right')
        entry.grid(row=i + startcol, column=1)

#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def initdisplay():

    if 'fig1' not in dict_plot:
        fig1, ax1 = plt.subplots()#tight_layout=True)
        fig1.set_figwidth(1.5*500/fig1.dpi)
        fig1.set_figheight(1.5*460/fig1.dpi)
        fig1.subplots_adjust(left=0.1, right=0.85, top=0.99, bottom=0.05)

        canvas1 = FigureCanvasTkAgg(fig1, master=frame_display)   #DRAWING FIGURES ON GUI FRAME
        canvas1.draw()
        canvas1.get_tk_widget().pack(side=TOP)#, fill=BOTH, expand=True)
        fig1.canvas.mpl_connect('motion_notify_event', cursor_coords)  #CONNECTING MOUSE CLICK ACTION
        fig1.canvas.mpl_connect('scroll_event', zoom)


        fig2, (ax2, ax3) = plt.subplots(nrows=2, sharex=True)
        fig2.set_figwidth(1.5*500/fig2.dpi)
        fig2.set_figheight(1.5*500/fig2.dpi)
        fig2.subplots_adjust(hspace=0, top=0.94, bottom=0.18)

        ax2.plot(dict_data['spectral_axis'], np.zeros_like(dict_data['spectral_axis']))

        canvas2 = FigureCanvasTkAgg(fig2, master=frame_line)
        canvas2.draw()
        canvas2.get_tk_widget().pack(side=TOP, fill=BOTH, expand=True)

        dict_plot['fig1']    = fig1
        dict_plot['ax1']     = ax1
        dict_plot['canvas1'] = canvas1

        dict_plot['fig2']    = fig2
        dict_plot['ax2']     = ax2
        dict_plot['ax3']     = ax3
        dict_plot['canvas2'] = canvas2


    dict_plot['ax1'].clear()
    dict_plot['ax1'].set_xlabel('x', fontsize=16)
    dict_plot['ax1'].set_ylabel('y', fontsize=16)

    if 'cax' in dict_plot:
        dict_plot['cax'].remove()

    path_map = glob.glob(dict_params['path_fig1'])[0]
    data = fits.getdata(path_map)
    
    def isolate_largest(data):
    
        from skimage import measure
        
        mask = np.full_like(data, data)
        
        mask[~np.isnan(mask)] = 1
        mask[np.isnan(mask)] = 0

        mask_labeled = measure.label(mask, connectivity=1)

        count_max = 0
        index_max = 0
        for i in range(np.max(mask_labeled)):
            if(i==0): continue

            count = np.count_nonzero(mask_labeled==i)

            if(count>count_max):
                count_max = count
                index_max = i

        mask = np.where(mask_labeled==index_max, 1., 0.)
        mask[mask==0.] = np.nan
        
        return mask
    
    mask = isolate_largest(data)
        
    data = np.where(np.isnan(mask), np.nan, data)    

    var = var_mapselect.get()
    if(var=='Integrated flux'):
        clim = np.nanpercentile(data, (0,99.9))
        label_cbar = r'Int. flux (Jy beam$^{-1}$ km s$^{-1}$)'
    if(var=='SGfit V.F.'):
        clim = np.nanpercentile(data, (2,98))
        label_cbar = r'LoS velocity (km s$^{-1}$)'
    if(var=='SGfit VDISP'):
        clim = (0,50)
        label_cbar = r'Velocity dispersion (km s$^{-1}$)'
    if(var=='N-Gauss'):
        clim = (1,_params['max_ngauss'])
        label_cbar = r'$N_\mathrm{gauss}$'
    if(var=='SGfit peak S/N'):
        clim = (0, np.nanmax(data))
        label_cbar = r'Peak S/N'
    
    if 'clim_{}'.format(var) in dict_plot:
        clim = dict_plot['clim_{}'.format(var)]
    else: dict_plot['clim_{}'.format(var)] = clim
    
    img1 = dict_plot['ax1'].imshow(fits.getdata(path_map), interpolation='none', cmap='jet', clim=clim)

    fillentry(entry_climlo, clim[0])
    fillentry(entry_climhi, clim[1])

    dict_plot['ax1'].invert_yaxis()
    #_,dict_plot['cax'] = colorbar(img1, cbarwidth=0.03, ticks=[0,1,2,3,4,5])
    _,dict_plot['cax'] = colorbar(img1, cbarwidth=0.03, label=label_cbar)

    dict_plot['canvas1'].draw()

    dict_plot['ax2'].clear()
    dict_plot['ax3'].clear()
    
    # dict_plot['ax2'].plot(dict_data['spectral_axis'], np.zeros_like(dict_data['spectral_axis']))
    # dict_plot['ax3'].plot(dict_data['spectral_axis'], np.zeros_like(dict_data['spectral_axis']))

    dict_plot['canvas2'].draw()

    # plt.close(fig)

#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def read_ngfit(path_cube=None, path_classified=None):
    if path_cube:
        dict_params['path_cube'] = path_cube
    if path_classified:
        dict_params['path_classified'] = path_classified

    dict_params['path_fig1'] = f"{dict_params['path_classified']}/ngfit/ngfit.G*_1.1.fits"
    dict_data['cube'] = fits.getdata(dict_params['path_cube']) * dict_params['multiplier_cube']
    if(len(dict_data['cube'].shape)>3): dict_data['cube'] = dict_data['cube'][0,:,:,:]
    dict_data['spectral_axis'] = SpectralCube.read(dict_params['path_cube']).spectral_axis.value * dict_params['multiplier_spectral_axis']
    dict_data['imsize'] = dict_data['cube'][0, :, :].shape

    n_gauss = _params['max_ngauss']
    amps        = np.empty(n_gauss, dtype=object)
    vels        = np.empty(n_gauss, dtype=object)
    disps       = np.empty(n_gauss, dtype=object)
    ngfit_bgs   = np.empty(n_gauss, dtype=object)
    ngfit_rms   = np.empty(n_gauss, dtype=object)
    ngfit_sn    = np.empty(n_gauss, dtype=object)

    sgfit_bg    = fits.getdata(glob.glob(f"{dict_params['path_classified']}/ngfit/ngfit.G{n_gauss}_1.3.fits")[0])
    data_noise  = fits.getdata(glob.glob(f"{dict_params['path_classified']}/sgfit/sgfit.G{n_gauss}_1.4.fits")[0])
    dict_data['noise'] = data_noise

    for i in range(n_gauss):
        name_amp        = glob.glob(f"{dict_params['path_classified']}/ngfit/ngfit.G{n_gauss}_{i+1}.5.fits")[0]
        name_vel        = glob.glob(f"{dict_params['path_classified']}/ngfit/ngfit.G{n_gauss}_{i+1}.1.fits")[0]
        name_disp       = glob.glob(f"{dict_params['path_classified']}/ngfit/ngfit.G{n_gauss}_{i+1}.2.fits")[0]
        ngfit_bg_slice  = glob.glob(f"{dict_params['path_classified']}/ngfit/ngfit.G{n_gauss}_{i+1}.3.fits")[0]
        ngfit_rms_slice = glob.glob(f"{dict_params['path_classified']}/ngfit/ngfit.G{n_gauss}_{i+1}.4.fits")[0]
        ngfit_sn_slice  = glob.glob(f"{dict_params['path_classified']}/ngfit/ngfit.G{n_gauss}_{i+1}.6.fits")[0]

    
        amps[i]   = fits.getdata(name_amp)
        vels[i]   = fits.getdata(name_vel)
        disps[i]  = fits.getdata(name_disp)
        ngfit_bgs[i]  = fits.getdata(ngfit_bg_slice)
        ngfit_rms[i]  = fits.getdata(ngfit_rms_slice)
        ngfit_sn[i]  = fits.getdata(ngfit_sn_slice)

        #print(ngfit_rms[i][24,38])
        #sys.exit()

    del data_noise

    dict_data['amps']  = amps
    dict_data['vels']  = vels
    dict_data['disps'] = disps
    dict_data['bg']    = ngfit_bgs
    dict_data['rms']    = ngfit_rms
    dict_data['sn']    = ngfit_sn

    initdisplay()


#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def loaddata():

    def browse_cube():
        path_cube = filedialog.askopenfilename(title='Path to cube', filetypes=[('FITS file', '.fits .FITS')])
        if(len(path_cube)==0): return

        print(path_cube)

        if(len(fits.getdata(path_cube).shape)<3 or len(SpectralCube.read(path_cube).spectral_axis)==1):
            messagebox.showerror("Error", "Cube should have at least three dimensions.")
            return
        
        fillentry(entry_path_cube, path_cube)

        #possible_path_classified = glob.glob(os.path.dirname(path_cube)+'/baygaud_output*/output_merged/classified*')
        possible_path_classified = glob.glob(os.path.dirname(path_cube) + '/' + _params['_combdir'] + '.%d' % _classified_index)
        if(len(possible_path_classified)==1):
            browse_classified(possible_path_classified[0])
        elif(len(possible_path_classified)>1):
            browse_classified(initialdir=os.path.dirname(possible_path_classified[0]))

    def browse_classified(path_classified=None, initialdir=None):
        if(path_classified==None):
            path_classified = filedialog.askdirectory(title='Path to classified directory', initialdir=initialdir)
            if(len(path_classified)==0): return

        #ifexists = os.path.exists(path_classified+"/single_gfit")
        ifexists = os.path.exists(path_classified)

        if(ifexists==False):
            messagebox.showerror("Error", "No proper data found inside.")
            return

        fillentry(entry_path_classified, path_classified)  

    def btncmd_toplv_browse_cube():
        browse_cube()

    def btncmd_toplv_browse_classified():
        browse_classified()



    def btncmd_toplv_apply():
        dict_params['path_cube'] = entry_path_cube.get()
        dict_params['path_classified'] = entry_path_classified.get()
        read_ngfit()

        dict_plot['toplv'].destroy()
   

    def btncmd_toplv_cancel():
        toplv.destroy()

    toplv = Toplevel(root)

    frame_toplv1 = Frame(toplv)
    frame_toplv2 = Frame(toplv)

    makelabelentry(frame_toplv1, ['path_cube', 'path_classified'], [], 0, 20, 20)

    btn_toplv_browsecube = Button(frame_toplv1, text='Browse', command=btncmd_toplv_browse_cube)
    btn_toplv_browsecube.grid(row=0, column=2)

    btn_toplv_browseclassified = Button(frame_toplv1, text='Browse', command=btncmd_toplv_browse_classified)
    btn_toplv_browseclassified.grid(row=1, column=2)

    ttk.Separator(frame_toplv2, orient='horizontal').pack(fill=BOTH)

    btn_toplv_apply = Button(frame_toplv2, text='Apply', command=btncmd_toplv_apply)
    btn_toplv_cancel = Button(frame_toplv2, text='Cancel', command=btncmd_toplv_cancel)
    btn_toplv_cancel.pack(side='right')
    btn_toplv_apply.pack(side='right')

    frame_toplv1.pack()
    frame_toplv2.pack(fill=BOTH)

    dict_plot['toplv'] = toplv

#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def apply_mapselect(*args):

    var = var_mapselect.get()
    n_gauss = _params['max_ngauss']
    # ['Integrated flux', 'SGfit velocity', 'SGfit vdisp', 'Ngauss', 'SGfit Peak S/N']

    if(var=='Integrated flux'):
        dict_params['path_fig1'] = dict_params['path_classified']+'/sgfit/sgfit.G%d_1.0.fits' % n_gauss
    if(var=='SGfit V.F.'):
        dict_params['path_fig1'] = dict_params['path_classified']+'/sgfit/sgfit.G%d_1.1.fits' % n_gauss
    if(var=='SGfit VDISP'):
        dict_params['path_fig1'] = dict_params['path_classified']+'/sgfit/sgfit.G%d_1.2.fits' % n_gauss
    if(var=='N-Gauss'):
        dict_params['path_fig1'] = dict_params['path_classified']+'/sgfit/sgfit.G%d_1.7.fits' % n_gauss
    if(var=='SGfit peak S/N'):
        dict_params['path_fig1'] = dict_params['path_classified']+'/sgfit/sgfit.G%d_1.6.fits' % n_gauss

    initdisplay()
    
    
#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def apply_clim(*args):

    climlo = float(var_climlo.get())
    climhi = float(var_climhi.get())

    mapname = var_mapselect.get()
    
    dict_plot['clim_{}'.format(mapname)] = [climlo, climhi]

    initdisplay()
    
#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def callback_entry_climlo_clicked(*args):
    entry_climlo.selection_range(0, END)
    
#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def callback_entry_climhi_clicked(*args):
    entry_climhi.selection_range(0, END)


#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def fix_cursor(event):
    dict_plot['fix_cursor'] = (dict_plot['fix_cursor']+1)%2

#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def plot_profiles():
    try:
        n_gauss = _params['max_ngauss']
        x, y = dict_params['cursor_xy']
        ax2, ax3 = dict_plot['ax2'], dict_plot['ax3']
        ax2.clear()
        ax3.clear()

        bg = dict_data['bg'][0][y, x] * dict_params['multiplier_cube']
        rms = dict_data['rms'][0][y, x] * dict_params['multiplier_cube']
        rms_axis = np.full_like(dict_data['spectral_axis'], rms)
        spectral_axis = dict_data['spectral_axis']
        cube = dict_data['cube'][:, y, x]

        ax2.step(spectral_axis, cube)
        subed = np.full_like(cube, cube)
        total = np.zeros_like(spectral_axis)

        dict_params['path_fig1'] = f"{dict_params['path_classified']}/sgfit/sgfit.G%d_1.7.fits" % n_gauss
        ng_opt_fits = glob.glob(dict_params['path_fig1'])[0]
        ng_opt = fits.getdata(ng_opt_fits)

        if(np.isnan(ng_opt[y,x])==False):                                                                                                                                                        
    
            for i in range(ng_opt[y, x].astype(int)):
                vel = dict_data['vels'][i][y, x]
                disp = dict_data['disps'][i][y, x]
                amp = dict_data['amps'][i][y, x]
                sn = dict_data['sn'][i][y, x]

                if np.any(np.isnan([vel, disp, amp])):
                    continue

                ploty = gauss_model(spectral_axis, amp, vel, disp) * dict_params['multiplier_cube']
                total += ploty
                ploty += bg
                ax2.plot(spectral_axis, ploty, label=f'G{i + 1} (S/N: {sn:.2f})', color=colors[i], ls='-', alpha=0.5)
                ax2.legend(loc='upper right')
                ploty -= bg

            panel_label(dict_plot['ax2'], '(x, y: N-Gauss)=(%d, %d: %d)' % (x, y, ng_opt[y, x]), fontsize=13)
        panel_label(dict_plot['ax3'], 'Residuals', fontsize=13)

        total += bg
        subed -= total

        dict_plot['ax2'].plot(dict_data['spectral_axis'], total, color='red', ls='--', linewidth=3, alpha=0.5)
        dict_plot['ax3'].step(dict_data['spectral_axis'], subed, color='orange', ls='-', alpha=0.7)
        dict_plot['ax3'].plot(dict_data['spectral_axis'], rms_axis, color='purple', ls='--', alpha=0.7)
        dict_plot['ax3'].plot(dict_data['spectral_axis'], -1*rms_axis, color='purple', ls='--', alpha=0.7)

        dict_plot['ax2'].text(-0.12, -0, 'Flux density ({})'.format(dict_params['unit_cube']), ha='center', va='center', transform = dict_plot['ax2'].transAxes, rotation=90, fontsize=16)
        dict_plot['ax3'].set_xlabel(r'Spectral axis (km$\,$s$^{-1}$)', fontsize=16)

        dict_plot['ax2'].margins(x=0.02, y=0.15)
        dict_plot['ax3'].margins(x=0.02, y=0.05)

        dict_plot['ax2'].xaxis.set_tick_params(labelsize=14)
        dict_plot['ax2'].yaxis.set_tick_params(labelsize=14)
        dict_plot['ax3'].xaxis.set_tick_params(labelsize=14)
        dict_plot['ax3'].yaxis.set_tick_params(labelsize=14)

        dict_plot['canvas2'].draw()
    except IndexError:
        pass


#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def cursor_coords(event):
    if not dict_plot['fix_cursor']:
        if event.inaxes:
            cursor_xy = (round(event.xdata), round(event.ydata))
            if dict_params['cursor_xy'] != cursor_xy:
                dict_params['cursor_xy'] = cursor_xy
                plot_profiles()

#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def zoom(event):
    ax = dict_plot['ax1']
    canvas = dict_plot['canvas1']
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    xdata, ydata = event.xdata, event.ydata
    base_scale = 2
    scale_factor = 1 / base_scale if event.button == 'up' else base_scale if event.button == 'down' else 1
    width = (xlim[1] - xlim[0]) * scale_factor
    height = (ylim[1] - ylim[0]) * scale_factor
    relx = (xlim[1] - xdata) / (xlim[1] - xlim[0])
    rely = (ylim[1] - ydata) / (ylim[1] - ylim[0])
    new_xlim = (np.max([xdata - width * (1 - relx), 0]), np.min([xdata + width * relx, dict_data['imsize'][1] - 1]))
    new_ylim = (np.max([ydata - height * (1 - rely), 0]), np.min([ydata + height * rely, dict_data['imsize'][0] - 1]))
    ax.set_xlim(*new_xlim)
    ax.set_ylim(*new_ylim)
    canvas.draw()


#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
# TK ROOT()
root = Tk()

root.title(title)
root.resizable(False, False)

menubar = Menu(root)

frame_master = Frame(root)
frame_L = Frame(frame_master, height=500, width=550, bg='white')
frame_M = Frame(frame_master, height=500, width=50, bg='white')
frame_R = Frame(frame_master, height=500, width=500, bg='white')

frame_display = Frame(frame_L, height=500, width=550, bg='white')
frame_display.pack()

frame_LB = Frame(frame_L)

frame_LB_space = Frame(frame_LB, width=200)
frame_LB_space.pack(side='left')

frame_LB_climlo = Frame(frame_LB)
var_climlo = StringVar()
label_climlo = Label(frame_LB_climlo, text='clim_low', width=10, anchor='e')
entry_climlo = Entry(frame_LB_climlo, width=10, justify='right', textvariable=var_climlo, validate="focusout", validatecommand=apply_clim)
entry_climlo.bind("<FocusIn>", callback_entry_climlo_clicked)
label_climlo.pack(side='left')
entry_climlo.pack(side='right')
frame_LB_climlo.pack(side='left')

frame_LB_climhi = Frame(frame_LB)
var_climhi = StringVar()
label_climhi = Label(frame_LB_climhi, text='clim_high', width=10, anchor='e')
entry_climhi = Entry(frame_LB_climhi, width=10, justify='right', textvariable=var_climhi, validate="focusout", validatecommand=apply_clim)
entry_climhi.bind("<FocusIn>", callback_entry_climhi_clicked)
label_climhi.pack(side='left')
entry_climhi.pack(side='right')
frame_LB_climhi.pack(side='left')

OptionList = ['Integrated flux', 'SGfit V.F.', 'SGfit VDISP', 'N-Gauss', 'SGfit peak S/N']
var_mapselect = StringVar()
var_mapselect.set(OptionList[1])

dropdown_mapselect = OptionMenu(frame_LB, var_mapselect, *OptionList)
dropdown_mapselect.pack(side='right')
var_mapselect.trace("w", apply_mapselect)

frame_LB.pack(fill=BOTH, expand=True)

frame_line = Frame(frame_R, width=500,height=500, bg='white')
frame_line.pack()

frame_L.pack(fill=BOTH, expand=True, side='left')
frame_M.pack(fill=BOTH, expand=True, side='left')
frame_R.pack(fill=BOTH, expand=True, side='right')
frame_master.pack(fill=BOTH, expand=True)


root.config(menu=menubar)
root.bind('f', fix_cursor)
root.bind('<Return>', apply_clim)

if len(sys.argv) == 1:
    print("")
    print(91*"_")
    print(91*"")
    print(" :: baygaud_viewer.py usage ::")
    print(91*"")
    print(" usage-1: running baygaud_viewer.py with baygaud_params file")
    print(" > python3 baygaud_viewer.py [ARG1: _baygaud_params.txt] [ARG2: output-index, 1, 2, ...]")
    print(" output-index is the postfix number of the baygaud segments merged directory.")
    print(" i.e., 'segmts_merged_n_classified.[output-index]' in 'wdir'")
    print(" e.g.,")
    print(" > python3 baygaud_viewer.py _baygaud_params.ngc2403.txt 1")

    print(91*"-")
    print(" usage-2: running baygaud_viewer.py with the DEFAULT baygaud_params file")
    print("        : the DEFAULT baygaud_params file: _baygaud_params.py")
    print(" > python3 baygaud_viewer.py [ARG1: output-index, 1, 2, ...]")
    print(" e.g.,")
    print(" > python3 baygaud_viewer.py 1")
    print(91*"_")
    print("")
    sys.exit()

elif len(sys.argv) == 2:
    ("WARNING: No configfile supplied, trying default values")
    _params=default_params()
    _classified_index = int(sys.argv[1])


elif len(sys.argv) == 3:
    configfile = sys.argv[1]
    _params=read_configfile(configfile)
    _classified_index = int(sys.argv[2])


_path_cube = f"{_params['wdir']}/{_params['input_datacube']}"
_path_classified = f"{_params['wdir']}/{_params['_combdir']}.{_classified_index}"
read_ngfit(path_cube=_path_cube, path_classified=_path_classified)

root.mainloop()


#-- END OF SUB-ROUTINE____________________________________________________________#

