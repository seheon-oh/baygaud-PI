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
import signal
from tkinter import *
from tkinter import filedialog, messagebox, ttk
from tkinter.font import Font

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import astropy.units as u
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from spectral_cube import SpectralCube
from _baygaud_params import read_configfile

title = 'baygaud-PI viewer'

window_params = {'cursor_xy': (-1, -1), 'tomJy': 1000.0, 'unit_cube': r'mJy$\,$beam$^{-1}$', 'tokms': 0.001}
_params = {}
window_plot = {'fix_cursor': False}
plt.rcParams["hatch.linewidth"] = 1
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"

colors = ['tab:blue', 'tab:orange', 'tab:red', 'tab:green', 'tab:purple', 'tab:yellow', 'tab:black', 'tab:magenta', 'tab:cyan'] * 2

#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def gauss_model(x, amp, vel, disp):  # no bg added
    return amp * np.exp(-((x - vel) ** 2) / (2 * disp ** 2))

#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def create_colorbar(img, spacing=0, cbarwidth=0.01, orientation='vertical', pos='right', label='', ticks=[0], fontsize=10):
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
def update_entry(entry, content):
    entry['state'] = 'normal'
    entry.delete(0, "end")
    entry.insert(0, content)
    if entry['state'] == 'readonly':
        entry['state'] = 'readonly'

#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def create_label_and_entry(frame, array, title=[], startcol=0, widthlabel=10, widthentry=10):
    if not title:
        title = array

    for i, content in enumerate(array):
        label = Label(frame, text=title[i], width=widthlabel, anchor='e')
        label.grid(row=i + startcol, column=0, padx=5)
        entry = Entry(frame, width=widthentry, justify='right')
        entry.grid(row=i + startcol, column=1)

#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def initialize_display():
    if 'fig1' not in window_plot:
        fig1, ax1 = plt.subplots()
        fig1.set_figwidth(1.45 * 500 / fig1.dpi)
        fig1.set_figheight(1.45 * 460 / fig1.dpi)
        fig1.subplots_adjust(left=0.1, right=0.85, top=0.99, bottom=0.05)

        canvas1 = FigureCanvasTkAgg(fig1, master=frame_display)
        canvas1.draw()
        canvas1.get_tk_widget().pack(side=TOP, fill=BOTH, expand=True)
        fig1.canvas.mpl_connect('motion_notify_event', update_cursor_coords)
        fig1.canvas.mpl_connect('scroll_event', handle_zoom)

        fig2, (ax2, ax3) = plt.subplots(nrows=2, sharex=True)
        fig2.set_figwidth(1.45 * 500 / fig2.dpi)
        fig2.set_figheight(1.45 * 500 / fig2.dpi)
        fig2.subplots_adjust(hspace=0, top=0.94, bottom=0.18)

        ax2.plot(_params['spectral_axis'], np.zeros_like(_params['spectral_axis']))

        canvas2 = FigureCanvasTkAgg(fig2, master=frame_line)
        canvas2.draw()
        canvas2.get_tk_widget().pack(side=TOP, fill=BOTH, expand=True)

        window_plot['fig1'] = fig1
        window_plot['ax1'] = ax1
        window_plot['canvas1'] = canvas1

        window_plot['fig2'] = fig2
        window_plot['ax2'] = ax2
        window_plot['ax3'] = ax3
        window_plot['canvas2'] = canvas2

    window_plot['ax1'].clear()
    window_plot['ax1'].set_xlabel('x', fontsize=10)
    window_plot['ax1'].set_ylabel('y', fontsize=10)
    window_plot['ax1'].yaxis.set_tick_params(labelsize=10)
    window_plot['ax1'].xaxis.set_tick_params(labelsize=10)

    if 'cax' in window_plot:
        window_plot['cax'].remove()

    path_map = glob.glob(window_params['path_fig1'])[0]
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
            if i == 0:
                continue

            count = np.count_nonzero(mask_labeled == i)
            if count > count_max:
                count_max = count
                index_max = i

        mask = np.where(mask_labeled == index_max, 1., 0.)
        mask[mask == 0.] = np.nan

        return mask

    mask = isolate_largest(data)
    data_isolated = np.where(np.isnan(mask), np.nan, data)
    clim = np.nanpercentile(data_isolated, (5, 95))

    var = var_mapselect.get()
    if var == 'Integrated flux':
        label_cbar = r'Int. flux (Jy beam$^{-1}$ km s$^{-1}$)'
    elif var == 'SGfit V.F.':
        label_cbar = r'LoS velocity (km s$^{-1}$)'
    elif var == 'SGfit VDISP':
        label_cbar = r'Velocity dispersion (km s$^{-1}$)'
    elif var == 'N-Gauss':
        clim = (1, _params['max_ngauss'])
        label_cbar = r'$N_\mathrm{gauss}$'
    elif var == 'SGfit peak S/N':
        clim = (0, np.nanmax(data))
        label_cbar = r'Peak S/N'

    if 'clim_{}'.format(var) in window_plot:
        clim = window_plot['clim_{}'.format(var)]
    else:
        window_plot['clim_{}'.format(var)] = clim

    img1 = window_plot['ax1'].imshow(data, interpolation='none', cmap='rainbow', clim=clim)

    update_entry(entry_climlo, clim[0])
    update_entry(entry_climhi, clim[1])

    window_plot['ax1'].invert_yaxis()
    _, window_plot['cax'] = create_colorbar(img1, cbarwidth=0.03, label=label_cbar, fontsize=10)
    window_plot['cax'].yaxis.set_tick_params(labelsize=10)

    window_plot['canvas1'].draw()

    window_plot['ax2'].clear()
    window_plot['ax3'].clear()
    window_plot['canvas2'].draw()

#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def read_ngfit_data(cube_fits=None, path_classified=None):
    if cube_fits:
        window_params['cube_fits'] = cube_fits
    if path_classified:
        window_params['path_classified'] = path_classified

    window_params['path_fig1'] = f"{window_params['path_classified']}/ngfit/ngfit.G*_1.1.fits"
    _params['cube'] = fits.getdata(window_params['cube_fits']) * window_params['tomJy']
    if len(_params['cube'].shape) > 3:
        _params['cube'] = _params['cube'][0, :, :]

    with fits.open(window_params['cube_fits'], 'update') as hdu:
        _ctype3 = hdu[0].header['CTYPE3']

    if _ctype3 != 'VOPT*':  # not optical
        _params['spectral_axis'] = SpectralCube.read(window_params['cube_fits']).with_spectral_unit(u.km/u.s, velocity_convention='radio').spectral_axis.value
    else:
        _params['spectral_axis'] = SpectralCube.read(window_params['cube_fits']).with_spectral_unit(u.km/u.s, velocity_convention='optical').spectral_axis.value

    _params['imsize'] = _params['cube'][0, :, :].shape

    n_gauss = _params['max_ngauss']
    amps = np.empty(n_gauss, dtype=object)
    vels = np.empty(n_gauss, dtype=object)
    disps = np.empty(n_gauss, dtype=object)
    ngfit_bgs = np.empty(n_gauss, dtype=object)
    ngfit_rms = np.empty(n_gauss, dtype=object)
    ngfit_sn = np.empty(n_gauss, dtype=object)

    sgfit_bg = fits.getdata(glob.glob(f"{window_params['path_classified']}/ngfit/ngfit.G{n_gauss}_1.3.fits")[0])

    for i in range(n_gauss):
        name_amp = glob.glob(f"{window_params['path_classified']}/ngfit/ngfit.G{n_gauss}_{i+1}.5.fits")[0]
        name_vel = glob.glob(f"{window_params['path_classified']}/ngfit/ngfit.G{n_gauss}_{i+1}.1.fits")[0]
        name_disp = glob.glob(f"{window_params['path_classified']}/ngfit/ngfit.G{n_gauss}_{i+1}.2.fits")[0]
        ngfit_bg_slice = glob.glob(f"{window_params['path_classified']}/ngfit/ngfit.G{n_gauss}_{i+1}.3.fits")[0]
        ngfit_rms_slice = glob.glob(f"{window_params['path_classified']}/ngfit/ngfit.G{n_gauss}_{i+1}.4.fits")[0]
        ngfit_sn_slice = glob.glob(f"{window_params['path_classified']}/ngfit/ngfit.G{n_gauss}_{i+1}.6.fits")[0]

        amps[i] = fits.getdata(name_amp)
        vels[i] = fits.getdata(name_vel)
        disps[i] = fits.getdata(name_disp)
        ngfit_bgs[i] = fits.getdata(ngfit_bg_slice)
        ngfit_rms[i] = fits.getdata(ngfit_rms_slice)
        ngfit_sn[i] = fits.getdata(ngfit_sn_slice)

    _params['amps'] = amps
    _params['vels'] = vels
    _params['disps'] = disps
    _params['bg'] = ngfit_bgs
    _params['rms'] = ngfit_rms
    _params['sn'] = ngfit_sn

    initialize_display()

#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def load_data():
    def browse_cube():
        cube_fits = filedialog.askopenfilename(title='Path to cube', filetypes=[('FITS file', '.fits .FITS')])
        if len(cube_fits) == 0:
            return

        if len(fits.getdata(cube_fits).shape) < 3 or len(SpectralCube.read(cube_fits).spectral_axis) == 1:
            messagebox.showerror("Error", "Cube should have at least three dimensions.")
            return

        update_entry(entry_cube_fits, cube_fits)

        possible_path_classified = glob.glob(os.path.dirname(cube_fits) + '/' + _params['_combdir'] + '.%d' % _classified_index)
        if len(possible_path_classified) == 1:
            browse_classified(possible_path_classified[0])
        elif len(possible_path_classified) > 1:
            browse_classified(initialdir=os.path.dirname(possible_path_classified[0]))

    def browse_classified(path_classified=None, initialdir=None):
        if path_classified is None:
            path_classified = filedialog.askdirectory(title='Path to classified directory', initialdir=initialdir)
            if len(path_classified) == 0:
                return

        ifexists = os.path.exists(path_classified)
        if ifexists == False:
            messagebox.showerror("Error", "No proper data found inside.")
            return

        update_entry(entry_path_classified, path_classified)

    def btncmd_toplv_apply():
        window_params['cube_fits'] = entry_cube_fits.get()
        window_params['path_classified'] = entry_path_classified.get()
        read_ngfit_data()
        window_plot['toplv'].destroy()

    def btncmd_toplv_cancel():
        toplv.destroy()

    toplv = Toplevel(root)
    frame_toplv1 = Frame(toplv)
    frame_toplv2 = Frame(toplv)

    create_label_and_entry(frame_toplv1, ['cube_fits', 'path_classified'], [], 0, 20, 20)

    btn_toplv_browsecube = Button(frame_toplv1, text='Browse', command=browse_cube)
    btn_toplv_browsecube.grid(row=0, column=2)

    btn_toplv_browseclassified = Button(frame_toplv1, text='Browse', command=browse_classified)
    btn_toplv_browseclassified.grid(row=1, column=2)

    ttk.Separator(frame_toplv2, orient='horizontal').pack(fill=BOTH)

    btn_toplv_apply = Button(frame_toplv2, text='Apply', command=btncmd_toplv_apply)
    btn_toplv_cancel = Button(frame_toplv2, text='Cancel', command=btncmd_toplv_cancel)
    btn_toplv_cancel.pack(side='right')
    btn_toplv_apply.pack(side='right')

    frame_toplv1.pack()
    frame_toplv2.pack(fill=BOTH)

    window_plot['toplv'] = toplv

#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def apply_map_selection(*args):
    var = var_mapselect.get()
    n_gauss = _params['max_ngauss']

    if var == 'Integrated flux':
        window_params['path_fig1'] = window_params['path_classified'] + '/sgfit/sgfit.G%d_1.0.fits' % n_gauss
    elif var == 'SGfit V.F.':
        window_params['path_fig1'] = window_params['path_classified'] + '/sgfit/sgfit.G%d_1.1.fits' % n_gauss
    elif var == 'SGfit VDISP':
        window_params['path_fig1'] = window_params['path_classified'] + '/sgfit/sgfit.G%d_1.2.fits' % n_gauss
    elif var == 'N-Gauss':
        window_params['path_fig1'] = window_params['path_classified'] + '/sgfit/sgfit.G%d_1.7.fits' % n_gauss
    elif var == 'SGfit peak S/N':
        window_params['path_fig1'] = window_params['path_classified'] + '/sgfit/sgfit.G%d_1.6.fits' % n_gauss

    initialize_display()

#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def apply_clim_values(*args):
    climlo = float(var_climlo.get())
    climhi = float(var_climhi.get())

    mapname = var_mapselect.get()
    window_plot['clim_{}'.format(mapname)] = [climlo, climhi]
    initialize_display()

#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def handle_climlo_entry_click(*args):
    entry_climlo.selection_range(0, END)

#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def handle_climhi_entry_click(*args):
    entry_climhi.selection_range(0, END)

#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def toggle_fix_cursor(event):
    window_plot['fix_cursor'] = (window_plot['fix_cursor'] + 1) % 2

#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def add_panel_label(ax, label, x=0.00, y=1.05, fontsize=6):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=fontsize, verticalalignment='bottom', horizontalalignment='left', clip_on=False)

#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def plot_profiles():
    try:
        n_gauss = _params['max_ngauss']
        x, y = window_params['cursor_xy']
        ax2, ax3 = window_plot['ax2'], window_plot['ax3']
        ax2.clear()
        ax3.clear()

        bg = _params['bg'][0][y, x] * window_params['tomJy']
        rms = _params['rms'][0][y, x] * window_params['tomJy']
        rms_axis = np.full_like(_params['spectral_axis'], rms)
        spectral_axis = _params['spectral_axis']
        cube = _params['cube'][:, y, x]

        ax2.step(spectral_axis, cube, linewidth=2.0)
        input_prof = np.full_like(cube, cube)
        total = np.zeros_like(spectral_axis)

        window_params['path_fig1'] = f"{window_params['path_classified']}/sgfit/sgfit.G%d_1.7.fits" % n_gauss
        ng_opt_fits = glob.glob(window_params['path_fig1'])[0]
        ng_opt = fits.getdata(ng_opt_fits)

        if not np.isnan(ng_opt[y, x]):
            for i in range(ng_opt[y, x].astype(int)):
                vel = _params['vels'][i][y, x]
                disp = _params['disps'][i][y, x]
                amp = _params['amps'][i][y, x]
                sn = _params['sn'][i][y, x]

                if np.any(np.isnan([vel, disp, amp])):
                    continue

                ploty = gauss_model(spectral_axis, amp, vel, disp) * window_params['tomJy']
                total += ploty
                ploty += bg

                label = f'G{i + 1:<2} (f: {amp*1000:>.1f} | x: {vel:>.1f} | s: {disp:>.1f} | S/N: {sn:>.1f})'
                ax2.plot(spectral_axis, ploty, label=label, color=colors[i], ls='-', alpha=0.5, linewidth=1.0)
                ploty -= bg

            ax2.legend(fontsize=10.0)
            add_panel_label(ax2, '(x, y | N-Gauss)=(%d, %d | %d)' % (x, y, ng_opt[y][x]), fontsize=10)

        add_panel_label(window_plot['ax3'], 'Residuals', 0.05, 0.85, fontsize=10)
        total += bg
        res = input_prof - total

        window_plot['ax3'].step(_params['spectral_axis'], res, color='orange', ls='-', linewidth=2.0, alpha=0.7)
        window_plot['ax2'].plot(_params['spectral_axis'], total, color='red', ls='--', linewidth=1.5, alpha=0.5)
        window_plot['ax3'].plot(_params['spectral_axis'], rms_axis, color='purple', ls='--', linewidth=1.0, alpha=0.7)
        window_plot['ax3'].plot(_params['spectral_axis'], -1 * rms_axis, color='purple', ls='--', linewidth=1.0, alpha=0.7)

        window_plot['ax2'].text(-0.12, -0, 'Flux density ({})'.format(window_params['unit_cube']), ha='center', va='center', transform=window_plot['ax2'].transAxes, rotation=90, fontsize=10)
        window_plot['ax3'].set_xlabel(r'Spectral axis (km$\,$s$^{-1}$)', fontsize=10)

        window_plot['ax2'].margins(x=0.02, y=0.15)
        window_plot['ax3'].margins(x=0.02, y=0.05)

        window_plot['ax2'].xaxis.set_tick_params(labelsize=10)
        window_plot['ax2'].yaxis.set_tick_params(labelsize=10)
        window_plot['ax3'].xaxis.set_tick_params(labelsize=10)
        window_plot['ax3'].yaxis.set_tick_params(labelsize=10)

        window_plot['canvas2'].draw()
    except IndexError:
        pass

#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def update_cursor_coords(event):
    if not window_plot['fix_cursor']:
        if event.inaxes:
            cursor_xy = (round(event.xdata), round(event.ydata))
            if window_params['cursor_xy'] != cursor_xy:
                window_params['cursor_xy'] = cursor_xy
                plot_profiles()

#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
def handle_zoom(event):
    ax = window_plot['ax1']
    canvas = window_plot['canvas1']
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    xdata, ydata = event.xdata, event.ydata
    base_scale = 2
    scale_factor = 1 / base_scale if event.button == 'up' else base_scale if event.button == 'down' else 1
    width = (xlim[1] - xlim[0]) * scale_factor
    height = (ylim[1] - ylim[0]) * scale_factor
    relx = (xlim[1] - xdata) / (xlim[1] - xlim[0])
    rely = (ylim[1] - ydata) / (ylim[1] - ylim[0])
    new_xlim = (np.max([xdata - width * (1 - relx), 0]), np.min([xdata + width * relx, _params['imsize'][1] - 1]))
    new_ylim = (np.max([ydata - height * (1 - rely), 0]), np.min([ydata + height * rely, _params['imsize'][0] - 1]))
    ax.set_xlim(*new_xlim)
    ax.set_ylim(*new_ylim)
    canvas.draw()

#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
# TK ROOT()
root = Tk()

root.title(title)

# Start with a standard size
root.geometry("1920x1080")

# Variable to keep track of fullscreen state
is_fullscreen = False

# Fonts
default_font = Font(family="TkDefaultFont", size=12)
label_font = Font(family="TkDefaultFont", size=12)
entry_font = Font(family="TkDefaultFont", size=12)

def adjust_fonts(new_size):
    default_font.configure(size=new_size)
    label_font.configure(size=new_size)
    entry_font.configure(size=new_size)
    for widget in [label_climlo, label_climhi, entry_climlo, entry_climhi, dropdown_mapselect]:
        widget.config(font=default_font)

def toggle_fullscreen(event=None):
    global is_fullscreen
    is_fullscreen = not is_fullscreen
    root.attributes("-fullscreen", is_fullscreen)
    adjust_layout()

def adjust_layout():
    if is_fullscreen:
        width = root.winfo_screenwidth()
        height = root.winfo_screenheight()
    else:
        width = root.winfo_width()
        height = root.winfo_height()

    # Adjust frame sizes
    frame_L.config(width=width * 0.55, height=height)
    frame_M.config(width=width * 0.01, height=height)
    frame_R.config(width=width * 0.40, height=height)
    frame_display.config(width=width * 0.55, height=height * 0.5)
    frame_line.config(width=width * 0.40, height=height * 0.5)

    # Adjust font sizes
    new_font_size = int(height / 40)
    adjust_fonts(new_font_size)

    # Redraw canvases
    window_plot['canvas1'].get_tk_widget().config(width=width * 0.55, height=height * 0.5)
    window_plot['canvas2'].get_tk_widget().config(width=width * 0.40, height=height * 0.5)
    window_plot['canvas1'].draw()
    window_plot['canvas2'].draw()

root.bind("<F11>", toggle_fullscreen)
root.bind("<Escape>", lambda event: root.attributes("-fullscreen", False))

menubar = Menu(root)

frame_master = Frame(root)
frame_L = Frame(frame_master, bg='white')
frame_M = Frame(frame_master, bg='white')
frame_R = Frame(frame_master, bg='white')

frame_display = Frame(frame_L, bg='white')
frame_display.pack(fill=BOTH, expand=True)

frame_LB = Frame(frame_L)

frame_LB_space = Frame(frame_LB)
frame_LB_space.pack(side='left')

frame_LB_climlo = Frame(frame_LB)
var_climlo = StringVar()
label_climlo = Label(frame_LB_climlo, text='clim_low', anchor='e', font=label_font)
entry_climlo = Entry(frame_LB_climlo, justify='right', textvariable=var_climlo, validate="focusout", validatecommand=apply_clim_values, font=entry_font)
entry_climlo.bind("<FocusIn>", handle_climlo_entry_click)
label_climlo.pack(side='left')
entry_climlo.pack(side='right')
frame_LB_climlo.pack(side='left')

frame_LB_climhi = Frame(frame_LB)
var_climhi = StringVar()
label_climhi = Label(frame_LB_climhi, text='clim_high', anchor='e', font=label_font)
entry_climhi = Entry(frame_LB_climhi, justify='right', textvariable=var_climhi, validate="focusout", validatecommand=apply_clim_values, font=entry_font)
entry_climhi.bind("<FocusIn>", handle_climhi_entry_click)
label_climhi.pack(side='left')
entry_climhi.pack(side='right')
frame_LB_climhi.pack(side='left')

OptionList = ['Integrated flux', 'SGfit V.F.', 'SGfit VDISP', 'N-Gauss', 'SGfit peak S/N']
var_mapselect = StringVar()
var_mapselect.set(OptionList[1])

dropdown_mapselect = OptionMenu(frame_LB, var_mapselect, *OptionList)
dropdown_mapselect.pack(side='right')
var_mapselect.trace("w", apply_map_selection)

frame_LB.pack(fill=BOTH, expand=True)

frame_line = Frame(frame_R, bg='white')
frame_line.pack(fill=BOTH, expand=True)

frame_L.pack(fill=BOTH, expand=True, side='left')
frame_M.pack(fill=BOTH, expand=True, side='left')
frame_R.pack(fill=BOTH, expand=True, side='right')
frame_master.pack(fill=BOTH, expand=True)

root.config(menu=menubar)
root.bind('f', toggle_fix_cursor)
root.bind('<Return>', apply_clim_values)

def close_application():
    root.destroy()
    sys.exit(0)

root.protocol("WM_DELETE_WINDOW", close_application)

def signal_handler(sig, frame):
    close_application()

signal.signal(signal.SIGINT, signal_handler)

if len(sys.argv) == 3:
    if not os.path.exists(sys.argv[1]):
        print(f":: WARNING: No '{sys.argv[1]}' exist..")
        sys.exit()

    configfile = sys.argv[1]
    _params = read_configfile(configfile)
    _classified_index = int(sys.argv[2])

    print(f"||--- Running baygaud_viewer.py with {configfile} {_classified_index} ---||")

else:
    print(":: Usage: running baygaud_viewer.py with baygaud_params.yaml file")
    print(":: > python3 baygaud_viewer.py [ARG1: _baygaud_params.yaml] [ARG2: _classified_index-N")
    sys.exit()

_cube_fits = f"{_params['wdir']}/{_params['input_datacube']}"
_path_classified = f"{_params['wdir']}/{_params['_combdir']}.{_classified_index}"
read_ngfit_data(cube_fits=_cube_fits, path_classified=_path_classified)

def main():
    root.mainloop()

if __name__ == '__main__':
    main()

#-- END OF SUB-ROUTINE____________________________________________________________#
