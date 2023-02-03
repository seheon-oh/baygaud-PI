title = 'baygaud-PI viewer'

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

from _baygaud_params import _params


dict_params = {'cursor_xy':(-1,-1), 'multiplier_cube':1000.0, 'unit_cube':r'mJy$\,$beam$^{-1}$', 'multiplier_spectral_axis':0.001}
dict_data = {}
dict_plot = {'fix_cursor':False}
plt.rcParams["hatch.linewidth"] = 4
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"

colors = ['tab:blue', 'tab:orange', 'tab:red', 'tab:green', 'tab:purple', 'tab:yellow', 'tab:black', 'tab:magenta', 'tab:cyan', \
        'tab:blue', 'tab:orange', 'tab:red', 'tab:green', 'tab:purple', 'tab:yellow', 'tab:black', 'tab:magenta', 'tab:cyan']

def gaussian(x, amp, vel, disp, bg):
    # amp is the peak flux which equals to A/{sqrt(2*pi) x std} where A is the area (integrated intensity)
    val = amp * np.exp(-np.power(x-vel, 2.) / (2 * np.power(disp, 2.))) + bg
    return val

def gaussian1(x, amp, vel, disp):
    # amp is the peak flux which equals to A/{sqrt(2*pi) x std} where A is the area (integrated intensity)
    val = amp * np.exp(-np.power(x-vel, 2.) / (2 * np.power(disp, 2.)))
    return val

def colorbar(img, spacing=0, cbarwidth=0.01, orientation='vertical', pos='right', label='', ticks=[0], fontsize=13):

    ax = img.axes
    fig = ax.figure
    if(orientation=='vertical'):
        if(pos=='right'):
            cax = fig.add_axes([ax.get_position().x1+spacing, ax.get_position().y0, cbarwidth, ax.get_position().height])
        elif(pos=='left'):
            cax = fig.add_axes([ax.get_position().x0-spacing-cbarwidth, ax.get_position().y0, cbarwidth, ax.get_position().height])
            cax.yaxis.set_ticks_position('left')
    elif(orientation=='horizontal'):
        if(pos=='top'):
            cax = fig.add_axes([ax.get_position().x0, ax.get_position().y1+spacing, ax.get_position().width, cbarwidth])
            cax.tick_params(axis='x', labelbottom=False, labeltop=True)

            # cax.xaxis.tick_top()
        elif(pos=='bottom'):
            cax = fig.add_axes([ax.get_position().x0, ax.get_position().y0-spacing-cbarwidth, ax.get_position().width, cbarwidth])
    
    if(len(ticks)!=1):
        cbar = plt.colorbar(img, cax=cax, orientation=orientation, ticks=ticks)
    else: cbar = plt.colorbar(img, cax=cax, orientation=orientation)
    cbar.set_label(label=label, fontsize=fontsize)
    return cbar, cax

def label_panel(ax, text, xpos=0.05, ypos=0.95, color='black', fontsize=10, inside_box=False, pad=5.0):
    # MAKES A LABEL ON GIVEN PANEL

    # PARAMETERS:
    # ax: matplotlib ax
    # text: (str) message to write
    # xpos: xpos of labelbox (relative corrdinates, 0 to 1)
    # ypos: ypos of labelbox (relative corrdinates, 0 to 1)
    # color: color of the text
    # fontsize=: fontsize of the text
    # inside_box = whether to write the text inside a box
    # pad: space between the text and the surrounding box

    # RETURNS:
    # Nothing

    if(inside_box==True):
        ax.text(xpos, ypos, text, transform=ax.transAxes,
            fontsize=fontsize, color=color, verticalalignment='top', 
            bbox=dict(facecolor='none', edgecolor=color, pad=pad))
    else:
        ax.text(xpos, ypos, text, transform=ax.transAxes,
            fontsize=fontsize, color=color, verticalalignment='top', 
            bbox=dict(facecolor='none', edgecolor='none', pad=pad))

def fillentry(entry, content):
    if(entry['state']=='readonly'):
        entry['state']='normal'
        entry.delete(0, "end")
        entry.insert(0, content)
        entry['state']='readonly'
    else:
        entry.delete(0, "end")
        entry.insert(0, content)

def makelabelentry(frame, array, title, startcol, widthlabel, widthentry):
    if(len(title)==0):
        title=array
    for i, content in enumerate(array):
        globals()['label_%s'%(content)] = Label(frame, text=title[i], width=widthlabel, anchor='e')
        globals()['label_%s'%(content)].grid(row=i+startcol, column=0, padx=5)
        globals()['entry_%s'%(content)] = Entry(frame, width=widthentry, justify='right')
        globals()['entry_%s'%(content)].grid(row=i+startcol, column=1)


def initdisplay():

    if 'fig1' not in dict_plot:

        fig1, ax1 = plt.subplots()#tight_layout=True)
        fig1.set_figwidth(1.5*500/fig1.dpi)
        fig1.set_figheight(1.5*460/fig1.dpi)
        fig1.subplots_adjust(left=0.1, right=0.90, top=0.99, bottom=0.05)

        canvas1 = FigureCanvasTkAgg(fig1, master=frame_display)   #DRAWING FIGURES ON GUI FRAME
        canvas1.draw()
        canvas1.get_tk_widget().pack(side=TOP)#, fill=BOTH, expand=True)
        fig1.canvas.mpl_connect('motion_notify_event', tracecursor)  #CONNECTING MOUSE CLICK ACTION
        fig1.canvas.mpl_connect('scroll_event', zoom)


        fig2, (ax2, ax3) = plt.subplots(nrows=2, sharex=True)
        fig2.set_figwidth(1.5*500/fig2.dpi)
        fig2.set_figheight(1.5*500/fig2.dpi)
        fig2.subplots_adjust(hspace=0, top=0.96, bottom=0.16)

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
        dict_plot['cax'].clear()

    path_map = glob.glob(dict_params['path_fig1'])[0]
    img1 = dict_plot['ax1'].imshow(fits.getdata(path_map), interpolation='none', cmap='rainbow')


    dict_plot['ax1'].invert_yaxis()
    #_,dict_plot['cax'] = colorbar(img1, cbarwidth=0.03, ticks=[0,1,2,3,4,5])
    _,dict_plot['cax'] = colorbar(img1, cbarwidth=0.03)

    dict_plot['canvas1'].draw()

    dict_plot['ax2'].clear()
    dict_plot['ax3'].clear()
    
    # dict_plot['ax2'].plot(dict_data['spectral_axis'], np.zeros_like(dict_data['spectral_axis']))
    # dict_plot['ax3'].plot(dict_data['spectral_axis'], np.zeros_like(dict_data['spectral_axis']))

    dict_plot['canvas2'].draw()

    # plt.close(fig)


def readdata(path_cube=None, path_classified=None):

    if(path_cube!=None):
        dict_params['path_cube'] = path_cube

    if(path_classified!=None):
        dict_params['path_classified'] = path_classified

    #dict_params['path_fig1'] = dict_params['path_classified']+'/G*g01/*5.fits'
    dict_params['path_fig1'] = dict_params['path_classified']+'/sgfit/sgfit.G*_1.1.fits'

    #dict_data['cube'] = fits.getdata(dict_params['path_cube'])
    dict_data['cube'] = fits.getdata(dict_params['path_cube'])*dict_params['multiplier_cube']
    dict_data['spectral_axis'] = SpectralCube.read(dict_params['path_cube']).spectral_axis.value*dict_params['multiplier_spectral_axis']

    #print(dict_data['spectral_axis'])
    #print(dict_data['cube'][:, 50, 50])

    dict_data['imsize'] = dict_data['cube'][0,:,:].shape

    #n_gauss = len(glob.glob(dict_params['path_classified']+"/G0*/"))

    # 0. A = peak * sqrt(2pi) * std : area
    # xxx.0.fits
    # xxx.0.e.fits
    #----------------------------------
    # 1. x0 : central velocity in km/s
    # xxx.1.fits
    # xxx.1.e.fits
    #----------------------------------
    # 2. std : velocity dispersion in km/s
    # xxx.2.fits
    # xxx.2.e.fits
    #----------------------------------
    # 3. bg : background in Jy
    # xxx.3.fits
    # xxx.3.e.fits
    #----------------------------------
    # 4. rms : rms in Jy
    # xxx.4.fits
    # xxx.4.e.fits
    #----------------------------------
    # 5. peak_flux in Jy
    # xxx.5.fits
    # xxx.5.e.fits
    #----------------------------------
    # 6. peak s/n
    # xxx.6.fits
    # xxx.6.e.fits
    #----------------------------------
    # 7. N-gauss
    # xxx.7.fits
    # xxx.7.e.fits

    n_gauss = _params['max_ngauss']

    amps   = np.empty(n_gauss, dtype=object)
    vels   = np.empty(n_gauss, dtype=object) #3
    disps  = np.empty(n_gauss, dtype=object) #2
    ngfit_bgs  = np.empty(n_gauss, dtype=object) #2
    ngfit_rms  = np.empty(n_gauss, dtype=object) #2
    ngfit_sn  = np.empty(n_gauss, dtype=object) #2

    #bg = fits.getdata(glob.glob(dict_params['path_classified']+'/G0*g01/*.0.fits')[0])

    # background from sgfit to avoid overlap!!
    sgfit_bg = fits.getdata(glob.glob(dict_params['path_classified']+'/ngfit/ngfit.G%d_1.3.fits' % n_gauss)[0])

    # useless
    data_noise = fits.getdata(glob.glob(dict_params['path_classified']+'/sgfit/sgfit.G%d_1.4.fits' % n_gauss)[0])
    dict_data['noise'] = data_noise

    for i in range(0, n_gauss):
        name_amp   = glob.glob(dict_params['path_classified']+'/ngfit/ngfit.G%d_%d.5.fits' % (n_gauss, i+1))[0]
        name_vel   = glob.glob(dict_params['path_classified']+'/ngfit/ngfit.G%d_%d.1.fits' % (n_gauss, i+1))[0]
        name_disp  = glob.glob(dict_params['path_classified']+'/ngfit/ngfit.G%d_%d.2.fits' % (n_gauss, i+1))[0]
        ngfit_bg_slice  = glob.glob(dict_params['path_classified']+'/ngfit/ngfit.G%d_%d.3.fits' % (n_gauss, i+1))[0]

        # rms
        ngfit_rms_slice  = glob.glob(dict_params['path_classified']+'/ngfit/ngfit.G%d_%d.4.fits' % (n_gauss, i+1))[0]
        # sn
        ngfit_sn_slice  = glob.glob(dict_params['path_classified']+'/ngfit/ngfit.G%d_%d.6.fits' % (n_gauss, i+1))[0]

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
        possible_path_classified = glob.glob(os.path.dirname(path_cube) + '/' + _params['_combdir'])
        #possible_path_classified = filedialog.askopenfilename(title='Path to baygaud_combined')
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
        readdata()

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
    if(var=='SGfit peak S/N'):
        dict_params['path_fig1'] = dict_params['path_classified']+'/sgfit/sgfit.G%d_1.6.fits' % n_gauss
    if(var=='N-Gauss'):
        dict_params['path_fig1'] = dict_params['path_classified']+'/sgfit/sgfit.G%d_1.7.fits' % n_gauss

    initdisplay()

def fix_cursor(event):
    dict_plot['fix_cursor'] = (dict_plot['fix_cursor']+1)%2






# ---------------------------------------------------
# ---------------------------------------------------
# ---------------------------------------------------
# START



root = Tk()

root.title(title)
# root.bind("<Return>", lambda x: updatedisplay())
root.resizable(False, False)


menubar = Menu(root)

menu_1 = Menu(menubar, tearoff=0)
menu_1.add_command(label="Load data", command=loaddata)

menu_2 = Menu(menubar, tearoff=0)
menu_2.add_command(label='TBU')

menubar.add_cascade(label="Load", menu=menu_1)
menubar.add_cascade(label="Option", menu=menu_2)

frame_master = Frame(root)
frame_L = Frame(frame_master, height=500, width=500, bg='white')
frame_M = Frame(frame_master, height=500, width=50, bg='white')
frame_R = Frame(frame_master, height=500, width=500, bg='white')

frame_display = Frame(frame_L, height=500, width=500, bg='white')
frame_display.pack()

frame_mapselect = Frame(frame_L)
OptionList = ['Integrated flux', 'SGfit V.F.', 'SGfit VDISP', 'N-Gauss', 'SGfit peak S/N']
var_mapselect = StringVar()
var_mapselect.set(OptionList[3])

dropdown_mapselect = OptionMenu(frame_mapselect, var_mapselect, *OptionList)
# dropdown_mapselect.config
dropdown_mapselect.pack(side='right')
var_mapselect.trace("w", apply_mapselect)
frame_mapselect.pack(fill=BOTH, expand=True)

frame_line = Frame(frame_R, width=500,height=500, bg='white')

frame_line.pack()

frame_L.pack(fill=BOTH, expand=True, side='left')
frame_M.pack(fill=BOTH, expand=True, side='left')
frame_R.pack(fill=BOTH, expand=True, side='right')
frame_master.pack(fill=BOTH, expand=True)



def drawplots():
    n_gauss = _params['max_ngauss']
    x, y = dict_params['cursor_xy']
    dict_plot['ax2'].clear()
    dict_plot['ax3'].clear()

    bg = dict_data['bg'][0][y,x]*dict_params['multiplier_cube']
    rms = dict_data['rms'][0][y,x]*dict_params['multiplier_cube']
    rms_axis = np.full_like(dict_data['spectral_axis'], rms)

    dict_plot['ax2'].step(dict_data['spectral_axis'], dict_data['cube'][:,y,x])
    subed = np.full_like(dict_data['cube'][:,y,x], dict_data['cube'][:,y,x])
    total = np.zeros_like(dict_data['spectral_axis'])

    dict_params['path_fig1'] = dict_params['path_classified']+'/sgfit/sgfit.G%d_1.7.fits' % n_gauss
    ng_opt_fits   = glob.glob(dict_params['path_classified']+'/sgfit/sgfit.G%d_1.7.fits' % n_gauss)[0]
    ng_opt = fits.getdata(ng_opt_fits)

    for i in range(0, ng_opt[y, x].astype(int)):
        vel  = dict_data['vels'][i][y,x]
        disp = dict_data['disps'][i][y,x]
        amp  = dict_data['amps'][i][y,x]
        sn  = dict_data['sn'][i][y,x]
        #_bg  = dict_data['bgs'][i][y,x]

        if(np.any(np.isnan([vel, disp, amp]))): continue

        #ploty = gaussian(dict_data['spectral_axis'], amp, vel, disp, _bg)*dict_params['multiplier_cube']
        ploty = gaussian1(dict_data['spectral_axis'], amp, vel, disp)*dict_params['multiplier_cube']
        # ploty+=dict_data['bg'][y,x]*bgsub

        total += ploty

        ploty += bg
        dict_plot['ax2'].plot(dict_data['spectral_axis'], ploty, label='G%d (S/N: %.2f)' % ((i+1), sn), color=colors[i], ls='-', alpha=0.5)
        dict_plot['ax2'].legend(loc='upper right')
        ploty -= bg

    
    total += bg
    subed -= total

    dict_plot['ax2'].plot(dict_data['spectral_axis'], total, color='red', ls='--', linewidth=3, alpha=0.5)

    #dict_plot['ax3'].fill_between(dict_data['spectral_axis'], subed, hatch=r'//', color='lightgray', edgecolor='white')
    dict_plot['ax3'].step(dict_data['spectral_axis'], subed, color='orange', ls='-', alpha=0.7)
    #dict_plot['ax3'].plot(dict_data['spectral_axis'], rms, color='blue', ls='--', alpha=0.7)
    dict_plot['ax3'].plot(dict_data['spectral_axis'], rms_axis, color='purple', ls='--', alpha=0.7)
    dict_plot['ax3'].plot(dict_data['spectral_axis'], -1*rms_axis, color='purple', ls='--', alpha=0.7)

    #print(ng_opt[300, 300])

    #label_panel(dict_plot['ax2'], '(x, y, N-Gauss)=({},{})'.format(x,y))
    label_panel(dict_plot['ax2'], '(x, y: N-Gauss)=(%d, %d: %d)' % (x, y, ng_opt[y, x]), fontsize=13)
    label_panel(dict_plot['ax3'], 'Residuals', fontsize=13)

    # dict_plot['ax2'].set_ylabel('Flux density ({})'.format(dict_params['unit_cube']))


    dict_plot['ax2'].text(-0.12, -0, 'Flux density ({})'.format(dict_params['unit_cube']), ha='center', va='center', transform = dict_plot['ax2'].transAxes, rotation=90, fontsize=16)
    dict_plot['ax3'].set_xlabel(r'Spectral axis (km$\,$s$^{-1}$)', fontsize=16)

    dict_plot['ax2'].margins(x=0.02, y=0.15)
    dict_plot['ax3'].margins(x=0.02, y=0.05)

    dict_plot['ax2'].xaxis.set_tick_params(labelsize=14)
    dict_plot['ax2'].yaxis.set_tick_params(labelsize=14)
    dict_plot['ax3'].xaxis.set_tick_params(labelsize=14)
    dict_plot['ax3'].yaxis.set_tick_params(labelsize=14)

    # canvas_line1 = FigureCanvasTkAgg(fig2, master=frame_line1)
    dict_plot['canvas2'].draw()




def tracecursor(event):
    if(dict_plot['fix_cursor']==False):
        # x,y=event.x, event.y
        if event.inaxes:
            # ax = event.inaxes
            cursor_xy = (round(event.xdata),round(event.ydata))
            # print(cursor_x, cursor_y)

            if(dict_params['cursor_xy']==cursor_xy[0] and dict_params['cursor_xy'][1]==cursor_xy[1]):
                return
            else:
                dict_params['cursor_xy']=cursor_xy
                drawplots()   

def zoom(event):
    cur_xlim = dict_plot['ax1'].get_xlim()
    cur_ylim = dict_plot['ax1'].get_ylim()

    xdata = event.xdata # get event x location
    ydata = event.ydata # get event y location

    base_scale = 2.

    if event.button == 'up':
        # deal with zoom in
        scale_factor = 1 / base_scale
    elif event.button == 'down':
        # deal with zoom out
        scale_factor = base_scale
    else:
        # deal with something that should never happen
        scale_factor = 1
        print(event.button)

    new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
    new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

    relx = (cur_xlim[1] - xdata)/(cur_xlim[1] - cur_xlim[0])
    rely = (cur_ylim[1] - ydata)/(cur_ylim[1] - cur_ylim[0])

    dict_plot['ax1'].set_xlim(np.max([xdata - new_width * (1-relx),0]), np.min([xdata + new_width * (relx),dict_data['imsize'][1]-1]))
    dict_plot['ax1'].set_ylim(np.max([ydata - new_height * (1-rely),0]), np.min([ydata + new_height * (rely),dict_data['imsize'][0]-1]))
    # ax.figure.canvas.draw()
    dict_plot['canvas1'].draw()

root.config(menu=menubar)
root.bind('f', fix_cursor)
# print(sys.argv)
if(len(sys.argv) == 1):
    #readdata(path_cube=sys.argv[1], path_classified=sys.argv[2])
    _path_cube = _params['wdir'] + '/' + _params['input_datacube']
    _path_classified = _params['wdir'] + '/' + _params['_combdir']
    readdata(path_cube= _path_cube, path_classified= _path_classified)

root.mainloop()

#-- END OF SUB-ROUTINE____________________________________________________________#

