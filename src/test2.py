#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
# run dynesty for each line profile
#@jit(nopython=True)
@ray.remote(num_cpus=1)
#@ray.remote
def baygaud_nested_sampling(_inputDataCube, _x, _peak_sn_map, _sn_int_map, _params, _is, _ie, i, _js, _je, _cube_mask_2d):

    _max_ngauss = _params['max_ngauss']
    _vel_min = _params['vel_min']
    _vel_max = _params['vel_max']
    _cdelt3 = _params['cdelt3']

    gfit_results = np.zeros(((_je-_js), _max_ngauss, 2*(2+3*_max_ngauss) + 7 + _max_ngauss), dtype=np.float32)
    _x_boundaries = np.full(2*_max_ngauss, fill_value=-1E11, dtype=np.float32)


    for j in range(0, _je -_js):

        _f_max = np.max(_inputDataCube[:,j+_js,i]) # peak flux : being used for normalization
        _f_min = np.min(_inputDataCube[:,j+_js,i]) # lowest flux : being used for normalization

        # prior arrays for the 1st single Gaussian fit
        gfit_priors_init = np.zeros(2*5, dtype=np.float32)

        gfit_priors_init = [0.0, 0.0, \
                            0.001, 0.001, 0.001, \
                            0.9, 0.6, \
                            0.999, 0.999, 1.0]

        if _cube_mask_2d[j+_js, i] <= 0 : # if masked, then skip : NOTE THE MASK VALUE SHOULD BE zero or negative.
            print("mask filtered: %d %d | peak S/N: %.1f :: S/N limit: %.1f | integrated S/N: %.1f :: S/N limit: %.1f :: f_min: %e :: f_max: %e" \
                % (i, j+_js, _peak_sn_map[j+_js, i], _params['peak_sn_limit'], _sn_int_map[j+_js, i], _params['int_sn_limit'], _f_min, _f_max))

            # save the current profile location
            for l in range(0, _max_ngauss):
                gfit_results[j][l][2*(3*_max_ngauss+2)+l] = _params['_rms_med'] # rms: the one derived from derive_rms_npoints_sgfit
                gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+0] = -1E11 # this is for sgfit: log-Z
                gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+1] = _is
                gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+2] = _ie
                gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+3] = _js
                gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+4] = _je
                gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+5] = i
                gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+6] = j + _js
            continue

        elif _sn_int_map[j+_js, i] < _params['int_sn_limit'] or _peak_sn_map[j+_js, i] < _params['peak_sn_limit'] \
            or np.isnan(_f_max) or np.isnan(_f_min) \
            or np.isinf(_f_min) or np.isinf(_f_min):

            print("low S/N: %d %d | peak S/N: %.1f :: S/N limit: %.1f | integrated S/N: %.1f :: S/N limit: %.1f :: f_min: %e :: f_max: %e" \
                % (i, j+_js, _peak_sn_map[j+_js, i], _params['peak_sn_limit'], _sn_int_map[j+_js, i], _params['int_sn_limit'], _f_min, _f_max))

            # save the current profile location
            for l in range(0, _max_ngauss):
                gfit_results[j][l][2*(3*_max_ngauss+2)+l] = _params['_rms_med'] # rms: the one derived from derive_rms_npoints_sgfit
                gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+0] = -1E11 # this is for sgfit: log-Z
                gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+1] = _is
                gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+2] = _ie
                gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+3] = _js
                gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+4] = _je
                gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+5] = i
                gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+6] = j + _js
            continue


        # for g1 fit results
        nparams_g1 = 3*1 + 2
        gfit_priors_init_g1 = np.zeros(nparams_g1, dtype=np.float32)

        for k in range(0, _max_ngauss):
            ngauss = k+1  # set the number of gaussian
            ndim = 3*ngauss + 2
            nparams = ndim

            if(ndim * (ndim + 1) // 2 > _params['nlive']):
                _params['nlive'] = 1 + ndim * (ndim + 1) // 2 # optimal minimum nlive

            print("processing: %d %d | peak s/n: %.1f | integrated s/n: %.1f | gauss-%d" % (i, j+_js, _peak_sn_map[j+_js, i], _sn_int_map[j+_js, i], ngauss))

            # go0
            if _params['_dynesty_class_'] == 'static':
                #---------------------------------------------------------
                # run dynesty 2.0.3
                sampler = NestedSampler(loglike_d, optimal_prior, ndim,
                    nlive=_params['nlive'],
                    update_interval=_params['update_interval'],
                    sample=_params['sample'],
                    bound=_params['bound'],
                    facc=_params['facc'],
                    fmove=_params['fmove'],
                    max_move=_params['max_move'],
                    logl_args=[(_inputDataCube[:,j+_js,i]-_f_min)/(_f_max-_f_min), _x, ngauss], ptform_args=[ngauss, gfit_priors_init])
                sampler.run_nested(dlogz=_params['dlogz'], maxiter=_params['maxiter'], maxcall=_params['maxcall'], print_progress=False)

            elif _params['_dynesty_class_'] == 'dynamic':
                #---------------------------------------------------------
                # run dynesty 2.0.3
                sampler = DynamicNestedSampler(loglike_d, optimal_prior, ndim,
                    nlive=_params['nlive'],
                    sample=_params['sample'],
                    #update_interval=_params['update_interval'],
                    bound=_params['bound'],
                    facc=_params['facc'],
                    fmove=_params['fmove'],
                    max_move=_params['max_move'],
                    logl_args=[(_inputDataCube[:,j+_js,i]-_f_min)/(_f_max-_f_min), _x, ngauss], ptform_args=[ngauss, gfit_priors_init])
                sampler.run_nested(dlogz_init=_params['dlogz'], maxiter_init=_params['maxiter'], maxcall_init=_params['maxcall'], print_progress=False)

            #---------------------------------------------------------
            _gfit_results_temp, _logz = get_dynesty_sampler_results(sampler)

            #---------------------------------------------------------
            # param1, param2, param3 ....param1-e, param2-e, param3-e
            #gfit_results[j][k][0~2*nparams] = _gfit_results_temp[0~2*nparams]
            gfit_results[j][k][:2*nparams] = _gfit_results_temp
            #---------------------------------------------------------

            #---------------------------------------------------------
            # derive rms of the profile given the current ngfit <---- || normalised (0~1) units ||
            _rms_ngfit = little_derive_rms_npoints(_inputDataCube, i, j+_js, _x, _f_min, _f_max, ngauss, _gfit_results_temp)
            #---------------------------------------------------------

            #---------------------------------------------------------
            if ngauss == 1: # check the peak s/n
                # load the normalised sgfit results : --> derive rms for s/n
                #_bg_sgfit = _gfit_results_temp[1]
                #_x_sgfit = _gfit_results_temp[2]
                #_std_sgfit = _gfit_results_temp[3]
                #_p_sgfit = _gfit_results_temp[4]
                # peak flux of the sgfit
                #_f_sgfit =_p_sgfit * exp( -0.5*((_x - _x_sgfit) / _std_sgfit)**2) + _bg_sgfit

# v1.0
#                #---------------------------------------------------------
#                # update gfit_priors_init
#                nparams_n = 3*(ngauss+1) + 2
#                gfit_priors_init = np.zeros(2*nparams_n, dtype=np.float32)
#                # lower bound : the parameters for the current ngaussian components
#                # nsigma_prior_range_gfit=3.0 (default)
#                gfit_priors_init[:nparams] = _gfit_results_temp[:nparams] - _params['nsigma_prior_range_gfit']*_gfit_results_temp[nparams:2*nparams]
#                # upper bound : the parameters for the current ngaussian components
#                gfit_priors_init[nparams+3:2*nparams+3] = _gfit_results_temp[:nparams] + _params['nsigma_prior_range_gfit']*_gfit_results_temp[nparams:2*nparams]
#                #---------------------------------------------------------



# v1.1
#                #---------------------------------------------------------
#                # update gfit_priors_init
#                nparams_n = 3*ngauss + 2
#
#                gfit_priors_init = np.zeros(2*nparams_n, dtype=np.float32)
#                # lower bound : the parameters for the current ngaussian components
#                # nsigma_prior_range_gfit=3.0 (default)
#                gfit_priors_init[:nparams] = _gfit_results_temp[:nparams] - _params['nsigma_prior_range_gfit']*_gfit_results_temp[nparams:2*nparams]
#                gfit_priors_init[:nparams] = np.where((gfit_priors_init[:nparams] < 0), 0, gfit_priors_init[:nparams])
#
#                # upper bound : the parameters for the current ngaussian components
#                gfit_priors_init[nparams+3:2*nparams+3] = _gfit_results_temp[:nparams] + _params['nsigma_prior_range_gfit']*_gfit_results_temp[nparams:2*nparams]
#                gfit_priors_init[nparams+3:2*nparams+3] = np.where((gfit_priors_init[nparams+3:2*nparams+3] > 1), 1, gfit_priors_init[nparams+3:2*nparams+3])
#                #---------------------------------------------------------

                #---------------------------------------------------------
                # v1.2
                # update gfit_priors_init
                #nparams_n = 3*ngauss + 2
                #gfit_priors_init_g1 = np.zeros(2*nparams_n, dtype=np.float32)
                gfit_priors_init_g1 = _gfit_results_temp[:nparams_g1]

                #---------------------------------------------------------


                # peak s/n : more accurate peak s/n from the first sgfit
                # <-- || normalised units (0~1)||
                _bg_sgfit = _gfit_results_temp[1]
                _p_sgfit = _gfit_results_temp[4] # bg already subtracted
                _peak_sn_sgfit = _p_sgfit/_rms_ngfit

                if _peak_sn_sgfit < _params['peak_sn_limit']: 
                    print("skip the rest of Gaussian fits: %d %d | rms:%.1f | bg:%.1f | peak:%.1f | peak_sgfit s/n: %.1f < %.1f" % (i, j+_js, _rms_ngfit, _bg_sgfit, _p_sgfit, _peak_sn_sgfit, _params['peak_sn_limit']))

                    # save the current profile location
                    for l in range(0, _max_ngauss):
                        if l == 0:
                        # for sgfit
                            gfit_results[j][l][2*(3*_max_ngauss+2)+l] = _rms_ngfit # this is for sgfit : rms
                            gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+0] = _logz # this is for sgfit: log-Z
                        else:
                            gfit_results[j][l][2*(3*_max_ngauss+2)+l] = 0 # put a blank value
                            gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+0] = -1E11 # put a blank value

                        gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+1] = _is
                        gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+2] = _ie
                        gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+3] = _js
                        gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+4] = _je
                        gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+5] = i
                        gfit_results[j][l][2*(3*_max_ngauss+2)+_max_ngauss+6] = _js + j

                    #________________________________________________________________________________________|
                    #|---------------------------------------------------------------------------------------|
                    # unit conversion
                    # sigma-flux --> data cube units
                    gfit_results[j][k][0] = gfit_results[j][k][0]*(_f_max - _f_min) # sigma-flux
                    # background --> data cube units
                    gfit_results[j][k][1] = gfit_results[j][k][1]*(_f_max - _f_min) + _f_min # bg-flux
                    gfit_results[j][k][6 + 3*k] = gfit_results[j][k][6 + 3*k]*(_f_max - _f_min) # bg-flux-e
                    _bg_flux = gfit_results[j][k][1]
        
                    for m in range(0, k+1):
                        #________________________________________________________________________________________|
                        # UNIT CONVERSION
                        #________________________________________________________________________________________|
                        # velocity, velocity-dispersion --> km/s
                        if _cdelt3 > 0: # if velocity axis is with increasing order
                            gfit_results[j][k][2 + 3*m] = gfit_results[j][k][2 + 3*m]*(_vel_max - _vel_min) + _vel_min # velocity
                        elif _cdelt3 < 0: # if velocity axis is with decreasing order
                            gfit_results[j][k][2 + 3*m] = gfit_results[j][k][2 + 3*m]*(_vel_min - _vel_max) + _vel_max # velocity

                        gfit_results[j][k][3 + 3*m] = gfit_results[j][k][3 + 3*m]*(_vel_max - _vel_min) # velocity-dispersion

                        #________________________________________________________________________________________|
                        # peak flux --> data cube units
                        gfit_results[j][k][4 + 3*m] = gfit_results[j][k][4 + 3*m]*(_f_max - _f_min) # peak flux
        
                        #________________________________________________________________________________________|
                        # velocity-e, velocity-dispersion-e --> km/s
                        gfit_results[j][k][7 + 3*(m+k)] = gfit_results[j][k][7 + 3*(m+k)]*(_vel_max - _vel_min) # velocity-e
                        gfit_results[j][k][8 + 3*(m+k)] = gfit_results[j][k][8 + 3*(m+k)]*(_vel_max - _vel_min) # velocity-dispersion-e
                        #________________________________________________________________________________________|
                        gfit_results[j][k][9 + 3*(m+k)] = gfit_results[j][k][9 + 3*(m+k)]*(_f_max - _f_min) # flux-e

                    # lastly put rms 
                    gfit_results[j][k][2*(3*_max_ngauss+2)+k] = gfit_results[j][k][2*(3*_max_ngauss+2)+k]*(_f_max - _f_min) # rms-(k+1)gfit
                    #________________________________________________________________________________________|
                    #|---------------------------------------------------------------------------------------|
                    continue
            #---------------------------------------------------------


# v.1.1
            # update optimal priors based on the current ngaussian fit results
            if ngauss < _max_ngauss:
                nparams_n = 3*(ngauss+1) + 2 # <-- ( + 1)
                # re-declare for the next ngauss fitting ( + 1)
                gfit_priors_init = np.zeros(2*nparams_n, dtype=np.float32)

                # load g1fit results
                g1fit_x = gfit_priors_init_g1[2]
                g1fit_std = gfit_priors_init_g1[3]
                g1fit_p = gfit_priors_init_g1[4]

                #---------------------------------------------------------
                # LOWER bound : the parameters for the current ngaussian components
                gfit_priors_init[:nparams_n] = 0.001

                # x 
                gfit_priors_init[2:nparams_n:3] = g1fit_x - g1fit_std * _params['x_prior_lowerbound_factor']
                # std
                gfit_priors_init[3:nparams_n:3] = _params['std_prior_lowerbound_factor'] * g1fit_std
                gfit_priors_init[3:nparams_n:3] = np.where( (gfit_priors_init[3:nparams_n:3]*(_vel_max - _vel_min) < (_cdelt3/1000.)), (_cdelt3/1000.)/(_vel_max - _vel_min), gfit_priors_init[3:nparams_n:3])
                # amp
                gfit_priors_init[4:nparams_n:3] = _params['p_prior_lowerbound_factor'] * g1fit_p


                #---------------------------------------------------------
                # UPPPER bound : the parameters for the current ngaussian components
                gfit_priors_init[nparams_n:2*nparams_n] = 0.999

                # x
                gfit_priors_init[nparams_n+2:2*nparams_n:3] = g1fit_x + g1fit_std * _params['x_prior_upperbound_factor']
                # std
                gfit_priors_init[nparams_n+3:2*nparams_n:3] = _params['std_prior_upperbound_factor'] * g1fit_std
                # amp
                gfit_priors_init[nparams_n+4:2*nparams_n:3] = _params['p_prior_upperbound_factor'] * g1fit_p

                gfit_priors_init = np.where(gfit_priors_init<0, 0, gfit_priors_init)
                gfit_priors_init = np.where(gfit_priors_init>1, 1, gfit_priors_init)




# v.1.0
#           # update optimal priors based on the current ngaussian fit results
#            if ngauss < _max_ngauss:
#                nparams_n = 3*ngauss + 2
#                gfit_priors_init = np.zeros(2*nparams_n, dtype=np.float32)
#                # lower bound : the parameters for the current ngaussian components
#                # nsigma_prior_range_gfit=3.0 (default)
#                gfit_priors_init[:nparams] = _gfit_results_temp[:nparams] - _params['nsigma_prior_range_gfit']*_gfit_results_temp[nparams:2*nparams]
#                # upper bound : the parameters for the current ngaussian components
#                gfit_priors_init[nparams+3:2*nparams+3] = _gfit_results_temp[:nparams] + _params['nsigma_prior_range_gfit']*_gfit_results_temp[nparams:2*nparams]
#    
#                # the parameters for the next gaussian component: based on the current ngaussians
#                _x_min_t = _gfit_results_temp[2:nparams:3].min()
#                _x_max_t = _gfit_results_temp[2:nparams:3].max()
#                _std_min_t = _gfit_results_temp[3:nparams:3].min()
#                _std_max_t = _gfit_results_temp[3:nparams:3].max()
#                _p_min_t = _gfit_results_temp[4:nparams:3].min()
#                _p_max_t = _gfit_results_temp[4:nparams:3].max()
#
#                # sigma_prior_lowerbound_factor=0.2 (default), sigma_prior_upperbound_factor=2.0 (default)
#                gfit_priors_init[0] = _params['sigma_prior_lowerbound_factor']*_gfit_results_temp[0]
#                gfit_priors_init[nparams_n] = _params['sigma_prior_upperbound_factor']*_gfit_results_temp[0]
#
#                # bg_prior_lowerbound_factor=0.2 (defaut), bg_prior_upperbound_factor=2.0 (default)
#                gfit_priors_init[1] = _params['bg_prior_lowerbound_factor']*_gfit_results_temp[1]
#                gfit_priors_init[nparams_n + 1] = _params['bg_prior_upperbound_factor']*_gfit_results_temp[1]
#
#                #print("x:", _x_min_t, _x_max_t, "std:", _std_min_t, _std_max_t, "p:",_p_min_t, _p_max_t)
#
#                #____________________________________________
#                # x: lower bound
#                if ngauss == 1:
#                    # x_lowerbound_gfit=0.1 (default), x_upperbound_gfit=0.9 (default)
#                    gfit_priors_init[nparams] = _params['x_lowerbound_gfit']
#                    gfit_priors_init[2*nparams+3] = _params['x_upperbound_gfit']
#                    #if gfit_priors_init[nparams] < 0 : gfit_priors_init[nparams] = 0
#                else:
#                    # x_prior_lowerbound_factor=5 (default), x_prior_upperbound_factor=5 (default)
#                    #gfit_priors_init[nparams] = _x_min_t - _params['x_prior_lowerbound_factor']*_std_max_t
#                    #gfit_priors_init[2*nparams+3] = _x_max_t + _params['x_prior_upperbound_factor']*_std_max_t
#                    #if gfit_priors_init[2*nparams+3] > 1 : gfit_priors_init[2*nparams+3] = 1
#
#
#                    gfit_priors_init[nparams] = 0.2
#                    gfit_priors_init[2*nparams+3] = 0.4
#
#
#                #____________________________________________
#                # std: lower bound
#                # std_prior_lowerbound_factor=0.1 (default)
#                gfit_priors_init[nparams+1] = _params['std_prior_lowerbound_factor']*_std_min_t
#                #gfit_priors_init[nparams+1] = 0.01
#                #if gfit_priors_init[nparams+1] < 0 : gfit_priors_init[nparams+1] = 0
#                # std: upper bound
#                # std_prior_upperbound_factor=3.0 (default)
#                gfit_priors_init[2*nparams+4] = _params['std_prior_upperbound_factor']*_std_max_t
#                #gfit_priors_init[2*nparams+4] = 0.9
#                #if gfit_priors_init[2*nparams+4] > 1 : gfit_priors_init[2*nparams+4] = 1
#
#                gfit_priors_init[nparams+1] = 0.01
#                gfit_priors_init[2*nparams+4] = 0.2
#    
#                #____________________________________________
#                # p: lower bound
#                # p_prior_lowerbound_factor=0.05 (default)
#                gfit_priors_init[nparams+2] = _params['p_prior_lowerbound_factor']*_p_max_t # 5% of the maxium flux
#                # p: upper bound
#                # p_prior_upperbound_factor=1.0 (default)
#                gfit_priors_init[2*nparams+5] = _params['p_prior_upperbound_factor']*_p_max_t
#
#                gfit_priors_init[nparams+2] = 0. 
#                gfit_priors_init[2*nparams+5] = 1.0
#
#
#                gfit_priors_init = np.where(gfit_priors_init<0, 0, gfit_priors_init)
#                gfit_priors_init = np.where(gfit_priors_init>1, 1, gfit_priors_init)












            gfit_results[j][k][2*(3*_max_ngauss+2)+k] = _rms_ngfit # rms_(k+1)gfit
            gfit_results[j][k][2*(3*_max_ngauss+2)+_max_ngauss+0] = _logz
            gfit_results[j][k][2*(3*_max_ngauss+2)+_max_ngauss+1] = _is
            gfit_results[j][k][2*(3*_max_ngauss+2)+_max_ngauss+2] = _ie
            gfit_results[j][k][2*(3*_max_ngauss+2)+_max_ngauss+3] = _js
            gfit_results[j][k][2*(3*_max_ngauss+2)+_max_ngauss+4] = _je
            gfit_results[j][k][2*(3*_max_ngauss+2)+_max_ngauss+5] = i
            gfit_results[j][k][2*(3*_max_ngauss+2)+_max_ngauss+6] = _js + j
            #print(gfit_results[j][k])

            #|-----------------------------------------|
            #|-----------------------------------------|
            # example: 3 gaussians : bg + 3 * (x, std, peak) 
            #  ______________________________________  |
            # |_G1___________________________________| |
            # |_000000000000000000000000000000000000_| |
            # |_000000000000000000000000000000000000_| |
            # |_G1-rms : 0 : 0 : log-Z : xs-xe-ys-ye_| |
            #  ______________________________________  |
            # |_G1___________________________________| |
            # |_G2___________________________________| |
            # |_000000000000000000000000000000000000_| |
            # |_0 : G2-rms : 0 : log-Z : xs-xe-ys-ye_| |
            #  ______________________________________  |
            # |_G1___________________________________| |
            # |_G2___________________________________| |
            # |_G3___________________________________| |
            # |_0 : 0 : G3-rms : log-Z : xs-xe-ys-ye_| |

            #|-----------------------------------------|
            #gfit_results[j][k][0] : dist-sig
            #gfit_results[j][k][1] : bg
            #gfit_results[j][k][2] : g1-x --> *(vel_max-vel_min) + vel_min
            #gfit_results[j][k][3] : g1-s --> *(vel_max-vel_min)
            #gfit_results[j][k][4] : g1-p
            #gfit_results[j][k][5] : g2-x --> *(vel_max-vel_min) + vel_min
            #gfit_results[j][k][6] : g2-s --> *(vel_max-vel_min)
            #gfit_results[j][k][7] : g2-p
            #gfit_results[j][k][8] : g3-x --> *(vel_max-vel_min) + vel_min
            #gfit_results[j][k][9] : g3-s --> *(vel_max-vel_min)
            #gfit_results[j][k][10] : g3-p

            #gfit_results[j][k][11] : dist-sig-e
            #gfit_results[j][k][12] : bg-e
            #gfit_results[j][k][13] : g1-x-e --> *(vel_max-vel_min)
            #gfit_results[j][k][14] : g1-s-e --> *(vel_max-vel_min)
            #gfit_results[j][k][15] : g1-p-e
            #gfit_results[j][k][16] : g2-x-e --> *(vel_max-vel_min)
            #gfit_results[j][k][17] : g2-s-e --> *(vel_max-vel_min)
            #gfit_results[j][k][18] : g2-p-e
            #gfit_results[j][k][19] : g3-x-e --> *(vel_max-vel_min)
            #gfit_results[j][k][20] : g3-s-e --> *(vel_max-vel_min)
            #gfit_results[j][k][21] : g3-p-e --> *(f_max-bg_flux)

            #gfit_results[j][k][22] : g1-rms --> *(f_max-bg_flux) : the bg rms for the case with single gaussian fitting
            #gfit_results[j][k][23] : g2-rms --> *(f_max-bg_flux) : the bg rms for the case with double gaussian fitting
            #gfit_results[j][k][24] : g3-rms --> *(f_max-bg_flux) : the bg rms for the case with triple gaussian fitting

            #gfit_results[j][k][25] : log-Z : log-evidence : log-marginalization likelihood

            #gfit_results[j][k][26] : xs
            #gfit_results[j][k][27] : xe
            #gfit_results[j][k][28] : ys
            #gfit_results[j][k][29] : ye
            #gfit_results[j][k][30] : x
            #gfit_results[j][k][31] : y
            #|-----------------------------------------|

            #________________________________________________________________________________________|
            #|---------------------------------------------------------------------------------------|
            # UNIT CONVERSION
            # sigma-flux --> data cube units
            gfit_results[j][k][0] = gfit_results[j][k][0]*(_f_max - _f_min) # sigma-flux
            # background --> data cube units
            gfit_results[j][k][1] = gfit_results[j][k][1]*(_f_max - _f_min) + _f_min # bg-flux
            gfit_results[j][k][6 + 3*k] = gfit_results[j][k][6 + 3*k]*(_f_max - _f_min) # bg-flux-e
            _bg_flux = gfit_results[j][k][1]

            for m in range(0, k+1):
                #________________________________________________________________________________________|
                # UNIT CONVERSION

                #________________________________________________________________________________________|
                # velocity, velocity-dispersion --> km/s
                if _cdelt3 > 0: # if velocity axis is with increasing order
                    gfit_results[j][k][2 + 3*m] = gfit_results[j][k][2 + 3*m]*(_vel_max - _vel_min) + _vel_min # velocity
                elif _cdelt3 < 0: # if velocity axis is with decreasing order
                    gfit_results[j][k][2 + 3*m] = gfit_results[j][k][2 + 3*m]*(_vel_min - _vel_max) + _vel_max # velocity

                gfit_results[j][k][3 + 3*m] = gfit_results[j][k][3 + 3*m]*(_vel_max - _vel_min) # velocity-dispersion

                #________________________________________________________________________________________|
                # peak flux --> data cube units : (_f_max - _f_min) should be used for scaling 
                gfit_results[j][k][4 + 3*m] = gfit_results[j][k][4 + 3*m]*(_f_max - _f_min) # peak flux

                #________________________________________________________________________________________|
                # velocity-e, velocity-dispersion-e --> km/s
                gfit_results[j][k][7 + 3*(m+k)] = gfit_results[j][k][7 + 3*(m+k)]*(_vel_max - _vel_min) # velocity-e
                gfit_results[j][k][8 + 3*(m+k)] = gfit_results[j][k][8 + 3*(m+k)]*(_vel_max - _vel_min) # velocity-dispersion-e

                gfit_results[j][k][9 + 3*(m+k)] = gfit_results[j][k][9 + 3*(m+k)]*(_f_max - _f_min) # flux-e

            # lastly put rms 
            #________________________________________________________________________________________|
            gfit_results[j][k][2*(3*_max_ngauss+2)+k] = gfit_results[j][k][2*(3*_max_ngauss+2)+k]*(_f_max - _f_min) # rms-(k+1)gfit
            #________________________________________________________________________________________|
            #|---------------------------------------------------------------------------------------|


#    #________________________________________________________________________________________|
#    #________________________________________________________________________________________|
#    #________________________________________________________________________________________|
#    # ++++++++++++++++++++
#    # PROFILE CHECK
#    velocity = np.linspace(_vel_min, _vel_max, 1000)
#    plt.figure(figsize=(10, 6))
#
#    colors = ['black', 'green', 'blue', 'red']
#    ngauss_total = 4  # 사용할 Gaussian 함수의 총 개수
#    block = gfit_results[0]
#
#
#    for tt in range(1, ngauss_total+1):
#        for bb in range(tt-1, tt):
#            gaussian_params = block[bb]
#            background_value = gaussian_params[1]  # 배경값 추출
#            total_flux = np.full_like(velocity, background_value)  # 전체 flux를 background value로 초기화
#
#            for m in range(tt-1, tt):
#                nn = 2 + (m+1) * 3
#                for i in range(2, nn, 3):  # IndexError를 방지하기 위한 범위 조정
#                    mean = gaussian_params[i + 0]
#                    std_dev = gaussian_params[i + 1]
#                    amplitude = gaussian_params[i + 2]
#
#                    print("block:", bb, "gauss-", m,":", i, mean, std_dev, amplitude)
#                    # std_dev가 0이거나 amplitude가 0인 경우 해당 Gaussian 함수는 계산하지 않음
#                    if std_dev == 0 or amplitude == 0:
#                        continue
#
#                    # "divide by zero" 경고를 방지하기 위한 조건 추가
#                    gaussian_flux = amplitude * np.exp(-0.5 * ((velocity - mean) / std_dev) ** 2) if std_dev != 0 else 0
#                    total_flux += gaussian_flux
#
#                    plt.plot(velocity, gaussian_flux, label=f'Block {bb}', color=colors[bb])
#                plt.plot(velocity, total_flux, label=f'Block {bb}', color=colors[bb])
#
#    plt.title('Gaussian Fits Overplot')
#    plt.xlabel('Velocity (km/s)')
#    plt.ylabel('Flux')
#    plt.legend()
#    plt.show()
#    sys.exit()
#    #________________________________________________________________________________________|
#    #________________________________________________________________________________________|
#    #________________________________________________________________________________________|

    return gfit_results

    # Plot a summary of the run.
#    rfig, raxes = dyplot.runplot(sampler.results)
#    rfig.savefig("r.pdf")
    
    # Plot traces and 1-D marginalized posteriors.
#    tfig, taxes = dyplot.traceplot(sampler.results)
#    tfig.savefig("t.pdf")
    
    # Plot the 2-D marginalized posteriors.
    #cfig, caxes = dyplot.cornerplot(sampler.results)
    #cfig.savefig("c.pdf")
#-- END OF SUB-ROUTINE____________________________________________________________#