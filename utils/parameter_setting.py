import numpy as np

def parameters_set1():

    net_params = {'n_filters': 64, 'padding': 1, 'kernel_size': 3, 'sig_pred': True, 'local_flag': False,
                      'dilation_flag': 1,'psf_pred':False}

    train_params = {'lr': 8e-4, 'lr_decay': 0.88, 'w_decay': 0.1, 'max_iters': 80, 'interval': 500,
                    'ph_filt_thre': 160 * 5, 'P_locs_cse': True, 'results_path': "F://users\ltd\Dilatedloc_finall\Results_1212_valid_as_n//",'netresume':False}

    camera_params = {'camera': 'sCMOS', 'em_gain': 1.0, 'surv_p': 0.5, 'margin_empty': 0.1, 'backg': 200,
                     'qe': 1, 'spurious_c': 0.000, 'sig_read': 0.0001, 'e_per_adu': 1, 'baseline': 0,
                     'perlin_noise': False, 'pn_factor': 0.2, 'pn_res': 64}

    psf_params = {'ph_scale': 20000, 'Npixels': 51, 'z_scale': 700, 'initial_obj_stage': -800, 'pixel_size_x': 100,
                  'pixel_size_y': 100, 'Npupil': 64, 'NA': 1.49, 'refmed': 1.518, 'refcov': 1.518, 'refimm': 1.518,
                  'wavelength': 660,  'robust_training': False}

    data_params = {'nvalid':100, 'batch_size': 10, 'num_particles': 10, 'size_x': 128,
                   'size_y': 128, 'min_ph': 0.1, 'valid_data_path': "D://users//feiyue\Dilatedloc_finall\ValidingLocations_demo/", 'dataresume':False}

    eval_params = {'threshold': 0.3, 'min_int': 0.1, 'limited_x': [0, 14080], 'limited_y': [0, 14080], 'tolerance': 200,
                   'tolerance_ax': 500, 'batch_size':10,  'candi_thre': 0.3}

    infer_params = {'win_size': 128, 'padding': 0, "img_path": "D:/users/feiyue/simulate_data/Astigmatism_margin/Astigmatism_High_SNR_density=2_margin=32.tif",
                    "result_name": 'result0105-2.csv',
                    'net_path':  "S:/Users/Fei_Yue/DilatedLoc-main_v5/NewResults/Astigmatism_DilatedLoc_Unet_normal_density/High_SNR/best_checkpoint.pth.tar", # "D:/users/feiyue/Training_model/DilatedLoc_Unet_depthwise/Astigmatism/best_checkpoint.pth.tar",
                    'gt_path': "D:/users/feiyue/simulate_data/Astigmatism_margin/activations_Astigmatism_density=2_margin=32.csv",
                    'local_flag': False}
    #S://Users\Fei_Yue\DilatedLoc-main_v5\Training_model\DMO-Tetrapod\High_SNR\checkpoint.pth.tar
    settings = {'net_params': net_params, 'train_params': train_params, 'camera_params': camera_params,
                'psf_params': psf_params, 'data_params': data_params, 'eval_params': eval_params, 'infer_params': infer_params}


    return settings

def parameters_set1_NPC():

    net_params = {'n_filters': 64, 'padding': 1, 'kernel_size': 3, 'sig_pred': True, 'local_flag': True,
                      'dilation_flag': 1,'psf_pred':False}

    train_params = {'lr': 8e-4, 'lr_decay': 0.88, 'w_decay': 0.1, 'max_iters': 80, 'interval': 500,
                    'ph_filt_thre': 160 * 5, 'P_locs_cse': True, 'results_path': "S:/Users/Fei_Yue/DilatedLoc-main_v5/NewResults/Tetrapod_6um_NPC_DilatedLoc_Unet_local_context_correct/",
                    'netresume':False}

    camera_params = {'camera': 'sCMOS', 'em_gain': 1.0, 'surv_p': 0.5, 'margin_empty': 0.1, 'backg': 130.52,
                     'qe': 0.95, 'spurious_c': 0.002, 'sig_read': 1.535, 'e_per_adu': 0.7471, 'baseline': 100.0,
                     'perlin_noise': True, 'pn_factor': 0.2, 'pn_res': 64}

    psf_params = {'ph_scale': 6000, 'Npixels': 51, 'z_scale': 3000, 'initial_obj_stage': -800, 'pixel_size_x': 110,
                  'pixel_size_y': 110, 'Npupil': 64, 'NA': 1.35, 'refmed': 1.406, 'refcov': 1.525, 'refimm': 1.406,
                  'wavelength': 680,  'robust_training': False}

    data_params = {'nvalid': 10000, 'batch_size': 10, 'num_particles': 6, 'size_x': 128,
                   'size_y': 128, 'min_ph': 0.1, 'valid_data_path': "S:/Users/Fei_Yue/DilatedLoc-main_v5/NewResults/Tetrapod_6um_NPC_20230812_v1/High_SNR/",
                   'dataresume':False}

    eval_params = {'threshold': 0.3, 'min_int': 0.1, 'limited_x': [0, 14080], 'limited_y': [0, 14080], 'tolerance': 200,
                   'tolerance_ax': 500, 'batch_size': 40, 'nms_cont': False, 'candi_thre': 0.3}

    infer_params = {'win_size': 296, 'padding': 0, "img_path": "Q:/FY/NPC_NUP96_SNAP647_DMO_6um_defoucs_0_20ms_3/NPC_NUP96_SNAP647_DMO_6um_defoucs_0_20ms_3_MMStack_Pos0.ome.tif",
                    "result_name": 'result_NPC_local_context_test.csv',
                    'net_path':  "S:/Users/Fei_Yue/DilatedLoc-main_v5/NewResults/Tetrapod_6um_NPC_DilatedLoc_Unet_local_context/best_checkpoint.pth.tar",
                    'gt_path': "",
                    'local_flag': True}
    #S://Users\Fei_Yue\DilatedLoc-main_v5\Training_model\DMO-Tetrapod\High_SNR\checkpoint.pth.tar
    settings = {'net_params': net_params, 'train_params': train_params, 'camera_params': camera_params,
                'psf_params': psf_params, 'data_params': data_params, 'eval_params': eval_params, 'infer_params': infer_params}


    return settings
# demo1, factor=69.38, offset=50.54, backg=49.91
# def parameters_set(type, factor=69.38, offset=50.54, backg=49.91):
#
#     if type =='3D':
#         net_params = {'D':121, 'dilation_flag': 1, 'scaling_factor': 800.0}
#     else:
#         net_params = {'n_filters': 48, 'padding': 1, 'kernel_size': 3, 'factor': factor, 'offset': offset, 'sig_pred': True,
#                       'dilation_flag': 1, 'psf_pred': True}  # 原demo1的psf_pred给的是True
#
#     # training parameters
#     # train_params = {'lr': 6e-4, 'lr_decay': 0.85, 'w_decay': 0.1, 'max_iters':80, 'interval':500,
#     #                 'batch_size': 1, 'ph_filt_thre': 160 * 5, 'P_locs_cse': True}
#     train_params = {'lr': 6e-4, 'lr_decay': 0.9, 'w_decay': 0.1, 'max_iters': 80, 'interval': 500,
#                     'batch_size': 10, 'ph_filt_thre': 500, 'P_locs_cse': True}
#     # lr_decay=0.9, ph_filt_thre=500
#
#     # camera_params = {'camera': 'sCMOS', 'em_gain': 1.0, 'surv_p': 0.5, 'margin_empty': 0.05,'backg':180,
#     #                  'qe': 0.95, 'spurious_c': 0.002, 'sig_read': 1.6, 'e_per_adu': 0.5, 'baseline': 100,
#     #                  'perlin_noise': True, 'pn_factor': 0.2, 'pn_res': 64}
#     camera_params = {'camera': 'sCMOS', 'em_gain': 1.0, 'surv_p': 0.5, 'margin_empty': 0.05, 'backg': backg,
#                      'qe': 1, 'spurious_c': 0.000, 'sig_read': 0.0001, 'e_per_adu': 1, 'baseline': 0,
#                      'perlin_noise': False, 'pn_factor': 0.2, 'pn_res': 64}
#
#     # psf_params = {'ph_scale': 8000, 'Npixels': 51, 'z_scale': 1000, 'objstage0': -1000, 'pixel_size_x': 110,
#     #               'pixel_size_y': 110, 'Npupil': 64, 'NA': 1.5, 'refmed': 1.518, 'refcov': 1.524, 'refimm': 1.518,
#     #               'wavelength': 670, 'initial_obj_stage': -1000, 'bg': 160,'robust_training':True}
#
#     psf_params = {'ph_scale': 7000, 'Npixels': 51, 'z_scale': 700, 'objstage0': -800, 'pixel_size_x': 100,
#                   'pixel_size_y': 100, 'Npupil': 64, 'NA': 1.49, 'refmed': 1.518, 'refcov': 1.518, 'refimm': 1.518,
#                   'wavelength': 660, 'initial_obj_stage': -800, 'bg': backg, 'robust_training': False}
#     # Npupil是什么？objstage0是什么？bg是什么？
#
#     # data_params = {'nvalid': 2000, 'batch_size': train_params['batch_size'], 'num_particles': 12, 'size_x': 128,
#     #                'size_y': 128, 'z_scale': 1000, 'min_ph': 0.1,'valid_data_path': "./ValidingLocations_demo/",
#     #                 'results_path': "./Results_demoDMO/", 'dataresume':True,'netresume':False}
#     data_params = {'nvalid': 30, 'batch_size': train_params['batch_size'], 'num_particles': 12, 'size_x': 128,
#                    'size_y': 128, 'z_scale': 700, 'min_ph': 0.428, 'valid_data_path': "./ValidingLocations_demo/",
#                    'results_path': "./Results_demo0320test/", 'dataresume': False, 'netresume': False}
#     # 原demo1 nvalid(eval_imgs_number)=30, 原demo每次valid的数据相同
#
#     # eval_params = {'threshold': 0.3, 'min_int': 0.1, 'limited_x': [0,12800], 'limited_y': [0,12800], 'tolerance':200,
#     #                 'tolerance_ax': 500, 'z_scale': psf_params['z_scale'], 'ph_scale': psf_params['ph_scale'],
#     #                'batch_size':train_params['batch_size'],'nms_cont':False,'candi_thre':0.3}
#     eval_params = {'threshold': 0.3, 'min_int': 0.428, 'limited_x': [0, data_params['size_x']*psf_params['pixel_size_x']], 'limited_y': [0, data_params['size_y']*psf_params['pixel_size_y']],
#                    'tolerance': 200, 'pixel_size_x': psf_params['pixel_size_x'], 'pixel_size_y': psf_params['pixel_size_y'],
#                    'tolerance_ax': 500, 'z_scale': psf_params['z_scale'], 'ph_scale': psf_params['ph_scale'],
#                    'batch_size': train_params['batch_size'], 'nms_cont': False, 'candi_thre': 0.3}
#     # nms_thre=0.7, threshold是什么？
#
#     infer_params = {'win_size': 128, 'batch_size': 10, 'padding': 0,
#                     "img_path": 'W://ltd//anti//imgs_sim_128_2000_0315.tif', "result_name": 'result0317.csv',
#                     'net_path': "G://Dilatedloc//INFER//2023-03-16-checkpoint.pth.tar", 'gt_path': "gt.csv"}
#
#
#     settings = {'net_params': net_params, 'train_params': train_params, 'camera_params': camera_params,
#                 'psf_params': psf_params, 'data_params': data_params, 'eval_params': eval_params,
#                 'infer_params': infer_params}
#
#     return settings

# demo2, factor=201.8, offset=189.22, backg=186.64
# def parameters_set(type, factor=69.38, offset=50.54, backg=49.91):
#
#     if type =='3D':
#         net_params = {'D':121, 'dilation_flag': 1, 'scaling_factor': 800.0}
#     else:
#         net_params = {'n_filters': 48, 'padding': 1, 'kernel_size': 3, 'factor': factor, 'offset': offset, 'sig_pred': True,
#                       'dilation_flag': 1, 'psf_pred': True}  # 原demo1的psf_pred给的是True
#
#     # training parameters
#     # train_params = {'lr': 6e-4, 'lr_decay': 0.85, 'w_decay': 0.1, 'max_iters':80, 'interval':500,
#     #                 'batch_size': 1, 'ph_filt_thre': 160 * 5, 'P_locs_cse': True}
#     train_params = {'lr': 6e-4, 'lr_decay': 0.9, 'w_decay': 0.1, 'max_iters': 80, 'interval': 500,
#                     'batch_size': 10, 'ph_filt_thre': 800, 'P_locs_cse': True}
#     # lr_decay=0.9, ph_filt_thre=500
#
#     # camera_params = {'camera': 'sCMOS', 'em_gain': 1.0, 'surv_p': 0.5, 'margin_empty': 0.05,'backg':180,
#     #                  'qe': 0.95, 'spurious_c': 0.002, 'sig_read': 1.6, 'e_per_adu': 0.5, 'baseline': 100,
#     #                  'perlin_noise': True, 'pn_factor': 0.2, 'pn_res': 64}
#     camera_params = {'camera': 'sCMOS', 'em_gain': 1.0, 'surv_p': 0.5, 'margin_empty': 0.05, 'backg': backg,
#                      'qe': 0.95, 'spurious_c': 0.002, 'sig_read': 1.6, 'e_per_adu': 0.5, 'baseline': 100,
#                      'perlin_noise': False, 'pn_factor': 0.2, 'pn_res': 64}
#
#     # psf_params = {'ph_scale': 8000, 'Npixels': 51, 'z_scale': 1000, 'objstage0': -1000, 'pixel_size_x': 110,
#     #               'pixel_size_y': 110, 'Npupil': 64, 'NA': 1.5, 'refmed': 1.518, 'refcov': 1.524, 'refimm': 1.518,
#     #               'wavelength': 670, 'initial_obj_stage': -1000, 'bg': 160,'robust_training':True}
#
#     psf_params = {'ph_scale': 8000, 'Npixels': 51, 'z_scale': 700, 'objstage0': -800, 'pixel_size_x': 110,
#                   'pixel_size_y': 110, 'Npupil': 64, 'NA': 1.5, 'refmed': 1.406, 'refcov': 1.524, 'refimm': 1.518,
#                   'wavelength': 670, 'initial_obj_stage': -800, 'bg': backg, 'robust_training': False}
#     # Npupil是什么？objstage0是什么？bg是什么？
#
#     # data_params = {'nvalid': 2000, 'batch_size': train_params['batch_size'], 'num_particles': 12, 'size_x': 128,
#     #                'size_y': 128, 'z_scale': 1000, 'min_ph': 0.1,'valid_data_path': "./ValidingLocations_demo/",
#     #                 'results_path': "./Results_demoDMO/", 'dataresume':True,'netresume':False}
#     data_params = {'nvalid': 30, 'batch_size': train_params['batch_size'], 'num_particles': 12, 'size_x': 128,
#                    'size_y': 128, 'z_scale': 700, 'min_ph': 0.1, 'valid_data_path': "./ValidingLocations_demo/",
#                    'results_path': "./Results_demo0320test/", 'dataresume': False, 'netresume': False}
#     # 原demo1 nvalid(eval_imgs_number)=30, 原demo每次valid的数据相同
#
#     # eval_params = {'threshold': 0.3, 'min_int': 0.1, 'limited_x': [0,12800], 'limited_y': [0,12800], 'tolerance':200,
#     #                 'tolerance_ax': 500, 'z_scale': psf_params['z_scale'], 'ph_scale': psf_params['ph_scale'],
#     #                'batch_size':train_params['batch_size'],'nms_cont':False,'candi_thre':0.3}
#     eval_params = {'threshold': 0.3, 'min_int': 0.1, 'limited_x': [0, data_params['size_x']*psf_params['pixel_size_x']], 'limited_y': [0, data_params['size_y']*psf_params['pixel_size_y']],
#                    'tolerance': 200, 'pixel_size_x': psf_params['pixel_size_x'], 'pixel_size_y': psf_params['pixel_size_y'],
#                    'tolerance_ax': 500, 'z_scale': psf_params['z_scale'], 'ph_scale': psf_params['ph_scale'],
#                    'batch_size': train_params['batch_size'], 'nms_cont': False, 'candi_thre': 0.3}
#     # nms_thre=0.7, threshold是什么？
#
#     infer_params = {'win_size': 128, 'batch_size': 10, 'padding': 10,
#                     "img_path": 'W://ltd//anti//imgs_sim_128_2000_0315.tif', "result_name": 'result0317.csv',
#                     'net_path': "G://Dilatedloc//INFER//2023-03-16-checkpoint.pth.tar", 'gt_path': "gt.csv"}
#
#     settings = {'net_params': net_params, 'train_params': train_params, 'camera_params': camera_params,
#                 'psf_params': psf_params, 'data_params': data_params, 'eval_params': eval_params,
#                 'infer_params': infer_params}
#
#     return settings


# demo_fromfs_Perlin_robust_training
# def parameters_set(type, factor, offset, backg):
#
#     if type =='3D':
#         net_params = {'D':121,'dilation_flag': 1,'scaling_factor': 800.0}
#     else:
#         net_params = {'n_filters': 48, 'padding': 1, 'kernel_size': 3, 'factor': factor, 'offset': offset, 'sig_pred': True,
#                       'dilation_flag': 1, 'psf_pred': True}  # 原demo1的psf_pred给的是True
#
#     # training parameters
#     # train_params = {'lr': 6e-4, 'lr_decay': 0.85, 'w_decay': 0.1, 'max_iters':80, 'interval':500,
#     #                 'batch_size': 1, 'ph_filt_thre': 160 * 5, 'P_locs_cse': True}
#     train_params = {'lr': 6e-4, 'lr_decay': 0.9, 'w_decay': 0.1, 'max_iters': 80, 'interval': 500,
#                     'batch_size': 10, 'ph_filt_thre': 500, 'P_locs_cse': True}
#
#     # camera_params = {'camera': 'sCMOS', 'em_gain': 1.0, 'surv_p': 0.5, 'margin_empty': 0.05,'backg':180,
#     #                  'qe': 0.95, 'spurious_c': 0.002, 'sig_read': 1.6, 'e_per_adu': 0.5, 'baseline': 100,
#     #                  'perlin_noise': True, 'pn_factor': 0.2, 'pn_res': 64}
#     camera_params = {'camera': 'sCMOS', 'em_gain': 1.0, 'surv_p': 0.5, 'margin_empty': 0.00, 'backg': backg,
#                      'qe': 1.0, 'spurious_c': 0.000, 'sig_read': 0.0001, 'e_per_adu': 1.0, 'baseline': 0,
#                      'perlin_noise': True, 'pn_factor': 0.2, 'pn_res': 64}
#
#     # psf_params = {'ph_scale': 8000, 'Npixels': 51, 'z_scale': 1000, 'objstage0': -1000, 'pixel_size_x': 110,
#     #               'pixel_size_y': 110, 'Npupil': 64, 'NA': 1.5, 'refmed': 1.518, 'refcov': 1.524, 'refimm': 1.518,
#     #               'wavelength': 670, 'initial_obj_stage': -1000, 'bg': 160,'robust_training':True}
#
#     psf_params = {'ph_scale': 10000, 'Npixels': 51, 'z_scale': 700, 'objstage0': -2000, 'pixel_size_x': 100,
#                   'pixel_size_y': 100, 'Npupil': 64, 'NA': 1.49, 'refmed': 1.518, 'refcov': 1.518, 'refimm': 1.518,
#                   'wavelength': 660, 'initial_obj_stage': -2000, 'bg': camera_params['backg'], 'robust_training': True}
#
#     # data_params = {'nvalid': 2000, 'batch_size': train_params['batch_size'], 'num_particles': 12, 'size_x': 128,
#     #                'size_y': 128, 'z_scale': 1000, 'min_ph': 0.1,'valid_data_path': "./ValidingLocations_demo/",
#     #                 'results_path': "./Results_demoDMO/", 'dataresume':True,'netresume':False}
#     data_params = {'nvalid': 1000, 'batch_size': train_params['batch_size'], 'num_particles': 5, 'size_x': 64,
#                    'size_y': 64, 'z_scale': 700, 'min_ph': 0.1, 'valid_data_path': "./Results_dilatedloc_demofs_train_20230329/",
#                    'results_path': "./Results_dilatedloc_demofs_train_20230329/", 'dataresume': False, 'netresume': False}
#     # 原demo1 nvalid(eval_imgs_number)=30, 原demo每次valid的数据相同
#
#     # eval_params = {'threshold': 0.3, 'min_int': 0.1, 'limited_x': [0,12800], 'limited_y': [0,12800], 'tolerance':200,
#     #                 'tolerance_ax': 500, 'z_scale': psf_params['z_scale'], 'ph_scale': psf_params['ph_scale'],
#     #                'batch_size':train_params['batch_size'],'nms_cont':False,'candi_thre':0.3}
#     eval_params = {'threshold': 0.3, 'min_int': 0.1, 'limited_x': [0, 6400], 'limited_y': [0, 6400],
#                    'tolerance': 200, 'pixel_size_x': psf_params['pixel_size_x'], 'pixel_size_y': psf_params['pixel_size_y'],
#                    'tolerance_ax': 500, 'z_scale': psf_params['z_scale'], 'ph_scale': psf_params['ph_scale'],
#                    'batch_size': train_params['batch_size'], 'nms_cont': False, 'candi_thre': 0.3}
#     # nms_thre=0.7, threshold是什么？
#
#     infer_params = {'win_size': 64, 'batch_size': 10, 'padding': 0,
#                     "img_path": "S:/Users/Fei_Yue/Field Dependent PSF Learning/demo_notebooks/demo1_fs/demo_microtubles_SMR/sequence-as-stack-MT0.N1.HD-AS-Exp.tif",
#                     "result_name": "result_demo_SMR_HD.csv",
#                     'net_path': "S:/Users/Fei_Yue/DilatedLoc-main_v5/Results_dilatedloc_demofs_train_20230328/checkpoint.pth.tar",
#                     'gt_path': "S:/Users/Fei_Yue/Field Dependent PSF Learning/demo_notebooks/demo1_fs/demo_microtubles_SMR/activations_SMR.csv"}
#
#     settings = {'net_params': net_params, 'train_params': train_params, 'camera_params': camera_params,
#                 'psf_params': psf_params, 'data_params': data_params, 'eval_params': eval_params,
#                 'infer_params': infer_params}
#
#     return settings

#demo4 backg=127.45，factor=174.55, offset=133.39
# def parameters_set(type, factor=0, offset=0, backg=0):
#
#     if type =='3D':
#         net_params = {'D':121,'dilation_flag': 1,'scaling_factor': 800.0}
#     else:
#         net_params = {'n_filters': 48, 'padding': 1, 'kernel_size': 3, 'factor': factor, 'offset': offset, 'sig_pred': True,
#                       'dilation_flag': 1, 'psf_pred': True}  # 原demo1的psf_pred给的是True
#
#     # training parameters
#     # train_params = {'lr': 6e-4, 'lr_decay': 0.85, 'w_decay': 0.1, 'max_iters':80, 'interval':500,
#     #                 'batch_size': 1, 'ph_filt_thre': 160 * 5, 'P_locs_cse': True}
#     train_params = {'lr': 6e-4, 'lr_decay': 0.9, 'w_decay': 0.1, 'max_iters': 80, 'interval': 500,
#                     'batch_size': 10, 'ph_filt_thre': backg * 5.0, 'P_locs_cse': True}
#
#     # camera_params = {'camera': 'sCMOS', 'em_gain': 1.0, 'surv_p': 0.5, 'margin_empty': 0.05,'backg':180,
#     #                  'qe': 0.95, 'spurious_c': 0.002, 'sig_read': 1.6, 'e_per_adu': 0.5, 'baseline': 100,
#     #                  'perlin_noise': True, 'pn_factor': 0.2, 'pn_res': 64}
#     camera_params = {'camera': 'sCMOS', 'em_gain': 1.0, 'surv_p': 0.5, 'margin_empty': 0.05, 'backg': backg,
#                      'qe': 0.95, 'spurious_c': 0.002, 'sig_read': 1.6, 'e_per_adu': 0.5, 'baseline': 100.0,
#                      'perlin_noise': False, 'pn_factor': 0.2, 'pn_res': 64}
#
#     # psf_params = {'ph_scale': 8000, 'Npixels': 51, 'z_scale': 1000, 'objstage0': -1000, 'pixel_size_x': 110,
#     #               'pixel_size_y': 110, 'Npupil': 64, 'NA': 1.5, 'refmed': 1.518, 'refcov': 1.524, 'refimm': 1.518,
#     #               'wavelength': 670, 'initial_obj_stage': -1000, 'bg': 160,'robust_training':True}
#
#     psf_params = {'ph_scale': 8000, 'Npixels': 51, 'z_scale': 700, 'objstage0': -800, 'pixel_size_x': 110,
#                   'pixel_size_y': 110, 'Npupil': 64, 'NA': 1.5, 'refmed': 1.406, 'refcov': 1.524, 'refimm': 1.518,
#                   'wavelength': 670, 'initial_obj_stage': -800, 'bg': camera_params['backg'], 'robust_training': False}
#
#     # data_params = {'nvalid': 2000, 'batch_size': train_params['batch_size'], 'num_particles': 12, 'size_x': 128,
#     #                'size_y': 128, 'z_scale': 1000, 'min_ph': 0.1,'valid_data_path': "./ValidingLocations_demo/",
#     #                 'results_path': "./Results_demoDMO/", 'dataresume':True,'netresume':False}
#     data_params = {'nvalid': 30, 'batch_size': train_params['batch_size'], 'num_particles': 4, 'size_x': 128,
#                    'size_y': 128, 'z_scale': 700, 'min_ph': 0.1, 'valid_data_path': "./Results_train_demo4_params_abermap_20230409_highdensity/",
#                    'results_path': "./Results_train_demo4_params_abermap_20230409_highdensity/", 'dataresume': False, 'netresume': False}
#     # 原demo1 nvalid(eval_imgs_number)=30, 原demo每次valid的数据相同
#
#     # eval_params = {'threshold': 0.3, 'min_int': 0.1, 'limited_x': [0,12800], 'limited_y': [0,12800], 'tolerance':200,
#     #                 'tolerance_ax': 500, 'z_scale': psf_params['z_scale'], 'ph_scale': psf_params['ph_scale'],
#     #                'batch_size':train_params['batch_size'],'nms_cont':False,'candi_thre':0.3}
#     eval_params = {'threshold': 0.3, 'min_int': data_params['min_ph'], 'limited_x': [0, data_params['size_x']*psf_params['pixel_size_x']], 'limited_y': [0, data_params['size_y']*psf_params['pixel_size_y']],
#                    'tolerance': 200, 'pixel_size_x': psf_params['pixel_size_x'], 'pixel_size_y': psf_params['pixel_size_y'],
#                    'tolerance_ax': 500, 'z_scale': data_params['z_scale'], 'ph_scale': psf_params['ph_scale'],
#                    'batch_size': train_params['batch_size'], 'nms_cont': False, 'candi_thre': 0.3}
#     # nms_thre=0.7, threshold是什么？s
#
#     infer_params = {'win_size': 128, 'batch_size': 10, 'padding': 0,
#                     "img_path": "S:/Users/Fei_Yue/Simlated_Data_20000/New_No_Perlin/img_sim_LD_SA.tif",
#                     "result_name": "DilatedLoc_img_sim_HD_SA.csv",
#                     'net_path': "S:/Users/Fei_Yue/Training_Model/DilatedLoc_demo4/checkpoint.pth.tar",
#                     'gt_path': "S:/Users/Fei_Yue/Simlated_Data_20000/New_No_Perlin/img_sim_LD_SA.csv"}
#
#     settings = {'net_params': net_params, 'train_params': train_params, 'camera_params': camera_params,
#                 'psf_params': psf_params, 'data_params': data_params, 'eval_params': eval_params,
#                 'infer_params': infer_params}
#
#     return settings

# demo3 backg=177.45，factor=330.77, offset=185.52
# def parameters_set(type, factor, offset, backg):
#
#     if type =='3D':
#         net_params = {'D':121,'dilation_flag': 1,'scaling_factor': 800.0}
#     else:
#         net_params = {'n_filters': 48, 'padding': 1, 'kernel_size': 3, 'factor': factor, 'offset': offset, 'sig_pred': True,
#                       'dilation_flag': 1, 'psf_pred': True}  # 原demo1的psf_pred给的是True
#
#     # training parameters
#     # train_params = {'lr': 6e-4, 'lr_decay': 0.85, 'w_decay': 0.1, 'max_iters':80, 'interval':500,
#     #                 'batch_size': 1, 'ph_filt_thre': 160 * 5, 'P_locs_cse': True}
#     train_params = {'lr': 6e-4, 'lr_decay': 0.9, 'w_decay': 0.1, 'max_iters': 80, 'interval': 500,
#                     'batch_size': 10, 'ph_filt_thre': backg * 5, 'P_locs_cse': True}
#
#     # camera_params = {'camera': 'sCMOS', 'em_gain': 1.0, 'surv_p': 0.5, 'margin_empty': 0.05,'backg':180,
#     #                  'qe': 0.95, 'spurious_c': 0.002, 'sig_read': 1.6, 'e_per_adu': 0.5, 'baseline': 100,
#     #                  'perlin_noise': True, 'pn_factor': 0.2, 'pn_res': 64}
#     camera_params = {'camera': 'sCMOS', 'em_gain': 1.0, 'surv_p': 0.5, 'margin_empty': 0.05, 'backg': backg,
#                      'qe': 0.95, 'spurious_c': 0.002, 'sig_read': 1.6, 'e_per_adu': 0.5, 'baseline': 100.0,
#                      'perlin_noise': False, 'pn_factor': 0.2, 'pn_res': 64}
#
#     # psf_params = {'ph_scale': 8000, 'Npixels': 51, 'z_scale': 1000, 'objstage0': -1000, 'pixel_size_x': 110,
#     #               'pixel_size_y': 110, 'Npupil': 64, 'NA': 1.5, 'refmed': 1.518, 'refcov': 1.524, 'refimm': 1.518,
#     #               'wavelength': 670, 'initial_obj_stage': -1000, 'bg': 160,'robust_training':True}
#
#     psf_params = {'ph_scale': 8000, 'Npixels': 51, 'z_scale': 1500, 'objstage0': -1500, 'pixel_size_x': 110,
#                   'pixel_size_y': 110, 'Npupil': 64, 'NA': 1.35, 'refmed': 1.406, 'refcov': 1.524, 'refimm': 1.406,
#                   'wavelength': 670, 'initial_obj_stage': -1500, 'bg': camera_params['backg'], 'robust_training': False}
#
#     # data_params = {'nvalid': 2000, 'batch_size': train_params['batch_size'], 'num_particles': 12, 'size_x': 128,
#     #                'size_y': 128, 'z_scale': 1000, 'min_ph': 0.1,'valid_data_path': "./ValidingLocations_demo/",
#     #                 'results_path': "./Results_demoDMO/", 'dataresume':True,'netresume':False}
#     data_params = {'nvalid': 30, 'batch_size': train_params['batch_size'], 'num_particles': 4, 'size_x': 128,
#                    'size_y': 128, 'z_scale': 1500, 'min_ph': 0.1, 'valid_data_path': "./Results_demo3_train_20230408_highdensity/",
#                    'results_path': "./Results_demo3_train_20230408_highdensity/", 'dataresume': False, 'netresume': False}
#     # 原demo1 nvalid(eval_imgs_number)=30, 原demo每次valid的数据相同
#
#     # eval_params = {'threshold': 0.3, 'min_int': 0.1, 'limited_x': [0,12800], 'limited_y': [0,12800], 'tolerance':200,
#     #                 'tolerance_ax': 500, 'z_scale': psf_params['z_scale'], 'ph_scale': psf_params['ph_scale'],
#     #                'batch_size':train_params['batch_size'],'nms_cont':False,'candi_thre':0.3}
#     eval_params = {'threshold': 0.3, 'min_int': 0.1, 'limited_x': [0, data_params['size_x'] * psf_params['pixel_size_x']], 'limited_y': [0, data_params['size_y'] * psf_params['pixel_size_y']],
#                    'tolerance': 200, 'pixel_size_x': psf_params['pixel_size_x'], 'pixel_size_y': psf_params['pixel_size_y'],
#                    'tolerance_ax': 500, 'z_scale': psf_params['z_scale'], 'ph_scale': psf_params['ph_scale'],
#                    'batch_size': train_params['batch_size'], 'nms_cont': False, 'candi_thre': 0.3}
#     # nms_thre=0.7, threshold是什么？
#
#     infer_params = {'win_size': 64, 'batch_size': 10, 'padding': 0,
#                     "img_path": "S:/Users/Fei_Yue/Field Dependent PSF Learning/demo_notebooks/demo1_fs/demo_microtubles_SMR/sequence-as-stack-MT0.N1.HD-AS-Exp.tif",
#                     "result_name": "result_demo_SMR_HD.csv",
#                     'net_path': "S:/Users/Fei_Yue/DilatedLoc-main_v5/Results_dilatedloc_demofs_train_20230328/checkpoint.pth.tar",
#                     'gt_path': "S:/Users/Fei_Yue/Field Dependent PSF Learning/demo_notebooks/demo1_fs/demo_microtubles_SMR/activations_SMR.csv"}
#
#     settings = {'net_params': net_params, 'train_params': train_params, 'camera_params': camera_params,
#                 'psf_params': psf_params, 'data_params': data_params, 'eval_params': eval_params,
#                 'infer_params': infer_params}
#
#     return settings

# demo4 High SNR
# def parameters_set(type, factor=0, offset=0, backg=0):
#
#     if type =='3D':
#         net_params = {'D':121,'dilation_flag': 1,'scaling_factor': 800.0}
#     else:
#         net_params = {'n_filters': 48, 'padding': 1, 'kernel_size': 3, 'factor': factor, 'offset': offset, 'sig_pred': True,
#                       'dilation_flag': 1, 'psf_pred': True}  # 原demo1的psf_pred给的是True
#
#     # training parameters
#     # train_params = {'lr': 6e-4, 'lr_decay': 0.85, 'w_decay': 0.1, 'max_iters':80, 'interval':500,
#     #                 'batch_size': 1, 'ph_filt_thre': 160 * 5, 'P_locs_cse': True}
#     train_params = {'lr': 6e-4, 'lr_decay': 0.9, 'w_decay': 0.1, 'max_iters': 80, 'interval': 500,
#                     'batch_size': 10, 'ph_filt_thre': backg * 5.0, 'P_locs_cse': True}
#
#     # camera_params = {'camera': 'sCMOS', 'em_gain': 1.0, 'surv_p': 0.5, 'margin_empty': 0.05,'backg':180,
#     #                  'qe': 0.95, 'spurious_c': 0.002, 'sig_read': 1.6, 'e_per_adu': 0.5, 'baseline': 100,
#     #                  'perlin_noise': True, 'pn_factor': 0.2, 'pn_res': 64}
#     camera_params = {'camera': 'sCMOS', 'em_gain': 1.0, 'surv_p': 0.5, 'margin_empty': 0.05, 'backg': 30,
#                      'qe': 1, 'spurious_c': 0.000, 'sig_read': 0.0001, 'e_per_adu': 1, 'baseline': 0,
#                      'perlin_noise': False, 'pn_factor': 0.2, 'pn_res': 64}
#
#     # psf_params = {'ph_scale': 8000, 'Npixels': 51, 'z_scale': 1000, 'objstage0': -1000, 'pixel_size_x': 110,
#     #               'pixel_size_y': 110, 'Npupil': 64, 'NA': 1.5, 'refmed': 1.518, 'refcov': 1.524, 'refimm': 1.518,
#     #               'wavelength': 670, 'initial_obj_stage': -1000, 'bg': 160,'robust_training':True}
#
#     psf_params = {'ph_scale': 8000, 'Npixels': 51, 'z_scale': 700, 'objstage0': -800, 'pixel_size_x': 110,
#                   'pixel_size_y': 110, 'Npupil': 64, 'NA': 1.5, 'refmed': 1.406, 'refcov': 1.524, 'refimm': 1.518,
#                   'wavelength': 670, 'initial_obj_stage': -800, 'bg': camera_params['backg'], 'robust_training': False}
#
#     # data_params = {'nvalid': 2000, 'batch_size': train_params['batch_size'], 'num_particles': 12, 'size_x': 128,
#     #                'size_y': 128, 'z_scale': 1000, 'min_ph': 0.1,'valid_data_path': "./ValidingLocations_demo/",
#     #                 'results_path': "./Results_demoDMO/", 'dataresume':True,'netresume':False}
#     data_params = {'nvalid': 30, 'batch_size': train_params['batch_size'], 'num_particles': 4, 'size_x': 128,
#                    'size_y': 128, 'z_scale': 700, 'min_ph': 0.1, 'valid_data_path': "./Results_train_demo4_params_abermap_20230408_highdensity/",
#                    'results_path': "./Results_train_demo4_params_abermap_20230408_highdenisty/", 'dataresume': False, 'netresume': False}
#     # 原demo1 nvalid(eval_imgs_number)=30, 原demo每次valid的数据相同
#
#     # eval_params = {'threshold': 0.3, 'min_int': 0.1, 'limited_x': [0,12800], 'limited_y': [0,12800], 'tolerance':200,
#     #                 'tolerance_ax': 500, 'z_scale': psf_params['z_scale'], 'ph_scale': psf_params['ph_scale'],
#     #                'batch_size':train_params['batch_size'],'nms_cont':False,'candi_thre':0.3}
#     eval_params = {'threshold': 0.3, 'min_int': 0.1, 'limited_x': [0, data_params['size_x']*psf_params['pixel_size_x']], 'limited_y': [0, data_params['size_y']*psf_params['pixel_size_y']],
#                    'tolerance': 200, 'pixel_size_x': psf_params['pixel_size_x'], 'pixel_size_y': psf_params['pixel_size_y'],
#                    'tolerance_ax': 500, 'z_scale': psf_params['z_scale'], 'ph_scale': psf_params['ph_scale'],
#                    'batch_size': train_params['batch_size'], 'nms_cont': False, 'candi_thre': 0.3}
#     # nms_thre=0.7, threshold是什么？
#
#     infer_params = {'win_size': 64, 'batch_size': 10, 'padding': 0,
#                     "img_path": "S:/Users/Fei_Yue/DilatedLoc-main_v5/simulate_data_20230406/demo4_para/demo4_para_yx.tif",
#                     "result_name": "result_demo4_para_aber_SMR_yx.csv",
#                     'net_path': "S:/Users/Fei_Yue/DilatedLoc-main_v5/Results_train_demo4_params_abermap_20230406/checkpoint.pth.tar",
#                     'gt_path': "S:/Users/Fei_Yue/DilatedLoc-main_v5/simulate_data_20230406/demo4_para/demo4_activations.csv"}
#
#     settings = {'net_params': net_params, 'train_params': train_params, 'camera_params': camera_params,
#                 'psf_params': psf_params, 'data_params': data_params, 'eval_params': eval_params,
#                 'infer_params': infer_params}
#
#     return settings

# demo3 High SNR
# def parameters_set(type, factor, offset, backg):
#
#     if type =='3D':
#         net_params = {'D':121,'dilation_flag': 1,'scaling_factor': 800.0}
#     else:
#         net_params = {'n_filters': 48, 'padding': 1, 'kernel_size': 3, 'factor': factor, 'offset': offset, 'sig_pred': True,
#                       'dilation_flag': 1, 'psf_pred': True}  # 原demo1的psf_pred给的是True
#
#     # training parameters
#     # train_params = {'lr': 6e-4, 'lr_decay': 0.85, 'w_decay': 0.1, 'max_iters':80, 'interval':500,
#     #                 'batch_size': 1, 'ph_filt_thre': 160 * 5, 'P_locs_cse': True}
#     train_params = {'lr': 6e-4, 'lr_decay': 0.9, 'w_decay': 0.1, 'max_iters': 80, 'interval': 500,
#                     'batch_size': 10, 'ph_filt_thre': backg * 5, 'P_locs_cse': True}
#
#     # camera_params = {'camera': 'sCMOS', 'em_gain': 1.0, 'surv_p': 0.5, 'margin_empty': 0.05,'backg':180,
#     #                  'qe': 0.95, 'spurious_c': 0.002, 'sig_read': 1.6, 'e_per_adu': 0.5, 'baseline': 100,
#     #                  'perlin_noise': True, 'pn_factor': 0.2, 'pn_res': 64}
#     # camera_params = {'camera': 'sCMOS', 'em_gain': 1.0, 'surv_p': 0.5, 'margin_empty': 0.05, 'backg': backg,
#     #                  'qe': 0.95, 'spurious_c': 0.002, 'sig_read': 1.6, 'e_per_adu': 0.5, 'baseline': 100.0,
#     #                  'perlin_noise': False, 'pn_factor': 0.2, 'pn_res': 64}
#     camera_params = {'camera': 'sCMOS', 'em_gain': 1.0, 'surv_p': 0.5, 'margin_empty': 0.05, 'backg': backg,
#                      'qe': 1, 'spurious_c': 0.000, 'sig_read': 0.0001, 'e_per_adu': 1, 'baseline': 0,
#                      'perlin_noise': False, 'pn_factor': 0.2, 'pn_res': 64} # demo3 high SNR
#
#     # psf_params = {'ph_scale': 8000, 'Npixels': 51, 'z_scale': 1000, 'objstage0': -1000, 'pixel_size_x': 110,
#     #               'pixel_size_y': 110, 'Npupil': 64, 'NA': 1.5, 'refmed': 1.518, 'refcov': 1.524, 'refimm': 1.518,
#     #               'wavelength': 670, 'initial_obj_stage': -1000, 'bg': 160,'robust_training':True}
#
#     psf_params = {'ph_scale': 8000, 'Npixels': 51, 'z_scale': 1500, 'objstage0': -1500, 'pixel_size_x': 110,
#                   'pixel_size_y': 110, 'Npupil': 64, 'NA': 1.35, 'refmed': 1.406, 'refcov': 1.524, 'refimm': 1.406,
#                   'wavelength': 670, 'initial_obj_stage': -1500, 'bg': camera_params['backg'], 'robust_training': False}
#
#     # data_params = {'nvalid': 2000, 'batch_size': train_params['batch_size'], 'num_particles': 12, 'size_x': 128,
#     #                'size_y': 128, 'z_scale': 1000, 'min_ph': 0.1,'valid_data_path': "./ValidingLocations_demo/",
#     #                 'results_path': "./Results_demoDMO/", 'dataresume':True,'netresume':False}
#     data_params = {'nvalid': 30, 'batch_size': train_params['batch_size'], 'num_particles': 10, 'size_x': 128,
#                    'size_y': 128, 'z_scale': 1500, 'min_ph': 0.1, 'valid_data_path': "./Results_demo3_train_20230408_highdensity/",
#                    'results_path': "./Results_demo3_train_20230408_highdensity/", 'dataresume': False, 'netresume': False}
#     # 原demo1 nvalid(eval_imgs_number)=30, 原demo每次valid的数据相同
#
#     # eval_params = {'threshold': 0.3, 'min_int': 0.1, 'limited_x': [0,12800], 'limited_y': [0,12800], 'tolerance':200,
#     #                 'tolerance_ax': 500, 'z_scale': psf_params['z_scale'], 'ph_scale': psf_params['ph_scale'],
#     #                'batch_size':train_params['batch_size'],'nms_cont':False,'candi_thre':0.3}
#     eval_params = {'threshold': 0.3, 'min_int': 0.1, 'limited_x': [0, data_params['size_x'] * psf_params['pixel_size_x']], 'limited_y': [0, data_params['size_y'] * psf_params['pixel_size_y']],
#                    'tolerance': 200, 'pixel_size_x': psf_params['pixel_size_x'], 'pixel_size_y': psf_params['pixel_size_y'],
#                    'tolerance_ax': 500, 'z_scale': psf_params['z_scale'], 'ph_scale': psf_params['ph_scale'],
#                    'batch_size': train_params['batch_size'], 'nms_cont': False, 'candi_thre': 0.3}
#     # nms_thre=0.7, threshold是什么？
#
#     infer_params = {'win_size': 128, 'batch_size': 10, 'padding': 0,
#                     "img_path": "S:/Users/Fei_Yue/Simlated_Data_20000/SaddlePoint/img_sim_LD_SA.tif",
#                     "result_name": "DilatedLoc_result_img_sim_LD_SA.csv",
#                     'net_path': "S:/Users/Fei_Yue/Training_Model/DilatedLoc_demo4/checkpoint.pth.tar",
#                     'gt_path': "S:/Users/Fei_Yue/Simlated_Data_20000/SaddlePoint/img_sim_LD_SA.csv"}
#
#     settings = {'net_params': net_params, 'train_params': train_params, 'camera_params': camera_params,
#                 'psf_params': psf_params, 'data_params': data_params, 'eval_params': eval_params,
#                 'infer_params': infer_params}
#
#     return settings

# demo_6um backg=177.45，factor=330.77, offset=185.52
# def parameters_set(type, factor, offset, backg):
#
#     if type =='3D':
#         net_params = {'D':121,'dilation_flag': 1,'scaling_factor': 800.0}
#     else:
#         net_params = {'n_filters': 48, 'padding': 1, 'kernel_size': 3, 'factor': factor, 'offset': offset, 'sig_pred': True,
#                       'dilation_flag': 1, 'psf_pred': True}  # 原demo1的psf_pred给的是True
#
#     # training parameters
#     # train_params = {'lr': 6e-4, 'lr_decay': 0.85, 'w_decay': 0.1, 'max_iters':80, 'interval':500,
#     #                 'batch_size': 1, 'ph_filt_thre': 160 * 5, 'P_locs_cse': True}
#     train_params = {'lr': 6e-4, 'lr_decay': 0.9, 'w_decay': 0.1, 'max_iters': 80, 'interval': 500,
#                     'batch_size': 10, 'ph_filt_thre': backg * 5, 'P_locs_cse': True}
#
#     # camera_params = {'camera': 'sCMOS', 'em_gain': 1.0, 'surv_p': 0.5, 'margin_empty': 0.05,'backg':180,
#     #                  'qe': 0.95, 'spurious_c': 0.002, 'sig_read': 1.6, 'e_per_adu': 0.5, 'baseline': 100,
#     #                  'perlin_noise': True, 'pn_factor': 0.2, 'pn_res': 64}
#     # camera_params = {'camera': 'sCMOS', 'em_gain': 1.0, 'surv_p': 0.5, 'margin_empty': 0.05, 'backg': backg,
#     #                  'qe': 0.95, 'spurious_c': 0.002, 'sig_read': 1.6, 'e_per_adu': 0.5, 'baseline': 100.0,
#     #                  'perlin_noise': False, 'pn_factor': 0.2, 'pn_res': 64}
#     camera_params = {'camera': 'sCMOS', 'em_gain': 1.0, 'surv_p': 0.5, 'margin_empty': 0.05, 'backg': 30,
#                     'qe': 1, 'spurious_c': 0.000, 'sig_read': 0.0001, 'e_per_adu': 1, 'baseline': 0,
#                     'perlin_noise': False, 'pn_factor': 0.2, 'pn_res': 64}
#
#     # psf_params = {'ph_scale': 8000, 'Npixels': 51, 'z_scale': 1000, 'objstage0': -1000, 'pixel_size_x': 110,
#     #               'pixel_size_y': 110, 'Npupil': 64, 'NA': 1.5, 'refmed': 1.518, 'refcov': 1.524, 'refimm': 1.518,
#     #               'wavelength': 670, 'initial_obj_stage': -1000, 'bg': 160,'robust_training':True}
#
#     psf_params = {'ph_scale': 8000, 'Npixels': 51, 'z_scale': 3000, 'objstage0': -1500, 'pixel_size_x': 100,
#                   'pixel_size_y': 100, 'Npupil': 64, 'NA': 1.35, 'refmed': 1.406, 'refcov': 1.524, 'refimm': 1.406,
#                   'wavelength': 670, 'initial_obj_stage': -1500, 'bg': camera_params['backg'], 'robust_training': False}
#
#     # data_params = {'nvalid': 2000, 'batch_size': train_params['batch_size'], 'num_particles': 12, 'size_x': 128,
#     #                'size_y': 128, 'z_scale': 1000, 'min_ph': 0.1,'valid_data_path': "./ValidingLocations_demo/",
#     #                 'results_path': "./Results_demoDMO/", 'dataresume':True,'netresume':False}
#     data_params = {'nvalid': 30, 'batch_size': train_params['batch_size'], 'num_particles': 10, 'size_x': 64,
#                    'size_y': 64, 'z_scale': 3000, 'min_ph': 0.1, 'valid_data_path': "./Results_Tetrapod_6um_train_20230418/",
#                    'results_path': "./Results_Tetrapod_6um_train_20230418/", 'dataresume': False, 'netresume': False}
#     # 原demo1 nvalid(eval_imgs_number)=30, 原demo每次valid的数据相同
#
#     # eval_params = {'threshold': 0.3, 'min_int': 0.1, 'limited_x': [0,12800], 'limited_y': [0,12800], 'tolerance':200,
#     #                 'tolerance_ax': 500, 'z_scale': psf_params['z_scale'], 'ph_scale': psf_params['ph_scale'],
#     #                'batch_size':train_params['batch_size'],'nms_cont':False,'candi_thre':0.3}
#     eval_params = {'threshold': 0.3, 'min_int': 0.1, 'limited_x': [0, data_params['size_x'] * psf_params['pixel_size_x']], 'limited_y': [0, data_params['size_y'] * psf_params['pixel_size_y']],
#                    'tolerance': 200, 'pixel_size_x': psf_params['pixel_size_x'], 'pixel_size_y': psf_params['pixel_size_y'],
#                    'tolerance_ax': 500, 'z_scale': 700, 'ph_scale': 10000,
#                    'batch_size': train_params['batch_size'], 'nms_cont': False, 'candi_thre': 0.3}
#     # nms_thre=0.7, threshold是什么？
#
#     infer_params = {'win_size': 64, 'batch_size': 10, 'padding': 0,
#                     "img_path": './sim_data/demo1_activations_new.tif',
#                     "result_name": './sim_data/result_demo1_activations_new.csv',
#                     'net_path': "X:/ltd/Results_demo0320test_641/checkpoint.pth.tar",
#                     'gt_path': "./sim_data/demo1_activations_new_noNo.csv"}
#
#     settings = {'net_params': net_params, 'train_params': train_params, 'camera_params': camera_params,
#                 'psf_params': psf_params, 'data_params': data_params, 'eval_params': eval_params,
#                 'infer_params': infer_params}
#
#     return settings

# DMO-Tetrapod PSF 理想条件下参数设置
# def parameters_set(type, factor=69.38, offset=50.54, backg=49.91):
#
#     if type =='3D':
#         net_params = {'D':121, 'dilation_flag': 1, 'scaling_factor': 800.0}
#     else:
#         net_params = {'n_filters': 48, 'padding': 1, 'kernel_size': 3, 'factor': factor, 'offset': offset, 'sig_pred': True,
#                       'dilation_flag': 1, 'psf_pred': True}  # 原demo1的psf_pred给的是True
#
#     # training parameters
#     train_params = {'lr': 6e-4, 'lr_decay': 0.9, 'w_decay': 0.1, 'max_iters': 80, 'interval': 500,
#                     'batch_size': 10, 'ph_filt_thre': 500, 'P_locs_cse': True}
#     # lr_decay=0.9, ph_filt_thre=500
#
#     camera_params = {'camera': 'sCMOS', 'em_gain': 1.0, 'surv_p': 0.5, 'margin_empty': 0.05, 'backg': backg,
#                      'qe': 1, 'spurious_c': 0.000, 'sig_read': 0.0001, 'e_per_adu': 1, 'baseline': 0,
#                      'perlin_noise': False, 'pn_factor': 0.2, 'pn_res': 64}
#
#     psf_params = {'ph_scale': 7000, 'Npixels': 51, 'z_scale': 1500, 'objstage0': -800, 'pixel_size_x': 100,
#                   'pixel_size_y': 100, 'Npupil': 64, 'NA': 1.49, 'refmed': 1.518, 'refcov': 1.518, 'refimm': 1.518,
#                   'wavelength': 660, 'initial_obj_stage': -800, 'bg': backg, 'robust_training': False}
#     # Npupil是什么？objstage0是什么？bg是什么？
#
#     data_params = {'nvalid': 30, 'batch_size': train_params['batch_size'], 'num_particles': 6, 'size_x': 128,
#                    'size_y': 128, 'z_scale': 1500, 'min_ph': 0.1, 'valid_data_path': "./NewResults_20230426/DMO-Tetrapod/",
#                    'results_path': "./NewResults_20230426/DMO-Tetrapod/", 'dataresume': False, 'netresume': False}
#     # 原demo1 nvalid(eval_imgs_number)=30, 原demo每次valid的数据相同
#
#     eval_params = {'threshold': 0.3, 'min_int': 0.1, 'limited_x': [0, data_params['size_x']*psf_params['pixel_size_x']], 'limited_y': [0, data_params['size_y']*psf_params['pixel_size_y']],
#                    'tolerance': 200, 'pixel_size_x': psf_params['pixel_size_x'], 'pixel_size_y': psf_params['pixel_size_y'],
#                    'tolerance_ax': 500, 'z_scale': 1500, 'ph_scale': psf_params['ph_scale'],
#                    'batch_size': train_params['batch_size'], 'nms_cont': False, 'candi_thre': 0.3}
#     # nms_thre=0.7, threshold是什么？
#
#     infer_params = {'win_size': 64, 'batch_size': 10, 'padding': 0,
#                     "img_path": 'W://ltd//anti//imgs_sim_128_2000_0315.tif', "result_name": 'result0317.csv',
#                     'net_path': "G://Dilatedloc//INFER//2023-03-16-checkpoint.pth.tar", 'gt_path': "gt.csv"}
#
#
#     settings = {'net_params': net_params, 'train_params': train_params, 'camera_params': camera_params,
#                 'psf_params': psf_params, 'data_params': data_params, 'eval_params': eval_params,
#                 'infer_params': infer_params}
#
#     return settings

# newest simulation parameters
def parameters_set(type, factor=69.38, offset=50.54, backg=49.91):

    if type =='3D':
        net_params = {'D':121, 'dilation_flag': 1, 'scaling_factor': 800.0}
    else:
        net_params = {'n_filters': 48, 'padding': 1, 'kernel_size': 3, 'factor': factor, 'offset': offset, 'sig_pred': True,
                      'dilation_flag': 1, 'psf_pred': False}  # 原demo1的psf_pred给的是True

    # training parameters
    train_params = {'lr': 6e-4, 'lr_decay': 0.9, 'w_decay': 0.1, 'max_iters': 80, 'interval': 500,
                    'batch_size': 10, 'ph_filt_thre': 500, 'P_locs_cse': True}
    # lr_decay=0.9, ph_filt_thre=500

    camera_params = {'camera': 'sCMOS', 'em_gain': 1.0, 'surv_p': 0.5, 'margin_empty': 0.05, 'backg': backg,
                     'qe': 1, 'spurious_c': 0.000, 'sig_read': 0.0001, 'e_per_adu': 1, 'baseline': 0,
                     'perlin_noise': False, 'pn_factor': 0.2, 'pn_res': 64}

    psf_params = {'ph_scale': 20000, 'Npixels': 51, 'z_scale': 1000, 'objstage0': -800, 'pixel_size_x': 100,
                  'pixel_size_y': 100, 'Npupil': 64, 'NA': 1.49, 'refmed': 1.518, 'refcov': 1.518, 'refimm': 1.518,
                  'wavelength': 660, 'initial_obj_stage': -800, 'bg': backg, 'robust_training': False}
    # Npupil是什么？objstage0是什么？bg是什么？

    data_params = {'nvalid': 30, 'batch_size': train_params['batch_size'], 'num_particles': 6, 'size_x': 128,
                   'size_y': 128, 'z_scale': 1000, 'min_ph': 0.1, 'valid_data_path': "./NewResults_20230426/DMO-Tetrapod/",
                   'results_path': "./NewResults_20230426/DMO-Tetrapod/", 'dataresume': False, 'netresume': False}
    # 原demo1 nvalid(eval_imgs_number)=30, 原demo每次valid的数据相同

    eval_params = {'threshold': 0.3, 'min_int': 0.1, 'limited_x': [0, data_params['size_x']*psf_params['pixel_size_x']], 'limited_y': [0, data_params['size_y']*psf_params['pixel_size_y']],
                   'tolerance': 200, 'pixel_size_x': psf_params['pixel_size_x'], 'pixel_size_y': psf_params['pixel_size_y'],
                   'tolerance_ax': 500, 'z_scale': 1000, 'ph_scale': psf_params['ph_scale'],
                   'batch_size': train_params['batch_size'], 'nms_cont': False, 'candi_thre': 0.3}
    # nms_thre=0.7, threshold是什么？

    infer_params = {'win_size': 128, 'batch_size': 10, 'padding': 0,
                    "img_path": 'W://ltd//anti//imgs_sim_128_2000_0315.tif', "result_name": 'result1108.csv',
                    'net_path': "G://Dilatedloc//INFER//2023-03-16-checkpoint.pth.tar", 'gt_path': "gt.csv"}


    settings = {'net_params': net_params, 'train_params': train_params, 'camera_params': camera_params,
                'psf_params': psf_params, 'data_params': data_params, 'eval_params': eval_params,
                'infer_params': infer_params}

    return settings

# NPC Tetrapod_6um setting parameters (sw)
# def parameters_set(type, factor=69.38, offset=50.54, backg=130.52):
#
#     if type =='3D':
#         net_params = {'D':121, 'dilation_flag': 1, 'scaling_factor': 800.0}
#     else:
#         net_params = {'n_filters': 48, 'padding': 1, 'kernel_size': 3, 'factor': factor, 'offset': offset, 'sig_pred': True,
#                       'dilation_flag': 1, 'psf_pred': False}  # 原demo1的psf_pred给的是True
#
#     # training parameters
#     train_params = {'lr': 6e-4, 'lr_decay': 0.9, 'w_decay': 0.1, 'max_iters': 80, 'interval': 500,
#                     'batch_size': 10, 'ph_filt_thre': 500, 'P_locs_cse': True}
#     # lr_decay=0.9, ph_filt_thre=500
#
#     camera_params = {'camera': 'sCMOS', 'em_gain': 1.0, 'surv_p': 0.5, 'margin_empty': 0.05, 'backg': backg,
#                      'qe': 0.95, 'spurious_c': 0.002, 'sig_read': 1.535, 'e_per_adu': 0.7471, 'baseline': 100.0,
#                      'perlin_noise': True, 'pn_factor': 0.2, 'pn_res': 64}
#
#     psf_params = {'ph_scale': 6000, 'Npixels': 51, 'z_scale': 3000, 'objstage0': -800, 'pixel_size_x': 110,
#                   'pixel_size_y': 110, 'Npupil': 64, 'NA': 1.35, 'refmed': 1.406, 'refcov': 1.525, 'refimm': 1.406,
#                   'wavelength': 680, 'initial_obj_stage': -800, 'bg': backg, 'robust_training': True}
#     # Npupil是什么？objstage0是什么？bg是什么？
#
#     data_params = {'nvalid': 30, 'batch_size': train_params['batch_size'], 'num_particles': 6, 'size_x': 128,
#                    'size_y': 128, 'z_scale': 3000, 'min_ph': 0.1, 'valid_data_path': "./NewResults_20230426/DMO-Tetrapod/",
#                    'results_path': "./NewResults_20230426/DMO-Tetrapod/", 'dataresume': False, 'netresume': False}
#     # 原demo1 nvalid(eval_imgs_number)=30, 原demo每次valid的数据相同
#
#     eval_params = {'threshold': 0.3, 'min_int': 0.05, 'limited_x': [0, data_params['size_x']*psf_params['pixel_size_x']], 'limited_y': [0, data_params['size_y']*psf_params['pixel_size_y']],
#                    'tolerance': 200, 'pixel_size_x': psf_params['pixel_size_x'], 'pixel_size_y': psf_params['pixel_size_y'],
#                    'tolerance_ax': 500, 'z_scale': 3000, 'ph_scale': psf_params['ph_scale'],
#                    'batch_size': train_params['batch_size'], 'nms_cont': False, 'candi_thre': 0.3}
#     # nms_thre=0.7, threshold是什么？
#
#     infer_params = {'win_size': 128, 'batch_size': 10, 'padding': 0,
#                     "img_path": 'W://ltd//anti//imgs_sim_128_2000_0315.tif', "result_name": 'result0317.csv',
#                     'net_path': "G://Dilatedloc//INFER//2023-03-16-checkpoint.pth.tar", 'gt_path': "gt.csv"}
#
#
#     settings = {'net_params': net_params, 'train_params': train_params, 'camera_params': camera_params,
#                 'psf_params': psf_params, 'data_params': data_params, 'eval_params': eval_params,
#                 'infer_params': infer_params}
#
#     return settings

# NPC Tetrapod_6um setting parameters (fs)
# def parameters_set(type, factor=69.38, offset=50.54, backg=130.52):
#
#     if type =='3D':
#         net_params = {'D':121, 'dilation_flag': 1, 'scaling_factor': 800.0}
#     else:
#         net_params = {'n_filters': 48, 'padding': 1, 'kernel_size': 3, 'factor': factor, 'offset': offset, 'sig_pred': True,
#                       'dilation_flag': 1, 'psf_pred': False}  # 原demo1的psf_pred给的是True
#
#     # training parameters
#     train_params = {'lr': 6e-4, 'lr_decay': 0.9, 'w_decay': 0.1, 'max_iters': 80, 'interval': 500,
#                     'batch_size': 10, 'ph_filt_thre': 800, 'P_locs_cse': True}
#     # lr_decay=0.9, ph_filt_thre=500
#
#     camera_params = {'camera': 'sCMOS', 'em_gain': 1.0, 'surv_p': 0.5, 'margin_empty': 0.05, 'backg': backg,
#                      'qe': 0.95, 'spurious_c': 0.002, 'sig_read': 1.6, 'e_per_adu': 0.5, 'baseline': 100.0,
#                      'perlin_noise': True, 'pn_factor': 0.2, 'pn_res': 64}
#
#     psf_params = {'ph_scale': 6000, 'Npixels': 61, 'z_scale': 3000, 'objstage0': -800, 'pixel_size_x': 110,
#                   'pixel_size_y': 110, 'Npupil': 64, 'NA': 1.35, 'refmed': 1.406, 'refcov': 1.524, 'refimm': 1.406,
#                   'wavelength': 670, 'initial_obj_stage': -800, 'bg': backg, 'robust_training': True}
#     # Npupil是什么？objstage0是什么？bg是什么？
#
#     data_params = {'nvalid': 30, 'batch_size': train_params['batch_size'], 'num_particles': 6, 'size_x': 128,
#                    'size_y': 128, 'z_scale': 3000, 'min_ph': 0.1, 'valid_data_path': "./NewResults_20230426/DMO-Tetrapod/",
#                    'results_path': "./NewResults_20230426/DMO-Tetrapod/", 'dataresume': False, 'netresume': False}
#     # 原demo1 nvalid(eval_imgs_number)=30, 原demo每次valid的数据相同
#
#     eval_params = {'threshold': 0.3, 'min_int': 0.05, 'limited_x': [0, data_params['size_x']*psf_params['pixel_size_x']], 'limited_y': [0, data_params['size_y']*psf_params['pixel_size_y']],
#                    'tolerance': 200, 'pixel_size_x': psf_params['pixel_size_x'], 'pixel_size_y': psf_params['pixel_size_y'],
#                    'tolerance_ax': 500, 'z_scale': 3000, 'ph_scale': psf_params['ph_scale'],
#                    'batch_size': train_params['batch_size'], 'nms_cont': False, 'candi_thre': 0.3}
#     # nms_thre=0.7, threshold是什么？
#
#     infer_params = {'win_size': 128, 'batch_size': 10, 'padding': 0,
#                     "img_path": 'W://ltd//anti//imgs_sim_128_2000_0315.tif', "result_name": 'result0317.csv',
#                     'net_path': "G://Dilatedloc//INFER//2023-03-16-checkpoint.pth.tar", 'gt_path': "gt.csv"}
#
#
#     settings = {'net_params': net_params, 'train_params': train_params, 'camera_params': camera_params,
#                 'psf_params': psf_params, 'data_params': data_params, 'eval_params': eval_params,
#                 'infer_params': infer_params}
#
#     return settings

def parameters_set_crlb(psf_type):

    eval_params = {'threshold': 0.9, 'min_int': 0.1, 'limited_x': [0, 5100], 'limited_y': [0, 5100],
                   'tolerance': 250, 'pixel_size_x': 100, 'pixel_size_y': 100,
                   'tolerance_ax': 500, 'z_scale': 700, 'ph_scale': 20000,
                   'batch_size': 10, 'nms_cont': False, 'candi_thre': 0.3}

    infer_params = {'win_size': 51, 'batch_size': 250, 'padding': 0,  # 如果要做大视场，用bg填充而不是0; meanwhile, 有影响。
                    "img_path": './CRLB_sampling_data/' + psf_type + "_sampling_data.tif", "result_name": './CRLB_sampling_data/DilatedLoc_crlb_' + psf_type + "_new_new.csv",
                    'net_path': "./Training_model/"+ psf_type + "/High_SNR/checkpoint.pth.tar",
                    'gt_path': './CRLB_sampling_data/' + psf_type + "_sampling_gt_noNo.csv"}


    settings = {'eval_params': eval_params, 'infer_params': infer_params}

    return settings

# Tetrad_6um, params is from_sw
# def parameters_set(type, factor=69.38, offset=50.54, backg=49.91):
#
#     if type =='3D':
#         net_params = {'D':121, 'dilation_flag': 1, 'scaling_factor': 800.0}
#     else:
#         net_params = {'n_filters': 48, 'padding': 1, 'kernel_size': 3, 'factor': factor, 'offset': offset, 'sig_pred': True,
#                       'dilation_flag': 1, 'psf_pred': False}  # 原demo1的psf_pred给的是True
#
#     # training parameters
#     train_params = {'lr': 6e-4, 'lr_decay': 0.9, 'w_decay': 0.1, 'max_iters': 80, 'interval': 500,
#                     'batch_size': 10, 'ph_filt_thre': 500, 'P_locs_cse': True}
#     # lr_decay=0.9, ph_filt_thre=500
#
#     camera_params = {'camera': 'sCMOS', 'em_gain': 1.0, 'surv_p': 0.5, 'margin_empty': 0.05, 'backg': backg,
#                      'qe': 0.95, 'spurious_c': 0.002, 'sig_read': 1.6, 'e_per_adu': 0.5, 'baseline': 100,
#                      'perlin_noise': False, 'pn_factor': 0.2, 'pn_res': 64}
#
#     psf_params = {'ph_scale': 20000, 'Npixels': 51, 'z_scale': 1500, 'objstage0': -800, 'pixel_size_x': 110,
#                   'pixel_size_y': 100, 'Npupil': 64, 'NA': 1.49, 'refmed': 1.406, 'refcov': 1.524, 'refimm': 1.406,
#                   'wavelength': 670, 'initial_obj_stage': -800, 'bg': backg, 'robust_training': False}
#     # Npupil是什么？objstage0是什么？bg是什么？
#
#     data_params = {'nvalid': 10, 'batch_size': train_params['batch_size'], 'num_particles': 6, 'size_x': 128,
#                    'size_y': 128, 'z_scale': 1500, 'min_ph': 0.1, 'valid_data_path': "./NewResults_20230426/DMO-Tetrapod/",
#                    'results_path': "./NewResults_20230426/DMO-Tetrapod/", 'dataresume': False, 'netresume': False}
#     # 原demo1 nvalid(eval_imgs_number)=30, 原demo每次valid的数据相同
#
#     eval_params = {'threshold': 0.3, 'min_int': 0.1, 'limited_x': [0, data_params['size_x']*psf_params['pixel_size_x']], 'limited_y': [0, data_params['size_y']*psf_params['pixel_size_y']],
#                    'tolerance': 200, 'pixel_size_x': psf_params['pixel_size_x'], 'pixel_size_y': psf_params['pixel_size_y'],
#                    'tolerance_ax': 500, 'z_scale': 3000, 'ph_scale': psf_params['ph_scale'],
#                    'batch_size': train_params['batch_size'], 'nms_cont': False, 'candi_thre': 0.3}
#     # nms_thre=0.7, threshold是什么？
#
#     infer_params = {'win_size': 128, 'batch_size': 10, 'padding': 0,
#                     "img_path": 'W://ltd//anti//imgs_sim_128_2000_0315.tif', "result_name": 'result0317.csv',
#                     'net_path': "G://Dilatedloc//INFER//2023-03-16-checkpoint.pth.tar", 'gt_path': "gt.csv"}
#
#
#     settings = {'net_params': net_params, 'train_params': train_params, 'camera_params': camera_params,
#                 'psf_params': psf_params, 'data_params': data_params, 'eval_params': eval_params,
#                 'infer_params': infer_params}
#
#     return settings