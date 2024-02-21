import pickle
import numpy as np
import torch

from tqdm import trange
from torch.cuda.amp import autocast as autocast

from utils.local_tifffile import *
from PSFLocModel import *
from utils.parameter_setting import *
from network.eval_utils import *
from utils.record_utils import *
from network.infer_utils import *
import sys
from utils.parameter_setting import *
from PSFLocModel import *
from utils.visual_utils import *
from utils.local_tifffile import *
import queue

from threading import Thread
from multiprocessing import Process,Queue

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"




def predict(device, q, setup_params):




    scale_factor =[100, 100, 1, 1]  # pixel x, pixel y, z scale, ph scale
    scale_factor[3] = setup_params['psf_params']['ph_scale']
    scale_factor[2] = setup_params['psf_params']['z_scale']


    time_sum = time.time()
    post_q = queue.Queue(maxsize=150)


    while q.qsize() == 0:
        continue
    psfinfer = PsfInfer(setup_params['infer_params'], q, post_q, device=device)
    psfeval = Eval(setup_params['eval_params'],  post_q, scale_factor)

    try:
        t1 = Thread(target=psfinfer.inferdata)
        t1.start()
        t2 = Thread(target=psfeval.inferlist)
        t2.start()

    except:
        print("error")
    finally:
        t1.join()
        t2.join()

    write_dict_csv(psfeval.res, setup_params['infer_params']['result_name'])
    time_sum = (time.time() - time_sum) * 1000
    print("time for inference is " + str(time_sum) + "ms. \n")




if __name__ == '__main__':

    setup_params = parameters_set1()
    q = Queue(maxsize=50)

    process_list=[]
    check_csv(setup_params['infer_params']['result_name'])
    try:

        p = Process(target = sample_generator, args=(setup_params['infer_params'], q,setup_params['eval_params']['batch_size'],))
        p.start()
        for i in range(torch.cuda.device_count()):
            process_list.append(Process(target=predict, args=("cuda:{0}".format(i), q, setup_params,)))
        for x in process_list:
            x.start()

    except:
        print("error")
    finally:
        p.join()
        for x in process_list:
            x.join()

    # both processes finished
    if setup_params['infer_params']['gt_path'] != "":
        c = time.time()
        perf_dict, pred_lists = assess(setup_params['infer_params']['result_name'],setup_params['infer_params']['gt_path'])
        print("access time: {0}".format(time.time() - c))

        for k, v in perf_dict.items():
            print('{} : {:0.3f}'.format(k, v))



    print("Done!")
