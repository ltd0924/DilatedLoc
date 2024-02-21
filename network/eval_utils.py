import time

import numpy as np
import copy
import torch

import matplotlib.pyplot as plt
import torch.nn.functional as func
from tqdm import tqdm
from utils.help_utils import *
from torch.cuda.amp import autocast as autocast
# calculates the jaccard coefficient approximation using per-voxel probabilities

class Eval:
    def __init__(self,eval_matrix,  que = None,  scale =[110, 110, 1500, 20000]):
        self.threshold=eval_matrix['threshold']
        self.min_int = eval_matrix['min_int']
        self.limited_x = eval_matrix['limited_x']
        self.limited_y = eval_matrix['limited_y']
        self.tolerance = eval_matrix['tolerance']
        self.tolerance_ax = eval_matrix['tolerance_ax']
        self.batch_size = eval_matrix['batch_size']
        self.candi_thre =eval_matrix['candi_thre']
        self.post_que = que
        self.scale = scale

    def inferlist_time(self):
        pred_list = {}
        post_time1 = 0
        post_time2 = 0
        post_time3 = 0
        num = 0
        for i in range(self.image_num):
            index, coord, Prob, xyzi_est = self.post_que.get(block = True)
            tmp, t_num,a, b,c = self.predlist_time(P=Prob, xyzi_est=xyzi_est, start=index, start_coord=coord)

            #print("image {0} cost time:{1}".format(i,c))
            post_time1 += a
            post_time2 += b
            post_time3 += c
            pred_list = {**pred_list, **tmp}
            num += t_num
        print('\nevaluation on {0} images, predict: {1}'.format(len(pred_list), num))
        print("post time {0}, {1}, {2}".format(post_time1, post_time2, post_time3))
        self.res = pred_list
    def predlist_time(self,p_nms, xyzi_est, start, start_coord=[0, 0]):
        #
        # aa = time.time()
        xyzi_est = cpu(xyzi_est)

        w, h  = xyzi_est.shape[-2], xyzi_est.shape[-1]

        xo = xyzi_est[:,0,:,:].reshape(-1,w,h)
        yo = xyzi_est[:,1, :, :].reshape(-1,w,h)
        zo = xyzi_est[:,2,:,:].reshape(-1,w,h)* self.scale[2]
        ints = xyzi_est[:,3,:,:].reshape(-1,w,h)* self.scale[3]


        # p_nms = self.nms_func(P, candi_thre=self.candi_thre)
        p_nms = cpu(p_nms)
        sample = np.where(p_nms > self.threshold, 1, 0)

        pred_list = {}
        num = 0
        cc = time.time()
        for i in range(len(xo)):
            pos = np.nonzero(sample[i])  # get the deterministic pixel position
            for j in range(len(pos[0])):
                x_tmp = (0.5 + pos[1][j] + float(start_coord[j][0]) + xo[i,pos[0][j], pos[1][j]]) * self.scale[0]
                y_tmp = (0.5 + pos[0][j] + float(start_coord[j][1]) + yo[i,pos[0][j], pos[1][j]]) * self.scale[1]
                if x_tmp < self.limited_x[0] or x_tmp > self.limited_x[1]:
                    continue
                if y_tmp < self.limited_y[0] or y_tmp > self.limited_y[1]:
                    continue
                k =i+start[j]+1
                if k not in pred_list:
                    pred_list[k] = []
                num += 1
                pred_list[k].append([k,x_tmp,
                                  y_tmp,
                                  zo[i,pos[0][j], pos[1][j]]  ,
                                  ints[i,pos[0][j], pos[1][j]],
                                  float(p_nms[i,pos[0][j], pos[1][j]])])
        #torch.cuda.synchronize()
        cc = time.time() -cc
        #
        return pred_list, num,  cc

    def predlist(self,P, xyzi_est, start, start_coord=[0, 0]):
        xyzi_est = cpu(xyzi_est)
        xo = xyzi_est[:,0,:,:].reshape([-1,xyzi_est.shape[-2],xyzi_est.shape[-1]])
        yo = xyzi_est[:,1, :, :].reshape([-1,xyzi_est.shape[-2],xyzi_est.shape[-1]])
        zo = xyzi_est[:,2,:,:].reshape([-1,xyzi_est.shape[-2],xyzi_est.shape[-1]])
        ints = xyzi_est[:,3,:,:].reshape([-1,xyzi_est.shape[-2],xyzi_est.shape[-1]])


        p_nms = cpu(P)


        sample = np.where(p_nms > self.threshold, 1, 0)

        pred_list = {}
        num = 0

        for i in range(len(xo)):
            pos = np.nonzero(sample[i])  # get the deterministic pixel position
            for j in range(len(pos[0])):
                x_tmp = (0.5 + pos[1][j] + float(start_coord[0]) + xo[i,pos[0][j], pos[1][j]]) * self.scale[0]
                y_tmp = (0.5 + pos[0][j] + float(start_coord[1]) + yo[i,pos[0][j], pos[1][j]]) * self.scale[1]
                if x_tmp < self.limited_x[0] or x_tmp > self.limited_x[1]:
                    continue
                if y_tmp < self.limited_y[0] or y_tmp > self.limited_y[1]:
                    continue
                k =i+start+1
                if k not in pred_list:
                    pred_list[k] = []
                num += 1
                pred_list[k].append([k,x_tmp,
                                  y_tmp,
                                  zo[i,pos[0][j], pos[1][j]]  * self.scale[2],
                                  ints[i,pos[0][j], pos[1][j]] * self.scale[3],
                                  float(p_nms[i,pos[0][j], pos[1][j]])])



        return pred_list, num







    def ShowRecovery3D(self,match):
        # define a figure for 3D scatter plot
        ax = plt.axes(projection='3d')

        # plot boolean recoveries in 3D
        ax.scatter(match[:, 0], match[:, 1], match[:, 2], c='b', marker='o', label='GT', depthshade=False)
        ax.scatter(match[:, 4], match[:, 5], match[:, 6], c='r', marker='^', label='Rec', depthshade=False)

        # add labels and and legend
        ax.set_xlabel('X [nm]')
        ax.set_ylabel('Y [nm]')
        ax.set_zlabel('Z [nm]')
        plt.legend()


    def inferlist_eval(self, pred_res):
        pred_list = {}
        num = 0
        post_time1 = 0
        post_time2 = 0
        post_time3 = 0
        for i in range(len(pred_res)):
            tmp, t_num, c = self.predlist_time(p_nms=pred_res["Prob"][i], xyzi_est=pred_res["preds"][i],
                                       start=pred_res["index"][i], start_coord=pred_res["coord"][i])
            pred_list = {**pred_list, **tmp}
            num += t_num

            post_time3 += c
        print('\nevaluation on {0} images, predict: {1}'.format(len(pred_list), num))

        print(post_time3)
        return pred_list



    def inferlist(self):
        # torch.cuda.synchronize()
        time_postprocess = time.time()
        pred_list = {}
        num = 0

        while 1:
            index, coord, Prob, xyzi_est = self.post_que.get(block=True)
            if index == -1:
                break
            tmp, t_num = self.predlist(P=Prob, xyzi_est=xyzi_est, start=index, start_coord=coord)
            pred_list = {**pred_list, **tmp}
            # print(num)
            num += t_num
        print('\nevaluation on {0} images, predict: {1}'.format(len(pred_list), num))
        self.res = pred_list
        torch.cuda.synchronize()
        time_postprocess = (time.time() - time_postprocess) * 1000
        print("time for post-process is " + str(time_postprocess) + "ms.")

