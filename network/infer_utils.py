import time

import torch
from tqdm import tqdm
from torch.cuda.amp import autocast as autocast
import numpy as np
from PSFLocModel import *
import queue
from threading import Thread
from utils.local_tifffile import *
from utils.record_utils import *
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from multiprocessing import Process

class InferDataset_set(Dataset):

    def __init__(self, infer_par, batch_size):
        self.tif_file = TiffFile(infer_par['img_path'], is_ome=True)
        self.total_shape = self.tif_file.series[0].shape
        self.win_size = infer_par['win_size']
        self.batch_size = batch_size
        self.padding = infer_par['padding']
        self.data_info = self.get_img_info(self.total_shape, self.win_size)
        self.slice_num = self.batch_size  # int(self.total_shape[0]/int(self.tif_file.fstat[6]/ 1024**3 + 1)/batch_size + 1) * batch_size
        self.start = 0
        self.end = self.slice_num
        self.tif_img = np.array(
            self.tif_file.asarray(key=slice(0, min(self.slice_num, self.total_shape[0])),
                                  series=0), dtype=np.float32)

        if self.total_shape[-1] == self.win_size:
            self.h_num = [0]
        else:
            self.h_num = list(range(0, self.total_shape[-1], self.win_size - self.padding))
        if self.total_shape[-2] == self.win_size:
            self.w_num = [0]
        else:
            self.w_num = list(range(0, self.total_shape[-2], self.win_size - self.padding))


    # sampling one example from the data

    def __len__(self):
        return int(np.ceil((len(self.w_num) * len(self.h_num) * self.total_shape[0])))

    def __getitem__(self, index):
        # select sample

        frame_index, fov_coord = self.data_info[index]
        if self.end == frame_index:
            self.tif_img = np.array(
                self.tif_file.asarray(key=slice(frame_index, min(frame_index + self.slice_num, self.total_shape[0])),
                                      series=0),
                dtype=np.float32)
            self.start = frame_index
            self.end = frame_index + self.slice_num
        c = frame_index%self.batch_size
        x = fov_coord[0]
        y = fov_coord[1]
        img = self.tif_img[c, x:x + self.win_size, y:y + self.win_size]

        return frame_index, np.array(fov_coord), img

    @staticmethod
    def get_img_info(total_shape, win_size):
        data_info = []
        for i in range(total_shape[0]):
            for j in range(int(np.ceil(total_shape[-1] / win_size))):
                for k in range(int(np.ceil(total_shape[-2] / win_size))):
                    data_info.append((i, [j * win_size, k * win_size]))
        return data_info



class PsfInfer_dis:
    def __init__(self,infer_par,dataloader,device):
        self.win_size = infer_par['win_size']
        self.padding = infer_par['padding']
        self.result_name = infer_par['result_name']

        # self.queue = queue
        # self.post_que = que
        self.model = LocalizationCNN(True, local_context=infer_par['local_flag'], feature=64)
        # self.model = LocalizationCNN(True)

        self.loadmodel(infer_par['net_path'])
        # self.model = torch.nn.DataParallel(self.model, device_ids=list(range(torch.cuda.device_count())))
        # print("current GPU nums:{0}",format(torch.cuda.device_count()))
        self.device = device
        self.model = self.model.to(self.device)

        print("current GPU nums:{0}".format(torch.cuda.device_count()))
        # self.device = torch.device(device)
        # self.model = self.model.to(self.device)
        self.train_loader = dataloader

        self.start_ind = check_csv(self.result_name)

    def loadmodel(self, path):
        with open(path, 'rb') as f:
            obj = f.read()
        checkpoint = pickle.loads(obj, encoding='latin1')
        self.model.load_state_dict(checkpoint['state_dict'])
        print(checkpoint['epoch'])
        print("=> loaded checkpoint")

    def inferdata(self):
        time_forward = 0
        self.model.eval()
        with torch.set_grad_enabled(False):
            with autocast():
                for _, (index, coord, img) in enumerate(tqdm(self.train_loader)):
                    # print(self.queue.qsize())
                    img = img.reshape([-1, 1, self.win_size, self.win_size])
                    img = torch.Tensor(img).float().to(self.device)

                    # time_inference = time.time()
                    P, xyzi_est, _ = self.model(img, test=True)
                    # torch.cuda.synchronize()

                    # time_forward = time_forward + (time.time() - time_inference) * 1000
                    # self.post_que.put([index, coord, P, xyzi_est])
                    # # # print("Outside: input size", img.size(),
                    # # #       "output_size", xyzi_est.size())
                    # self.queue.task_done()

        print("time for network forward is " + str(time_forward) + "ms.")


    def inferdata_No(self):
        res = {
            "index": [],
            "coord": [],
            "Prob": [],
            "preds": []
        }
        time_forward = 0
        self.model.eval()
        with torch.set_grad_enabled(False):
            with autocast():
                for _, (index, coord, img) in enumerate(tqdm(self.train_loader)):
                    # print(self.queue.qsize())
                    img = img.reshape([-1, 1, self.win_size, self.win_size])
                    img = torch.Tensor(img).float().to(self.device)
                    #self.post_que.put([index, coord, P, xyzi_est])
                    # torch.cuda.synchronize()
                    P, xyzi_est, _ = self.model(img, test=True)
                    res["index"].append(index)
                    res["coord"].append(coord)
                    res["Prob"].append(P)
                    res["preds"].append(xyzi_est)
                    # self.queue.task_done()
                print("time forward:{0}".format(time_forward))
                self.res = res

        print("time for network forward is " + str(time_forward) + "ms.")




def sample_generator(infer_par, queue, batch_size):
    tif_file = TiffFile(infer_par['img_path'], is_ome=True)
    total_shape = tif_file.series[0].shape
    win_size = infer_par['win_size']
    padding = infer_par['padding']
    slice_num = batch_size  # int(self.total_shape[0]/int(self.tif_file.fstat[6]/ 1024**3 + 1)/batch_size + 1) * batch_size
    startnum = 0
    endnum = slice_num
    tif_img = np.array(
        tif_file.asarray(key=slice(0, min(slice_num, total_shape[0])),
                              series=0), dtype=np.float32)

    if total_shape[-1] == win_size:
        h_num = [0]
    else:
        h_num = list(range(0, total_shape[-1], win_size - padding))
    if total_shape[-2] == win_size:
        w_num = [0]
    else:
        w_num = list(range(0, total_shape[-2], win_size - padding))
    for i in range(1):
        for frame_index in range(0, total_shape[0], batch_size):
            if frame_index >= endnum:
                # torch.cuda.empty_cache()
                tif_img = np.array(
                    tif_file.asarray(
                        key=slice(frame_index, min(frame_index + slice_num, total_shape[0])), series=0),
                    dtype=np.float32)
                startnum = frame_index
                endnum = frame_index + slice_num
            #
            for x in w_num:
                for y in h_num:
                    frame_c = frame_index
                    frame_index -= startnum
                    if len(tif_img.shape) == 2:
                        img = tif_img
                    else:
                        img = tif_img[frame_index:min(len(tif_img), frame_index + batch_size),
                              x:x + win_size, y:y + win_size]
                    queue.put([frame_c, [x, y], img])
        startnum = 0
        endnum = slice_num
        tif_img = np.array(
            tif_file.asarray(key=slice(0, min(slice_num, total_shape[0])),
                             series=0), dtype=np.float32)
    for i in range(torch.cuda.device_count()):
        queue.put([-1,-1,-1])


class InferDataset():
    # initialization of the dataset
    def __init__(self, infer_par, q, batch_size):
        self.tif_file = TiffFile(infer_par['img_path'], is_ome=True)
        self.total_shape = self.tif_file.series[0].shape
        self.win_size = infer_par['win_size']
        self.batch_size = batch_size
        self.padding = infer_par['padding']
        self.slice_num = self.batch_size  # int(self.total_shape[0]/int(self.tif_file.fstat[6]/ 1024**3 + 1)/batch_size + 1) * batch_size
        self.start = 0
        self.queue = q
        self.end = self.slice_num
        self.tif_img = np.array(
            self.tif_file.asarray(key=slice(0, min(self.slice_num, self.total_shape[0])),
                                  series=0), dtype=np.float32)

        if self.total_shape[-1] == self.win_size:
            self.h_num = [0]
        else:
            self.h_num = list(range(0,self.total_shape[-1],self.win_size-self.padding))
        if self.total_shape[-2] == self.win_size:
            self.w_num = [0]
        else:
            self.w_num = list(range(0,self.total_shape[-2],self.win_size-self.padding))

    # sampling one example from the data

    def __len__(self):
        return int(np.ceil((len(self.w_num)* len(self.h_num) * self.total_shape[0])/self.batch_size))


    def sample_generator(self):
        #self.slice_num = self.batch_size
        # self.queue = q
        for i in range(10):
            for frame_index in range(0, self.total_shape[0], self.batch_size):
                if frame_index >= self.end:
                    #torch.cuda.empty_cache()
                    self.tif_img = np.array(
                        self.tif_file.asarray(key=slice(frame_index, min(frame_index + self.slice_num,self.total_shape[0])), series=0),
                        dtype=np.float32)
                    self.start = frame_index
                    self.end = frame_index + self.slice_num
                #
                for x in self.w_num:
                    for y in self.h_num:
                        frame_c = frame_index
                        frame_index -= self.start
                        if len(self.tif_img.shape)==2:
                            img = self.tif_img
                        else:
                            img = self.tif_img[frame_index:min(len(self.tif_img),frame_index+self.batch_size),x:x+self.win_size, y:y+self.win_size]
                        self.queue.put([frame_c, [x,y], img])
            self.start = 0
            self.end = self.slice_num
            self.tif_img = np.array(
                self.tif_file.asarray(key=slice(0, min(self.slice_num, self.total_shape[0])),
                                      series=0), dtype=np.float32)

class PsfInfer:
    def __init__(self,infer_par,queue, que, device):
        self.win_size = infer_par['win_size']
        self.padding = infer_par['padding']
        self.queue = queue
        self.post_que = que
        self.model = LocalizationCNN(True, local_context=infer_par['local_flag'], feature=64)
        self.loadmodel(infer_par['net_path'])
        self.device = device
        self.model = self.model.to(device)

    def loadmodel(self,path):
        with open(path, 'rb') as f:
            obj = f.read()
        checkpoint = pickle.loads(obj, encoding='latin1')
        self.model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint")

    def inferdata(self):
        time_forward = 0
        self.model.eval()

        with torch.set_grad_enabled(False):
            with autocast():
                while 1:
                    index, coord, img = self.queue.get(block=True)
                    if index == -1:
                        break
                    img = img.reshape([-1, 1, self.win_size, self.win_size])
                    img = torch.Tensor(img).float().to(self.device)
                    time_inference = time.time()
                    P, xyzi_est, _ = self.model(img, test=True)
                    torch.cuda.synchronize()
                    time_forward = time_forward + (time.time() - time_inference) * 1000
                    self.post_que.put([index, coord, P, xyzi_est])

        self.post_que.put([-1,-1,-1,-1])
        print("time for network forward is " + str(time_forward) + "ms.")


    def inferdata_No(self):
        res = {
            "index": [],
            "coord": [],
            "Prob": [],
            "preds": []
        }
        time_forward = 0
        self.model.eval()
        with torch.set_grad_enabled(False):
            with autocast():
                while 1:
                    index, coord, img = self.queue.get(block=True)
                    if index == -1:
                        break
                    img = img.reshape([-1, 1, self.win_size, self.win_size])
                    img = torch.Tensor(img).float().to(self.device)
                    P, xyzi_est, _ = self.model(img, test=True)
                    res["index"].append(index)
                    res["coord"].append(coord)
                    res["Prob"].append(P)
                    res["preds"].append(xyzi_est)
                self.res = res




