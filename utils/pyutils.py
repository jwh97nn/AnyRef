import numpy as np
import time
import sys
import torch
import os

class Logger(object):
    def __init__(self, outfile):
        self.terminal = sys.stdout
        self.log = open(outfile, "w")
        sys.stdout = self

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()


class AverageMeter:
    def __init__(self, *keys):
        self.__data = dict()
        for k in keys:
            self.__data[k] = [0.0, 0]

    def add(self, dict):
        for k, v in dict.items():
            self.__data[k][0] += v
            self.__data[k][1] += 1

    def get(self, *keys):
        if len(keys) == 1:
            return self.__data[keys[0]][0] / self.__data[keys[0]][1]
        else:
            v_list = [self.__data[k][0] / self.__data[k][1] for k in keys]
            return tuple(v_list)

    def pop(self, key=None):
        if key is None:
            for k in self.__data.keys():
                self.__data[k] = [0.0, 0]
        else:
            v = self.get(key)
            self.__data[key] = [0.0, 0]
            return v


class Timer:
    def __init__(self, starting_msg = None):
        self.start = time.time()
        self.stage_start = self.start

        if starting_msg is not None:
            print(starting_msg, time.ctime(time.time()))


    def update_progress(self, progress):
        self.elapsed = time.time() - self.start
        self.est_total = self.elapsed / progress
        self.est_remaining = self.est_total - self.elapsed
        self.est_finish = int(self.start + self.est_total)


    def str_est_finish(self):
        return str(time.ctime(self.est_finish))

    def get_stage_elapsed(self):
        return time.time() - self.stage_start

    def reset_stage(self):
        self.stage_start = time.time()


from multiprocessing.pool import ThreadPool

class BatchThreader:

    def __init__(self, func, args_list, batch_size, prefetch_size=4, processes=12):
        self.batch_size = batch_size
        self.prefetch_size = prefetch_size

        self.pool = ThreadPool(processes=processes)
        self.async_result = []

        self.func = func
        self.left_args_list = args_list
        self.n_tasks = len(args_list)

        # initial work
        self.__start_works(self.__get_n_pending_works())


    def __start_works(self, times):
        for _ in range(times):
            args = self.left_args_list.pop(0)
            self.async_result.append(
                self.pool.apply_async(self.func, args))


    def __get_n_pending_works(self):
        return min((self.prefetch_size + 1) * self.batch_size - len(self.async_result)
                   , len(self.left_args_list))



    def pop_results(self):

        n_inwork = len(self.async_result)

        n_fetch = min(n_inwork, self.batch_size)
        rtn = [self.async_result.pop(0).get()
                for _ in range(n_fetch)]

        to_fill = self.__get_n_pending_works()
        if to_fill == 0:
            self.pool.close()
        else:
            self.__start_works(to_fill)

        return rtn




def get_indices_of_pairs(radius, size):

    search_dist = []

    for x in range(1, radius):
        search_dist.append((0, x))

    for y in range(1, radius):
        for x in range(-radius + 1, radius):
            if x * x + y * y < radius * radius:
                search_dist.append((y, x))

    radius_floor = radius - 1

    full_indices = np.reshape(np.arange(0, size[0]*size[1], dtype=np.int64),
                                   (size[0], size[1]))

    cropped_height = size[0] - radius_floor
    cropped_width = size[1] - 2 * radius_floor

    indices_from = np.reshape(full_indices[:-radius_floor, radius_floor:-radius_floor],
                              [-1])

    indices_to_list = []

    for dy, dx in search_dist:
        indices_to = full_indices[dy:dy + cropped_height,
                     radius_floor + dx:radius_floor + dx + cropped_width]
        indices_to = np.reshape(indices_to, [-1])

        indices_to_list.append(indices_to)

    concat_indices_to = np.concatenate(indices_to_list, axis=0)

    return indices_from, concat_indices_to


def mask_iou(pred, target, eps=1e-7, size_average=True):
    r"""
        param: 
            pred: size [N x H x W]
            target: size [N x H x W]
        output:
            iou: size [1] (size_average=True) or [N] (size_average=False)
    """
    assert len(pred.shape) == 3 and pred.shape == target.shape

    N = pred.size(0)
    num_pixels = pred.size(-1) * pred.size(-2)
    no_obj_flag = (target.sum(2).sum(1) == 0)

    temp_pred = torch.sigmoid(pred)
    pred = (temp_pred > 0.5).int()
    inter = (pred * target).sum(2).sum(1)
    union = torch.max(pred, target).sum(2).sum(1)

    inter_no_obj = ((1 - target) * (1 - pred)).sum(2).sum(1)
    inter[no_obj_flag] = inter_no_obj[no_obj_flag]
    union[no_obj_flag] = num_pixels

    iou = torch.sum(inter / (union+eps)) / N

    return iou


def Eval_Fmeasure(pred, gt, measure_path, pr_num=255):
    r"""
        param:
            pred: size [N x H x W]
            gt: size [N x H x W]
        output:
            iou: size [1] (size_average=True) or [N] (size_average=False)
    """
    print('=> eval [FMeasure]..')
    pred = torch.sigmoid(pred) # =======================================[important]
    N = pred.size(0)
    beta2 = 0.3
    avg_f, img_num = 0.0, 0
    score = torch.zeros(pr_num)
    fLog = open(os.path.join(measure_path, 'FMeasure.txt'), 'w')
    print("{} videos in this batch".format(N))

    for img_id in range(N):
        # examples with totally black GTs are out of consideration
        if torch.mean(gt[img_id]) == 0.0:
            continue
        prec, recall = _eval_pr(pred[img_id], gt[img_id], pr_num)   #, cuda_flag=False)
        f_score = (1 + beta2) * prec * recall / (beta2 * prec + recall)
        f_score[f_score != f_score] = 0 # for Nan
        avg_f += f_score
        img_num += 1
        score = avg_f / img_num
        # print('score: ', score)
    fLog.close()

    return score.max().item()


def _eval_pr(y_pred, y, num, cuda_flag=True):
    if cuda_flag:
        prec, recall = torch.zeros(num).cuda(), torch.zeros(num).cuda()
        thlist = torch.linspace(0, 1 - 1e-10, num).cuda()
    else:
        prec, recall = torch.zeros(num), torch.zeros(num)
        thlist = torch.linspace(0, 1 - 1e-10, num)
    for i in range(num):
        y_temp = (y_pred >= thlist[i]).float()
        tp = (y_temp * y).sum()
        prec[i], recall[i] = tp / (y_temp.sum() + 1e-20), tp / (y.sum() + 1e-20)

    return prec, recall