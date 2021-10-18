import abc
import os
import datetime
import json
import torch
import tensorboardX
import util


class BaseTrainer(object):
    def __init__(self, args):
        self.args = args
        self._log_path = os.path.join(args.log_path, args.log_label, str(datetime.datetime.now()))
        if args.log_path and not os.path.exists(args.log_path):
            os.makedirs(args.log_path)
        self._summary_writer = tensorboardX.SummaryWriter(self._log_path)

        # CUDA devices
        self._device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
        self._gpu_count = torch.cuda.device_count()
        # if args.seed is not None:
        #     util.set_seed(args.seed)
        self._log_args()

    def _log_args(self):
        with open(self._log_path+".args", mode='w+', encoding='utf-8') as fp:
            json.dump(vars(self.args), fp)

    def _log_tensorboard(self, dataset_label: str, data_label: str, data: object, iteration: int):
        self._summary_writer.add_scalar('data/%s/%s' % (dataset_label, data_label), data, iteration)
