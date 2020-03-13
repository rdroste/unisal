import datetime
from inspect import getfullargspec
import json
import copy
import sys
import importlib
import time

import torch
import torch.nn.functional as F
import numpy as np


def get_kwargs_names(func):
    args = getfullargspec(func).args
    try:
        args.remove('self')
    except ValueError:
        pass
    return args


def get_kwargs_dict(obj):
    return {key: obj.__dict__[key]
            for key in get_kwargs_names(obj.__init__)}


class KwConfigClass:

    def asdict(self):
        return get_kwargs_dict(self)

    @classmethod
    def init_from_cfg_dir(cls, directory, **kwargs):
        with open(directory / f"{cls.__name__}.json", 'r') as f:
            config = json.load(f)
        if 'new_instance' in config:
            config['new_instance'] = False
        config.update(kwargs)
        return cls(**config)

    def save_cfg(self, directory):
        with open(directory / f"{self.__class__.__name__}.json", 'w') as f:
            json.dump(self.asdict(), f)


def get_timestamp():
    return str(datetime.datetime.now())[:-7].replace(' ', '_')


def load_module(directory, module_name, full_name):
    path = copy.deepcopy(sys.path)
    sys.path.insert(0, str(directory))
    if module_name is not None and module_name in sys.modules:
        del sys.modules[module_name]
    if full_name in sys.modules:
        del sys.modules[full_name]
    module = importlib.import_module(full_name)
    sys.path = path
    return module


def load_trainer(directory, **kwargs):
    train = load_module(directory / 'code_copy', 'bds_net', 'bds_net.train')
    print(train)
    train = train.Trainer.init_from_cfg_dir(directory, **kwargs)
    return train


def load_model(directory, **kwargs):
    # model = load_module(directory, 'code_copy', 'code_copy.model')
    model = load_module(directory / 'code_copy', 'bds_net', 'bds_net.model')
    print(model)
    model_cls = model.get_model()
    model = model_cls.init_from_cfg_dir(directory, **kwargs)
    return model


def load_dataset(directory, **kwargs):
    # data = load_module(directory, 'code_copy', 'code_copy.data')
    data = load_module(directory / 'code_copy', 'bds_net', 'bds_net.data')
    print(data)
    dataset_cls = data.get_dataset()
    dataset = dataset_cls.init_from_cfg_dir(directory, **kwargs)
    return dataset


class Timer:
    def __init__(self, name='', info='', verbose=True):
        self.name = name
        self.verbose = verbose
        self.since = time.time()
        if name and self.verbose:
            print(name + ' ' + info + '...')

    def finish(self):
        time_elapsed = time.time() - self.since
        if self.verbose:
            print('{} completed in {:.0f}m {:.0f}s'.format(
                self.name, time_elapsed // 60, time_elapsed % 60))
        return time_elapsed


def to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def normalize_tensor(tensor, rescale=False):
    tmin = torch.min(tensor)
    if rescale or tmin < 0:
        tensor -= tmin
    tsum = tensor.sum()
    if tsum > 0:
        return tensor / tsum
    print("Zero tensor")
    tensor.fill_(1. / tensor.numel())
    return tensor


def normalize_array(array, rescale=False):
    amin = np.amin(array)
    if rescale or amin < 0:
        array -= amin
    asum = array.sum()
    if asum > 0:
        return array / asum
    print("Zero array")
    array.fill(1. / array.size())
    return array


def log_softmax(x):
    x_size = x.size()
    x = x.view(x.size(0), -1)
    x = F.log_softmax(x, dim=1)
    return x.view(x_size)


def nss(pred, fixations):
    size = pred.size()
    new_size = (-1, size[-1] * size[-2])
    pred = pred.reshape(new_size)
    fixations = fixations.reshape(new_size)

    pred_normed = (pred - pred.mean(-1, True)) / pred.std(-1, keepdim=True)
    results = []
    for this_pred_normed, mask in zip(torch.unbind(pred_normed, 0),
                                      torch.unbind(fixations, 0)):
        if mask.sum() == 0:
            print("No fixations.")
            results.append(torch.ones([]).float().to(fixations.device))
            continue
        nss_ = torch.masked_select(this_pred_normed, mask)
        nss_ = nss_.mean(-1)
        results.append(nss_)
    results = torch.stack(results)
    results = results.reshape(size[:2])
    return results


def corr_coeff(pred, target):
    size = pred.size()
    new_size = (-1, size[-1] * size[-2])
    pred = pred.reshape(new_size)
    target = target.reshape(new_size)

    cc = []
    for x, y in zip(torch.unbind(pred, 0), torch.unbind(target, 0)):
        xm, ym = x - x.mean(), y - y.mean()
        r_num = torch.mean(xm * ym)
        r_den = torch.sqrt(
            torch.mean(torch.pow(xm, 2)) * torch.mean(torch.pow(ym, 2)))
        r = r_num / r_den
        cc.append(r)

    cc = torch.stack(cc)
    cc = cc.reshape(size[:2])
    return cc  # 1 - torch.square(r)


def kld_loss(pred, target):
    loss = F.kl_div(pred, target, reduction='none')
    loss = loss.sum(-1).sum(-1).sum(-1)
    return loss


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16,
                            np.int32, np.int64, np.uint8, np.uint16, np.uint32,
                            np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj,(np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def smooth_sequence(seq, method):
    shape = seq.shape

    seq = seq.reshape(shape[1], np.prod(shape[-2:]))
    if method[:3] == 'med':
        kernel_size = int(method[3:])
        ks2 = kernel_size // 2
        smoothed = np.zeros_like(seq)
        for idx in range(seq.shape[0]):
            smoothed[idx, :] = np.median(
                seq[max(0, idx - ks2):min(seq.shape[0], idx + ks2 + 1), :],
                axis=0)
        seq = smoothed.reshape(shape)
    else:
        raise NotImplementedError

    return seq
