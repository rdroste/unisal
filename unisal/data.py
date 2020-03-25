
from pathlib import Path
import os
import random
import json
import itertools
import copy

import torch
from torch.utils.data import Dataset, DataLoader, BatchSampler, RandomSampler, \
    SequentialSampler
from torchvision import transforms
import numpy as np
import cv2
import PIL
import scipy.io

from . import utils

default_data_dir = Path(__file__).resolve().parent.parent / "data"

# Set default paths
if "DHF1K_DATA_DIR" not in os.environ:
    os.environ["DHF1K_DATA_DIR"] = str(default_data_dir / "DHF1K")
if "SALICON_DATA_DIR" not in os.environ:
    os.environ["SALICON_DATA_DIR"] = str(default_data_dir / "SALICON")
if "HOLLYWOOD_DATA_DIR" not in os.environ:
    os.environ["HOLLYWOOD_DATA_DIR"] = str(
        default_data_dir / "Hollywood2_actions")
if "UCFSPORTS_DATA_DIR" not in os.environ:
    os.environ["UCFSPORTS_DATA_DIR"] = str(default_data_dir / "ucf-002")
if "MIT300_DATA_DIR" not in os.environ:
    os.environ["MIT300_DATA_DIR"] = str(default_data_dir / "MIT300")
if "MIT1003_DATA_DIR" not in os.environ:
    os.environ["MIT1003_DATA_DIR"] = str(default_data_dir / "MIT1003")
config_path = Path(__file__).resolve().parent / "cache"


def get_dataset():
    return DHF1KDataset


def get_dataloader(src='DHF1K'):
    if src in ('MIT1003',):
        return ImgSizeDataLoader
    return DataLoader


class SALICONDataset(Dataset, utils.KwConfigClass):

    source = 'SALICON'
    dynamic = False

    def __init__(self, phase='train', subset=None, verbose=1,
                 out_size=(288, 384), target_size=(480, 640),
                 preproc_cfg=None):
        self.phase = phase
        self.train = phase == 'train'
        self.subset = subset
        self.verbose = verbose
        self.out_size = out_size
        self.target_size = target_size
        self.preproc_cfg = {
            'rgb_mean': (0.485, 0.456, 0.406),
            'rgb_std': (0.229, 0.224, 0.225),
        }
        if preproc_cfg is not None:
            self.preproc_cfg.update(preproc_cfg)
        self.phase_str = 'val' if phase in ('valid', 'eval') else phase
        self.file_stem = f"COCO_{self.phase_str}2014_"
        self.file_nr = "{:012d}"

        self.samples = self.prepare_samples()
        if self.subset is not None:
            self.samples = self.samples[:int(len(self.samples) * subset)]
        # For compatibility with video datasets
        self.n_images_dict = {img_nr: 1 for img_nr in self.samples}
        self.target_size_dict = {
            img_nr: self.target_size for img_nr in self.samples}
        self.n_samples = len(self.samples)
        self.frame_modulo = 1

    def get_map(self, img_nr):
        map_file = self.dir / 'maps' / self.phase_str / (
                self.file_stem + self.file_nr.format(img_nr) + '.png')
        map = cv2.imread(str(map_file), cv2.IMREAD_GRAYSCALE)
        assert(map is not None)
        return map

    def get_img(self, img_nr):
        img_file = self.dir / 'images' / (
                self.file_stem + self.file_nr.format(img_nr) + '.jpg')
        img = cv2.imread(str(img_file))
        assert(img is not None)
        return np.ascontiguousarray(img[:, :, ::-1])

    def get_raw_fixations(self, img_nr):
        raw_fix_file = self.dir / 'fixations' / self.phase_str / (
                self.file_stem + self.file_nr.format(img_nr) + '.mat')
        fix_data = scipy.io.loadmat(raw_fix_file)
        fixations_array = [gaze[2] for gaze in fix_data['gaze'][:, 0]]
        return fixations_array, fix_data['resolution'].tolist()[0]

    def process_raw_fixations(self, fixations_array, res):
        fix_map = np.zeros(res, dtype=np.uint8)
        for subject_fixations in fixations_array:
            fix_map[subject_fixations[:, 1] - 1, subject_fixations[:, 0] - 1]\
                = 255
        return fix_map

    def get_fixation_map(self, img_nr):
        fix_map_file = self.dir / 'fixations' / self.phase_str / (
                self.file_stem + self.file_nr.format(img_nr) + '.png')
        if fix_map_file.exists():
            fix_map = cv2.imread(str(fix_map_file), cv2.IMREAD_GRAYSCALE)
        else:
            fixations_array, res = self.get_raw_fixations(img_nr)
            fix_map = self.process_raw_fixations(fixations_array, res)
            cv2.imwrite(str(fix_map_file), fix_map)
        return fix_map

    @property
    def dir(self):
        return Path(os.environ["SALICON_DATA_DIR"])

    def prepare_samples(self):
        samples = []
        for file in (self.dir / 'images').glob(self.file_stem + '*.jpg'):
            samples.append(int(file.stem[-12:]))
        return sorted(samples)

    def __len__(self):
        return len(self.samples)

    def preprocess(self, img, data='img'):
        transformations = [
            transforms.ToPILImage(),
        ]
        if data == 'img':
            transformations.append(transforms.Resize(
                self.out_size, interpolation=PIL.Image.LANCZOS))
        transformations.append(transforms.ToTensor())
        if data == 'img' and 'rgb_mean' in self.preproc_cfg:
            transformations.append(
                transforms.Normalize(
                    self.preproc_cfg['rgb_mean'], self.preproc_cfg['rgb_std']))
        elif data == 'sal':
            transformations.append(transforms.Lambda(utils.normalize_tensor))
        elif data == 'fix':
            transformations.append(
                transforms.Lambda(lambda fix: torch.gt(fix, 0.5)))

        processing = transforms.Compose(transformations)
        tensor = processing(img)
        return tensor

    def get_data(self, img_nr):
        img = self.get_img(img_nr)
        img = self.preprocess(img, data='img')
        if self.phase == 'test':
            return [1], img, self.target_size

        sal = self.get_map(img_nr)
        sal = self.preprocess(sal, data='sal')
        fix = self.get_fixation_map(img_nr)
        fix = self.preprocess(fix, data='fix')

        return [1], img, sal, fix, self.target_size

    def __getitem__(self, item):
        img_nr = self.samples[item]
        return self.get_data(img_nr)


class ImgSizeBatchSampler:

    def __init__(self, dataset, batch_size=1, shuffle=False, drop_last=False):
        assert(isinstance(dataset, MIT1003Dataset))
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        out_size_array = [
            dataset.size_dict[img_idx]['out_size']
            for img_idx in dataset.samples]
        self.out_size_set = sorted(list(set(out_size_array)))
        self.sample_idx_dict = {
            out_size: [] for out_size in self.out_size_set}
        for sample_idx, img_idx in enumerate(dataset.samples):
            self.sample_idx_dict[dataset.size_dict[img_idx]['out_size']].append(
                sample_idx)

        self.len = 0
        self.n_batches_dict = {}
        for out_size, sample_idx_array in self.sample_idx_dict.items():
            this_n_batches = len(sample_idx_array) // self.batch_size
            self.len += this_n_batches
            self.n_batches_dict[out_size] = this_n_batches

    def __iter__(self):
        batch_array = list(itertools.chain.from_iterable(
            [out_size for _ in range(n_batches)]
            for out_size, n_batches in self.n_batches_dict.items()))
        if not self.shuffle:
            random.seed(27)
        random.shuffle(batch_array)

        this_sample_idx_dict = copy.deepcopy(self.sample_idx_dict)
        for sample_idx_array in this_sample_idx_dict.values():
            random.shuffle(sample_idx_array)
        for out_size in batch_array:
            this_indices = this_sample_idx_dict[out_size][:self.batch_size]
            del this_sample_idx_dict[out_size][:self.batch_size]
            yield this_indices

    def __len__(self):
        return self.len


class ImgSizeDataLoader(DataLoader):

    def __init__(self, dataset, batch_size=1, shuffle=False, drop_last=False,
                 **kwargs):
        if batch_size == 1:
            if shuffle:
                sampler = RandomSampler(dataset)
            else:
                sampler = SequentialSampler(dataset)
            batch_sampler = BatchSampler(sampler, batch_size, drop_last)
        else:
            batch_sampler = ImgSizeBatchSampler(
                dataset, batch_size=batch_size, shuffle=shuffle,
                drop_last=drop_last)
        super().__init__(dataset, batch_sampler=batch_sampler, **kwargs)


class MIT300Dataset(Dataset, utils.KwConfigClass):

    source = 'MIT300'
    dynamic = False

    def __init__(self, phase='test'):
        assert(phase == 'test')
        self.phase = phase
        self.train = False
        self.target_size = None
        self.preproc_cfg = {
            'rgb_mean': (0.485, 0.456, 0.406),
            'rgb_std': (0.229, 0.224, 0.225),
        }
        self.samples, self.target_size_dict = self.load_data()

    def load_data(self):
        samples = []
        target_size_dict = {}
        file_list = list(self.dir.glob('*.jpg'))
        file_list = sorted(file_list, key=lambda x: int(x.stem[1:min(4, len(x.stem))]))
        for img_idx, file in enumerate(file_list):

            img = cv2.imread(str(file))
            img_size = img.shape[:2]
            ar = img_size[0] / img_size[1]

            min_prod = 100
            max_prod = 120
            ar_array = []
            size_array = []
            for n1 in range(7, 14):
                for n2 in range(7, 14):
                    if min_prod <= n1 * n2 <= max_prod:
                        this_ar = n1 / n2
                        this_ar_ratio = min((ar, this_ar)) / max((ar, this_ar))
                        ar_array.append(this_ar_ratio)
                        size_array.append((n1, n2))

            max_ar_ratio_idx = np.argmax(np.array(ar_array)).item()
            bn_size = size_array[max_ar_ratio_idx]
            out_size = tuple(r * 32 for r in bn_size)
            samples.append((file.name, out_size))
            target_size_dict[img_idx] = img_size

        return samples, target_size_dict

    @property
    def dir(self):
        return Path(os.environ["MIT300_DATA_DIR"]) / 'BenchmarkIMAGES'

    def __len__(self):
        return len(self.samples)

    def preprocess(self, img, out_size, data='img'):
        assert(data == 'img')

        transformations = [
            transforms.ToPILImage(),
            transforms.Resize(
                out_size, interpolation=PIL.Image.LANCZOS),
            transforms.ToTensor(),
        ]
        if 'rgb_mean' in self.preproc_cfg:
            transformations.append(
                transforms.Normalize(
                    self.preproc_cfg['rgb_mean'], self.preproc_cfg['rgb_std']))

        processing = transforms.Compose(transformations)
        tensor = processing(img)
        return tensor

    def get_data(self, item):
        img_name, out_size = self.samples[item]
        img_file = self.dir / img_name
        img = cv2.imread(str(img_file))
        assert(img is not None)
        img = np.ascontiguousarray(img[:, :, ::-1])
        img = self.preprocess(img, out_size, data='img')
        return [1], img, self.target_size_dict[item]

    def __getitem__(self, item):
        return self.get_data(item)


class MIT1003Dataset(Dataset, utils.KwConfigClass):

    source = 'MIT1003'
    n_train_val_images = 1003
    dynamic = False

    def __init__(self, phase='train', subset=None, verbose=1,
                 preproc_cfg=None, n_x_val=10, x_val_step=0, x_val_seed=27):
        self.phase = phase
        self.train = phase == 'train'
        self.subset = subset
        self.verbose = verbose
        self.preproc_cfg = {
            'rgb_mean': (0.485, 0.456, 0.406),
            'rgb_std': (0.229, 0.224, 0.225),
        }
        if preproc_cfg is not None:
            self.preproc_cfg.update(preproc_cfg)

        self.n_x_val = n_x_val
        self.x_val_step = x_val_step
        self.x_val_seed = x_val_seed

        # Cross-validation split
        n_images = self.n_train_val_images
        if x_val_step is None:
            self.samples = np.arange(0, n_images)
        else:
            print(f"X-Val step: {x_val_step}")
            assert(self.x_val_step < self.n_x_val)
            samples = np.arange(0, n_images)
            if self.x_val_seed > 0:
                np.random.seed(self.x_val_seed)
                np.random.shuffle(samples)
            val_start = int(len(samples) / self.n_x_val * self.x_val_step)
            val_end = int(len(samples) / self.n_x_val * (self.x_val_step + 1))
            samples = samples.tolist()
            if not self.train:
                self.samples = samples[val_start:val_end]
            else:
                del samples[val_start:val_end]
                self.samples = samples

        self.all_image_files, self.size_dict = self.load_data()
        if self.subset is not None:
            self.samples = self.samples[:int(len(self.samples) * subset)]
        # For compatibility with video datasets
        self.n_images_dict = {sample: 1 for sample in self.samples}
        self.target_size_dict = {
            img_idx: self.size_dict[img_idx]['target_size']
            for img_idx in self.samples}
        self.n_samples = len(self.samples)
        self.frame_modulo = 1

    def get_map(self, img_idx):
        map_file = self.fix_dir / self.all_image_files[img_idx]['map']
        map = cv2.imread(str(map_file), cv2.IMREAD_GRAYSCALE)
        assert(map is not None)
        return map

    def get_img(self, img_idx):
        img_file = self.img_dir / self.all_image_files[img_idx]['img']
        img = cv2.imread(str(img_file))
        assert(img is not None)
        return np.ascontiguousarray(img[:, :, ::-1])

    def get_fixation_map(self, img_idx):
        fix_map_file = self.fix_dir / self.all_image_files[img_idx]['pts']
        fix_map = cv2.imread(str(fix_map_file), cv2.IMREAD_GRAYSCALE)
        assert(fix_map is not None)
        return fix_map

    @property
    def dir(self):
        return Path(os.environ["MIT1003_DATA_DIR"])

    @property
    def fix_dir(self):
        return self.dir / 'ALLFIXATIONMAPS' / 'ALLFIXATIONMAPS'

    @property
    def img_dir(self):
        return self.dir / 'ALLSTIMULI' / 'ALLSTIMULI'

    def get_out_size_eval(self, img_size):
        ar = img_size[0] / img_size[1]

        min_prod = 100
        max_prod = 120
        ar_array = []
        size_array = []
        for n1 in range(7, 14):
            for n2 in range(7, 14):
                if min_prod <= n1 * n2 <= max_prod:
                    this_ar = n1 / n2
                    this_ar_ratio = min((ar, this_ar)) / max((ar, this_ar))
                    ar_array.append(this_ar_ratio)
                    size_array.append((n1, n2))

        max_ar_ratio_idx = np.argmax(np.array(ar_array)).item()
        bn_size = size_array[max_ar_ratio_idx]
        out_size = tuple(r * 32 for r in bn_size)

        return out_size

    def get_out_size_train(self, img_size):
        selection = (8, 13), (9, 13), (9, 12), (12, 9), (13, 9)
        ar = img_size[0] / img_size[1]
        ar_array = []
        size_array = []
        for n1, n2 in selection:
            this_ar = n1 / n2
            this_ar_ratio = min((ar, this_ar)) / max((ar, this_ar))
            ar_array.append(this_ar_ratio)
            size_array.append((n1, n2))

        max_ar_ratio_idx = np.argmax(np.array(ar_array)).item()
        bn_size = size_array[max_ar_ratio_idx]
        out_size = tuple(r * 32 for r in bn_size)

        return out_size

    def load_data(self):

        all_image_files = []
        for img_file in sorted(self.img_dir.glob("*.jpeg")):
            all_image_files.append({
                'img': img_file.name,
                'map': img_file.stem + "_fixMap.jpg",
                'pts': img_file.stem + "_fixPts.jpg",
            })
            assert((self.fix_dir / all_image_files[-1]['map']).exists())
            assert((self.fix_dir / all_image_files[-1]['pts']).exists())

        size_dict_file = config_path / "img_size_dict.json"
        if size_dict_file.exists():
            with open(size_dict_file, 'r') as f:
                size_dict = json.load(f)
                size_dict = {int(img_idx): val for
                                  img_idx, val in size_dict.items()}

        else:
            size_dict = {}
            for img_idx in range(self.n_train_val_images):
                img = cv2.imread(
                    str(self.img_dir / all_image_files[img_idx]['img']))
                size_dict[img_idx] = {'img_size': img.shape[:2]}
            with open(size_dict_file, 'w') as f:
                json.dump(size_dict, f)

        for img_idx in self.samples:
            img_size = size_dict[img_idx]['img_size']
            if self.phase in ('train', 'valid'):
                out_size = self.get_out_size_train(img_size)
            else:
                out_size = self.get_out_size_eval(img_size)
            if self.phase in ('train', 'valid'):
                target_size = tuple(sz * 2 for sz in out_size)
            else:
                target_size = img_size

            size_dict[img_idx].update({
                'out_size': out_size, 'target_size': target_size})

        return all_image_files, size_dict

    def __len__(self):
        return len(self.samples)

    def preprocess(self, img, out_size=None, data='img'):
        transformations = [
            transforms.ToPILImage(),
        ]
        if data in ('img', 'sal'):
            transformations.append(transforms.Resize(
                out_size, interpolation=PIL.Image.LANCZOS))
        else:
            transformations.append(transforms.Resize(
                out_size, interpolation=PIL.Image.NEAREST))
        transformations.append(transforms.ToTensor())
        if data == 'img' and 'rgb_mean' in self.preproc_cfg:
            transformations.append(
                transforms.Normalize(
                    self.preproc_cfg['rgb_mean'], self.preproc_cfg['rgb_std']))
        elif data == 'sal':
            transformations.append(transforms.Lambda(utils.normalize_tensor))
        elif data == 'fix':
            transformations.append(
                transforms.Lambda(lambda fix: torch.gt(fix, 0.5)))

        processing = transforms.Compose(transformations)
        tensor = processing(img)
        return tensor

    def get_data(self, img_idx):
        img = self.get_img(img_idx)
        out_size = self.size_dict[img_idx]['out_size']
        target_size = self.target_size_dict[img_idx]
        img = self.preprocess(img, out_size=out_size, data='img')
        if self.phase == 'test':
            return [1], img, target_size

        sal = self.get_map(img_idx)
        sal = self.preprocess(sal, target_size, data='sal')
        fix = self.get_fixation_map(img_idx)
        fix = self.preprocess(fix, target_size, data='fix')

        return [1], img, sal, fix, target_size

    def __getitem__(self, item):
        img_idx = self.samples[item]
        return self.get_data(img_idx)


class DHF1KDataset(Dataset, utils.KwConfigClass):

    img_channels = 1
    n_train_val_videos = 700
    test_vid_nrs = (701, 1000)
    frame_rate = 30
    source = 'DHF1K'
    dynamic = True

    def __init__(self,
                 seq_len=12,
                 frame_modulo=5,
                 max_seq_len=1e6,
                 preproc_cfg=None,
                 out_size=(224, 384), phase='train', target_size=(360, 640),
                 debug=False, val_size=100, n_x_val=3, x_val_step=2,
                 x_val_seed=0, seq_per_vid=1, subset=None, verbose=1,
                 n_images_file='dhf1k_n_images.dat', seq_per_vid_val=2,
                 sal_offset=None):
        self.phase = phase
        self.train = phase == 'train'
        if not self.train:
            preproc_cfg = {}
        elif preproc_cfg is None:
            preproc_cfg = {}
        preproc_cfg.update({
            'rgb_mean': (0.485, 0.456, 0.406),
            'rgb_std': (0.229, 0.224, 0.225),
        })
        self.preproc_cfg = preproc_cfg
        self.out_size = out_size
        self.debug = debug
        self.val_size = val_size
        self.n_x_val = n_x_val
        self.x_val_step = x_val_step
        self.x_val_seed = x_val_seed
        self.seq_len = seq_len
        self.seq_per_vid = seq_per_vid
        self.seq_per_vid_val = seq_per_vid_val
        self.frame_modulo = frame_modulo
        self.clip_len = seq_len * frame_modulo
        self.subset = subset
        self.verbose = verbose
        self.n_images_file = n_images_file
        self.target_size = target_size
        self.sal_offset = sal_offset
        self.max_seq_len = max_seq_len
        self._dir = None
        self._n_images_dict = None
        self.vid_nr_array = None

        # Evaluation
        if phase in ('eval', 'test'):
            self.seq_len = int(1e6)

        if self.phase in ('test',):
            self.vid_nr_array = list(range(
                self.test_vid_nrs[0], self.test_vid_nrs[1] + 1))
            self.samples, self.target_size_dict = self.prepare_samples()
            return

        # Cross-validation split
        n_videos = self.n_train_val_videos
        assert(self.val_size <= n_videos // self.n_x_val)
        assert(self.x_val_step < self.n_x_val)
        vid_nr_array = np.arange(1, n_videos + 1)
        if self.x_val_seed > 0:
            np.random.seed(self.x_val_seed)
            np.random.shuffle(vid_nr_array)
        val_start = (len(vid_nr_array) - self.val_size) //\
                    (self.n_x_val - 1) * self.x_val_step
        vid_nr_array = vid_nr_array.tolist()
        if not self.train:
            self.vid_nr_array =\
                vid_nr_array[val_start:val_start + self.val_size]
        else:
            del vid_nr_array[val_start:val_start + self.val_size]
            self.vid_nr_array = vid_nr_array

        if self.subset is not None:
            self.vid_nr_array =\
                self.vid_nr_array[:int(len(self.vid_nr_array) * self.subset)]

        self.samples, self.target_size_dict = self.prepare_samples()

    @property
    def n_images_dict(self):
        if self._n_images_dict is None:
            with open(config_path.parent / self.n_images_file, 'r') as f:
                self._n_images_dict = {
                    idx + 1: int(line) for idx, line in enumerate(f)
                    if idx + 1 in self.vid_nr_array}
        return self._n_images_dict

    @property
    def dir(self):
        if self._dir is None:
            self._dir = Path(os.environ["DHF1K_DATA_DIR"])
        return self._dir

    @property
    def n_samples(self):
        return len(self.vid_nr_array)

    def __len__(self):
        return len(self.samples)

    def prepare_samples(self):
        samples = []
        too_short = 0
        too_long = 0
        for vid_nr, n_images in self.n_images_dict.items():
            if self.phase in ('eval', 'test'):
                samples += [
                    (vid_nr, offset + 1) for offset in range(self.frame_modulo)]
                continue
            if n_images < self.clip_len:
                too_short += 1
                continue
            if n_images // self.frame_modulo > self.max_seq_len:
                too_long += 1
                continue
            if self.phase == 'train':
                samples += [(vid_nr, None)] * self.seq_per_vid
                continue
            elif self.phase == 'valid':
                x = n_images // (self.seq_per_vid_val * 2) - self.clip_len // 2
                start = max(1, x)
                end = min(n_images - self.clip_len, n_images - x)
                samples += [
                    (vid_nr, int(start)) for start in
                    np.linspace(start, end, self.seq_per_vid_val)]
                continue
        if self.phase not in ('eval', 'test') and self.n_images_dict:
            n_loaded = len(self.n_images_dict) - too_short - too_long
            print(f"{n_loaded} videos loaded "
                  f"({n_loaded / len(self.n_images_dict) * 100:.1f}%)")
            print(f"{too_short} videos are too short "
                  f"({too_short / len(self.n_images_dict) * 100:.1f}%)")
            print(f"{too_long} videos are too long "
                  f"({too_long / len(self.n_images_dict) * 100:.1f}%)")
        target_size_dict = {
            vid_nr: self.target_size for vid_nr in self.n_images_dict.keys()}
        return samples, target_size_dict

    def get_frame_nrs(self, vid_nr, start):
        n_images = self.n_images_dict[vid_nr]
        if self.phase in ('eval', 'test'):
            return list(range(start, n_images + 1, self.frame_modulo))
        return list(range(start, start + self.clip_len, self.frame_modulo))

    def get_annotation_dir(self, vid_nr):
        return self.dir / 'annotation' / f'{vid_nr:04d}'

    def get_data_file(self, vid_nr, f_nr, dkey):
        if dkey == 'frame':
            folder = 'images'
        elif dkey == 'sal':
            folder = 'maps'
        elif dkey == 'fix':
            folder = 'fixation'
        else:
            raise ValueError(f'Unknown data key {dkey}')
        return self.get_annotation_dir(vid_nr) / folder / f'{f_nr:04d}.png'

    def load_data(self, vid_nr, f_nr, dkey):
        read_flag = None if dkey == 'frame' else cv2.IMREAD_GRAYSCALE
        data_file = self.get_data_file(vid_nr, f_nr, dkey)
        if read_flag is not None:
            data = cv2.imread(str(data_file), read_flag)
        else:
            data = cv2.imread(str(data_file))
        if data is None:
            raise FileNotFoundError(data_file)
        if dkey == 'frame':
            data = np.ascontiguousarray(data[:, :, ::-1])

        if dkey == 'sal' and self.train and self.sal_offset is not None:
            data += self.sal_offset
            data[0, 0] = 0

        return data

    def preprocess_sequence(self, frame_seq, dkey, vid_nr):
        transformations = []
        if dkey == 'frame':
            transformations.append(transforms.ToPILImage())
            transformations.append(transforms.Resize(
                self.out_size, interpolation=PIL.Image.LANCZOS))
        transformations.append(transforms.ToTensor())
        if dkey == 'frame' and 'rgb_mean' in self.preproc_cfg:
            transformations.append(
                transforms.Normalize(
                    self.preproc_cfg['rgb_mean'], self.preproc_cfg['rgb_std']))
        elif dkey == 'sal':
            transformations.append(transforms.Lambda(utils.normalize_tensor))
        elif dkey == 'fix':
            transformations.append(
                transforms.Lambda(lambda fix: torch.gt(fix, 0.5)))

        processing = transforms.Compose(transformations)

        tensor = [processing(img) for img in frame_seq]
        tensor = torch.stack(tensor)
        return tensor

    def get_seq(self, vid_nr, frame_nrs, dkey):
        data_seq = [self.load_data(vid_nr, f_nr, dkey) for f_nr in frame_nrs]
        return self.preprocess_sequence(data_seq, dkey, vid_nr)

    def get_data(self, vid_nr, start):
        if start is None:
            max_start = self.n_images_dict[vid_nr] - self.clip_len + 1
            if max_start == 1:
                start = max_start
            else:
                start = np.random.randint(1, max_start)
        frame_nrs = self.get_frame_nrs(vid_nr, start)
        frame_seq = self.get_seq(vid_nr, frame_nrs, 'frame')
        target_size = self.target_size_dict[vid_nr]
        if self.phase == 'test' and self.source in ('DHF1K',):
            return frame_nrs, frame_seq, target_size
        sal_seq = self.get_seq(vid_nr, frame_nrs, 'sal')
        fix_seq = self.get_seq(vid_nr, frame_nrs, 'fix')
        return frame_nrs, frame_seq, sal_seq, fix_seq, target_size

    def __getitem__(self, item):
        vid_nr, start = self.samples[item]
        data = self.get_data(vid_nr, start)
        return data


class HollywoodDataset(DHF1KDataset):

    source = 'Hollywood'
    dynamic = True

    img_channels = 1
    n_videos = {
        'train': 747,
        'test': 884
    }
    test_vid_nrs = (1, 884)
    frame_rate = 24

    def __init__(self, out_size=(224, 416), val_size=75, n_images_file=None,
                 seq_per_vid_val=1, register_file='hollywood_register.json',
                 phase='train',
                 frame_modulo=4,
                 seq_len=12,
                 **kwargs):
        self.register = None
        self.phase_str = 'test' if phase in ('eval', 'test') else 'train'
        self.register_file = self.phase_str + "_" + register_file
        super().__init__(out_size=out_size, val_size=val_size,
                         n_images_file=n_images_file,
                         seq_per_vid_val=seq_per_vid_val,
                         x_val_seed=42, phase=phase, target_size=out_size,
                         frame_modulo=frame_modulo,
                         seq_len=seq_len,
                         **kwargs)
        if phase in ('eval', 'test'):
            self.target_size_dict = self.get_register()['vid_size_dict']

    @property
    def n_images_dict(self):
        if self._n_images_dict is None:
            self._n_images_dict = self.get_register()['n_images_dict']
            self._n_images_dict = {vid_nr: ni for vid_nr, ni
                                   in self._n_images_dict.items()
                                   if vid_nr // 100 in self.vid_nr_array}
        return self._n_images_dict

    def get_register(self):
        if self.register is None:
            register_file = config_path / self.register_file
            if register_file.exists():
                with open(config_path / register_file, 'r') as f:
                    self.register = json.load(f)
                for reg_key in ('n_images_dict', 'start_image_dict',
                                'vid_size_dict'):
                    self.register[reg_key] = {
                        int(key): val for key, val in
                        self.register[reg_key].items()}
            else:
                self.register = self.generate_register()
                with open(config_path / register_file, 'w') as f:
                    json.dump(self.register, f, indent=2)
        return self.register

    def generate_register(self):
        n_shots = {
            vid_nr: 0 for vid_nr in range(1, self.n_videos[self.phase_str] + 1)}
        n_images_dict = {}
        start_image_dict = {}
        vid_size_dict = {}

        for folder in sorted(self.dir.glob('actionclip*')):
            name = folder.stem
            vid_nr_start = 10 + len(self.phase_str)
            vid_nr = int(name[vid_nr_start:vid_nr_start + 5])
            shot_nr = int(name[-2:].replace("_", ""))
            n_shots[vid_nr] += 1

            vid_nr_shot_nr = 100 * vid_nr + shot_nr
            image_files = sorted((folder / 'images').glob('actionclip*.png'))
            n_images_dict[vid_nr_shot_nr] = len(image_files)
            start_image_dict[vid_nr_shot_nr] = int(image_files[0].stem[-5:])
            img = cv2.imread(str(image_files[0]))
            vid_size_dict[vid_nr_shot_nr] = tuple(img.shape[:2])

        return dict(
            n_shots=n_shots, n_images_dict=n_images_dict,
            start_image_dict=start_image_dict, vid_size_dict=vid_size_dict)

    def preprocess_sequence(self, frame_seq, dkey, vid_nr):
        transformations = [
            transforms.ToPILImage()
        ]

        vid_size = self.register['vid_size_dict'][vid_nr]
        if vid_size[0] != self.out_size[0]:
            interpolation = PIL.Image.LANCZOS if dkey in ('frame', 'sal')\
                else PIL.Image.NEAREST
            size = (self.out_size[0],
                    int(vid_size[1] * self.out_size[0] / vid_size[0]))
            transformations.append(
                transforms.Resize(size, interpolation=interpolation))

        transformations += [
            transforms.CenterCrop(self.out_size),
            transforms.ToTensor(),
        ]

        if dkey == 'frame' and 'rgb_mean' in self.preproc_cfg:
            transformations.append(
                transforms.Normalize(
                    self.preproc_cfg['rgb_mean'], self.preproc_cfg['rgb_std']))
        elif dkey == 'sal':
            transformations.append(transforms.Lambda(utils.normalize_tensor))
        elif dkey == 'fix':
            transformations.append(
                transforms.Lambda(lambda fix: torch.gt(fix, 0.5)))

        processing = transforms.Compose(transformations)

        tensor = [processing(img) for img in frame_seq]
        tensor = torch.stack(tensor)
        return tensor

    def preprocess_sequence_eval(self, frame_seq, dkey, vid_nr):
        transformations = []

        if dkey == 'frame':
            transformations.append(transforms.ToPILImage())
            transformations.append(
                transforms.Resize(
                    self.out_size, interpolation=PIL.Image.LANCZOS))

        transformations.append(transforms.ToTensor())
        if dkey == 'frame' and 'rgb_mean' in self.preproc_cfg:
            transformations.append(
                transforms.Normalize(
                    self.preproc_cfg['rgb_mean'], self.preproc_cfg['rgb_std']))
        elif dkey == 'sal':
            transformations.append(transforms.Lambda(utils.normalize_tensor))
        elif dkey == 'fix':
            transformations.append(
                transforms.Lambda(lambda fix: torch.gt(fix, 0.5)))

        processing = transforms.Compose(transformations)

        tensor = [processing(img) for img in frame_seq]
        tensor = torch.stack(tensor)
        return tensor

    def get_annotation_dir(self, vid_nr_shot_nr):
        vid_nr = vid_nr_shot_nr // 100
        shot_nr = vid_nr_shot_nr % 100
        return self.dir / f"actionclip{self.phase_str}{vid_nr:05d}_{shot_nr:1d}"

    def get_data_file(self, vid_nr_shot_nr, f_nr, dkey):
        if dkey == 'frame':
            folder = 'images'
        elif dkey == 'sal':
            folder = 'maps'
        elif dkey == 'fix':
            folder = 'fixation'
        else:
            raise ValueError(f'Unknown data key {dkey}')
        vid_nr = vid_nr_shot_nr // 100
        f_nr += self.register['start_image_dict'][vid_nr_shot_nr] - 1
        return self.get_annotation_dir(vid_nr_shot_nr) / folder /\
            f'actionclip{self.phase_str}{vid_nr:05d}_{f_nr:05d}.png'

    def get_seq(self, vid_nr, frame_nrs, dkey):
        data_seq = [self.load_data(vid_nr, f_nr, dkey) for f_nr in frame_nrs]
        preproc_fun = self.preprocess_sequence if self.phase \
            in ('train', 'valid') else self.preprocess_sequence_eval
        return preproc_fun(data_seq, dkey, vid_nr)

    @property
    def dir(self):
        if self._dir is None:
            self._dir = Path(os.environ["HOLLYWOOD_DATA_DIR"]) /\
                        ('training' if self.phase in ('train', 'valid')
                         else 'testing')
        return self._dir


class UCFSportsDataset(DHF1KDataset):

    source = 'UCFSports'
    dynamic = True

    img_channels = 1
    n_train_val_videos = 103
    test_vid_nrs = (1, 47)
    frame_rate = 24

    def __init__(self, out_size=(256, 384), val_size=10, n_images_file=None,
                 seq_per_vid_val=1, register_file='ucfsports_register.json',
                 phase='train',
                 frame_modulo=4,
                 seq_len=12,
                 **kwargs):
        self.phase_str = 'test' if phase in ('eval', 'test') else 'train'
        self.register_file = self.phase_str + "_" + register_file
        self.register = None
        super().__init__(out_size=out_size, val_size=val_size,
                         n_images_file=n_images_file,
                         seq_per_vid_val=seq_per_vid_val,
                         x_val_seed=27, target_size=out_size,
                         frame_modulo=frame_modulo, phase=phase,
                         seq_len=seq_len,
                         **kwargs)
        if phase in ('eval', 'test'):
            self.target_size_dict = self.get_register()['vid_size_dict']

    @property
    def n_images_dict(self):
        if self._n_images_dict is None:
            self._n_images_dict = self.get_register()['n_images_dict']
            self._n_images_dict = {vid_nr: ni for vid_nr, ni
                                   in self._n_images_dict.items()
                                   if vid_nr in self.vid_nr_array}
        return self._n_images_dict

    def get_register(self):
        if self.register is None:
            register_file = config_path / self.register_file
            if register_file.exists():
                with open(config_path / register_file, 'r') as f:
                    self.register = json.load(f)
                for reg_key in ('n_images_dict', 'vid_name_dict',
                                'vid_size_dict'):
                    self.register[reg_key] = {
                        int(key): val for key, val in
                        self.register[reg_key].items()}
            else:
                self.register = self.generate_register()
                with open(config_path / register_file, 'w') as f:
                    json.dump(self.register, f, indent=2)
        return self.register

    def generate_register(self):
        n_images_dict = {}
        vid_name_dict = {}
        vid_size_dict = {}

        for vid_idx, folder in enumerate(sorted(self.dir.glob('*-*'))):
            vid_nr = vid_idx + 1
            vid_name_dict[vid_nr] = folder.stem
            image_files = list((folder / 'images').glob('*.png'))
            n_images_dict[vid_nr] = len(image_files)
            img = cv2.imread(str(image_files[0]))
            vid_size_dict[vid_nr] = tuple(img.shape[:2])

        return dict(
            vid_name_dict=vid_name_dict, n_images_dict=n_images_dict,
            vid_size_dict=vid_size_dict)

    def preprocess_sequence(self, frame_seq, dkey, vid_nr):
        transformations = [
            transforms.ToPILImage()
        ]

        vid_size = self.register['vid_size_dict'][vid_nr]
        interpolation = PIL.Image.LANCZOS if dkey in ('frame', 'sal')\
            else PIL.Image.NEAREST
        out_size_ratio = self.out_size[1] / self.out_size[0]
        this_size_ratio = vid_size[1] / vid_size[0]
        if this_size_ratio < out_size_ratio:
            size = (int(self.out_size[1] / this_size_ratio), self.out_size[1])
        else:
            size = (self.out_size[0], int(self.out_size[0] * this_size_ratio))
        transformations.append(
            transforms.Resize(size, interpolation=interpolation))

        transformations += [
            transforms.CenterCrop(self.out_size),
            transforms.ToTensor(),
        ]

        if dkey == 'frame' and 'rgb_mean' in self.preproc_cfg:
            transformations.append(
                transforms.Normalize(
                    self.preproc_cfg['rgb_mean'], self.preproc_cfg['rgb_std']))
        elif dkey == 'sal':
            transformations.append(transforms.Lambda(utils.normalize_tensor))
        elif dkey == 'fix':
            transformations.append(
                transforms.Lambda(lambda fix: torch.gt(fix, 0.5)))

        processing = transforms.Compose(transformations)

        tensor = [processing(img) for img in frame_seq]
        tensor = torch.stack(tensor)
        return tensor

    preprocess_sequence_eval = HollywoodDataset.preprocess_sequence_eval

    def get_annotation_dir(self, vid_nr):
        vid_name = self.register['vid_name_dict'][vid_nr]
        return self.dir / vid_name

    def get_data_file(self, vid_nr, f_nr, dkey):
        if dkey == 'frame':
            folder = 'images'
        elif dkey == 'sal':
            folder = 'maps'
        elif dkey == 'fix':
            folder = 'fixation'
        else:
            raise ValueError(f'Unknown data key {dkey}')
        vid_name = self.register['vid_name_dict'][vid_nr]
        return self.get_annotation_dir(vid_nr) / folder /\
            f"{vid_name[:-4]}_{vid_name[-3:]}_{f_nr:03d}.png"

    get_seq = HollywoodDataset.get_seq

    @property
    def dir(self):
        if self._dir is None:
            self._dir = Path(os.environ["UCFSPORTS_DATA_DIR"]) /\
                        ('training' if self.phase in ('train', 'valid')
                         else 'testing')
        return self._dir


def get_optimal_out_size(img_size):
    ar = img_size[0] / img_size[1]
    min_prod = 100
    max_prod = 120
    ar_array = []
    size_array = []
    for n1 in range(7, 14):
        for n2 in range(7, 14):
            if min_prod <= n1 * n2 <= max_prod:
                this_ar = n1 / n2
                this_ar_ratio = min((ar, this_ar)) / max((ar, this_ar))
                ar_array.append(this_ar_ratio)
                size_array.append((n1, n2))

    max_ar_ratio_idx = np.argmax(np.array(ar_array)).item()
    bn_size = size_array[max_ar_ratio_idx]
    out_size = tuple(r * 32 for r in bn_size)
    return out_size


class FolderVideoDataset(Dataset):

    def __init__(self, images_path, frame_modulo=None, source=None):
        self.images_path = images_path
        self.frame_modulo = frame_modulo or 5
        self.preproc_cfg = {
            'rgb_mean': (0.485, 0.456, 0.406),
            'rgb_std': (0.229, 0.224, 0.225),
        }

        frame_files = sorted(list(images_path.glob("*")))
        frame_files = [file for file in frame_files
                         if file.suffix in ('.png', '.jpg', '.jpeg')]
        self.frame_files = frame_files
        self.vid_nr_array = [0]
        self.n_images_dict = {0: len(frame_files)}

        img = cv2.imread(str(frame_files[0]))
        img_size = tuple(img.shape[:2])
        self.target_size_dict = {0: img_size}

        if source == 'DHF1K' and img_size == (360, 640):
            self.out_size = (224, 384)

        elif source == 'Hollywood':
            self.out_size = (224, 416)

        elif source == 'UCFSports':
            self.out_size = (256, 384)

        else:
            self.out_size = get_optimal_out_size(img_size)

    def load_frame(self, f_nr):
        frame_file = self.frame_files[f_nr - 1]
        frame = cv2.imread(str(frame_file))
        if frame is None:
            raise FileNotFoundError(frame_file)
        frame = np.ascontiguousarray(frame[:, :, ::-1])
        return frame

    def preprocess_sequence(self, frame_seq):
        transformations = []
        transformations.append(transforms.ToPILImage())
        transformations.append(transforms.Resize(
            self.out_size, interpolation=PIL.Image.LANCZOS))
        transformations.append(transforms.ToTensor())
        if 'rgb_mean' in self.preproc_cfg:
            transformations.append(
                transforms.Normalize(
                    self.preproc_cfg['rgb_mean'], self.preproc_cfg['rgb_std']))
        processing = transforms.Compose(transformations)
        tensor = [processing(img) for img in frame_seq]
        tensor = torch.stack(tensor)
        return tensor

    def get_data(self, vid_nr, start):
        n_images = self.n_images_dict[vid_nr]
        frame_nrs = list(range(start, n_images + 1, self.frame_modulo))
        frame_seq = [self.load_frame(f_nr) for f_nr in frame_nrs]
        frame_seq = self.preprocess_sequence(frame_seq)
        target_size = self.target_size_dict[vid_nr]
        return frame_nrs, frame_seq, target_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        return self.get_data(item, 0)


class FolderImageDataset(Dataset):

    def __init__(self, images_path):
        self.images_path = images_path
        self.frame_modulo = 1
        self.preproc_cfg = {
            'rgb_mean': (0.485, 0.456, 0.406),
            'rgb_std': (0.229, 0.224, 0.225),
        }

        image_files = sorted(list(images_path.glob("*")))
        image_files = [file for file in image_files
                       if file.suffix in ('.png', '.jpg', '.jpeg')]
        self.image_files = image_files
        self.n_images_dict = {
            img_idx: 1 for img_idx in range(len(self.image_files))}

        self.target_size_dict = {}
        self.out_size_dict = {}
        for img_idx, file in enumerate(image_files):
            img = cv2.imread(str(file))
            img_size = tuple(img.shape[:2])
            self.target_size_dict[img_idx] = img_size
            self.out_size_dict[img_idx] = get_optimal_out_size(img_size)

    def load_image(self, img_idx):
        image_file = self.image_files[img_idx]
        image = cv2.imread(str(image_file))
        if image is None:
            raise FileNotFoundError(image_file)
        image = np.ascontiguousarray(image[:, :, ::-1])
        return image

    def preprocess(self, img, out_size):
        transformations = [
            transforms.ToPILImage(),
            transforms.Resize(
                out_size, interpolation=PIL.Image.LANCZOS),
            transforms.ToTensor(),
        ]
        if 'rgb_mean' in self.preproc_cfg:
            transformations.append(
                transforms.Normalize(
                    self.preproc_cfg['rgb_mean'], self.preproc_cfg['rgb_std']))
        processing = transforms.Compose(transformations)
        tensor = processing(img)
        return tensor

    def get_data(self, img_idx):
        file = self.image_files[img_idx]
        img = cv2.imread(str(file))
        assert (img is not None)
        img = np.ascontiguousarray(img[:, :, ::-1])
        out_size = self.out_size_dict[img_idx]
        img = self.preprocess(img, out_size)
        return [1], img, self.target_size_dict[img_idx]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, item):
        return self.get_data(item, 0)
