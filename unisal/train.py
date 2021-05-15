from pathlib import Path
import os
import subprocess
import inspect
import shutil
import json
from random import shuffle
from itertools import zip_longest
import pprint
import random
import copy
import time
from itertools import chain
import math

import torch
import torch.nn.functional as F
import cv2
from tensorboardX import SummaryWriter
import numpy as np

from . import salience_metrics
from . import utils
from . import model
from . import data

cv2.setNumThreads(0)

parent_dir = Path(__file__).resolve().parent.parent
if "TRAIN_DIR" not in os.environ:
    os.environ["TRAIN_DIR"] = str(parent_dir / "training_runs")
if "PRED_DIR" not in os.environ:
    os.environ["PRED_DIR"] = str(parent_dir / "predictions")


class Trainer(utils.KwConfigClass):
    """
    Trainer class that handles training, evaluation and inference.

    Arguments:
        num_epochs: Number of training epochs
        optim_algo: Optimization algorithm (e.g. 'SGD')
        momentum: Optimizer momentum if applicable
        lr: Learning rate
        lr_scheduler: Learning rate scheduler (e.g. 'ExponentialLR')
        lr_gamma: Learnign rate decay for 'ExponentialLR' scheduler
        weight_decay: Weight decay (except for CNN)
        cnn_weight_decay: Backbone CNN weight decay
        grad_clip: Gradient clipping magnitude
        loss_metrics: Loss metrics. Defautls equivalent to [1].
        loss_weights: Weights of the individual loss metrics. Defaults
            equivalent to [1].
        data_sources: Data sources. Default equivalent to [1].
        batch_size: DHF1K batch size
        salicon_batch_size: SALICON batch size
        hollywood_batch_size: Hollywood-2 batch size
        ucfsports_batch_size: UCFSports batch size
        salicon_weight: Weight of the SALICON loss. Default is 0.5 to
            account for the larger number of batches.
        hollywood_weight: Weight of the Hollywood-2 loss.
        ucfsports_weight: Weight of the UCF Sports loss.
        data_cfg: Dictionary with kwargs for the DHF1KDataset class.
        salicon_cfg: Dictionary with kwargs for the SALICONDataset class.
        hollywood_cfg: Dictionary with kwargs for the HollywoodDataset
            class.
        ucfsports_cfg: Dictionary with kwargs for the UCFSportsDataset
            class.
        shuffle_datasets: Whether to train on batches of the individual
            datasets in random order. If False, batches are drawn
            in alternating order.
        cnn_lr_factor: Factor of the backbone CNN learnign rate compared to
            the overall learning rate.
        train_cnn_after: Freeze the backbone CNN for N epochs.
        cnn_eval: If True, keep the backbone CNN in evaluation mode (use
            pretrained BatchNorm running estimates for mean and variance).
        model_cfg: Dictionary with kwards for the model class
        prefix: Prefix for the training folder name. Defaults to timestamp.
        suffix: Suffix for the training folder name.
        num_workers: Number of parallel workers for data loading.
        chkpnt_warmup: Number of epochs before saving the first checkpoint.
        chkpnt_epochs: Save a checkpoint every N epchs.
        tboard: Use TensorboardX to visualize the training.
        debug: Debug mode.
        new_instance: Always leave this parameter as True. Reserved for
            loading an Trainer class from a saved configuration file.

    [1] https://arxiv.org/abs/1801.07424

    """

    phases = ('train', 'valid')
    all_data_sources = ('DHF1K', 'Hollywood', 'UCFSports', 'SALICON')

    def __init__(self,
                 num_epochs=16,
                 optim_algo='SGD',
                 momentum=0.9,
                 lr=0.04,
                 lr_scheduler='ExponentialLR',
                 lr_gamma=0.8,
                 weight_decay=1e-4,
                 cnn_weight_decay=1e-5,
                 grad_clip=2.,
                 loss_metrics=('kld', 'nss', 'cc'),
                 loss_weights=(1, -0.1, -0.1),
                 data_sources=('DHF1K', 'Hollywood', 'UCFSports', 'SALICON'),
                 batch_size=4,
                 salicon_batch_size=32,
                 hollywood_batch_size=4,
                 ucfsports_batch_size=4,
                 salicon_weight=.5,
                 hollywood_weight=1.,
                 ucfsports_weight=1.,
                 data_cfg=None,
                 salicon_cfg=None,
                 hollywood_cfg=None,
                 ucfsports_cfg=None,
                 shuffle_datasets=True,
                 cnn_lr_factor=0.1,
                 train_cnn_after=2,
                 cnn_eval=True,
                 model_cfg=None,
                 prefix=None,
                 suffix='unisal',
                 num_workers=6,
                 chkpnt_warmup=3,
                 chkpnt_epochs=2,
                 tboard=True,
                 debug=False,
                 new_instance=True,
                 ):
        # Save training parameters
        self.num_epochs = num_epochs
        self.optim_algo = optim_algo
        self.momentum = momentum
        self.lr = lr
        self.lr_scheduler = lr_scheduler
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay
        self.cnn_weight_decay = cnn_weight_decay
        self.grad_clip = grad_clip
        self.loss_metrics = loss_metrics
        self.loss_weights = loss_weights
        self.data_sources = data_sources
        self.batch_size = batch_size
        self.salicon_batch_size = salicon_batch_size
        self.hollywood_batch_size = hollywood_batch_size
        self.ucfsports_batch_size = ucfsports_batch_size
        self.salicon_weight = salicon_weight
        self.hollywood_weight = hollywood_weight
        self.ucfsports_weight = ucfsports_weight
        self.data_cfg = data_cfg or {}
        self.salicon_cfg = salicon_cfg or {}
        self.hollywood_cfg = hollywood_cfg or {}
        self.ucfsports_cfg = ucfsports_cfg or {}
        self.shuffle_datasets = shuffle_datasets
        self.cnn_lr_factor = cnn_lr_factor
        self.train_cnn_after = train_cnn_after
        self.cnn_eval = cnn_eval
        self.model_cfg = model_cfg or {}
        if 'sources' not in self.model_cfg:
            self.model_cfg['sources'] = data_sources

        # Create training directory. Uses env var TRAIN_DIR
        self.suffix = suffix
        if prefix is None:
            prefix = utils.get_timestamp()
        self.prefix = prefix

        # Other opertational parameters
        self.num_workers = num_workers
        self.chkpnt_warmup = chkpnt_warmup
        self.chkpnt_epochs = chkpnt_epochs
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.tboard = tboard
        self.debug = debug

        if debug:
            self.num_workers = 0
            self.num_epochs = 4
            self.chkpnt_epochs = 2
            self.chkpnt_warmup = 1
            self.suffix += '_debug'
            self.data_cfg.update({'subset': 0.02})
            self.salicon_cfg.update({'subset': 0.02})
            self.hollywood_cfg.update({'subset': 0.04})
            self.ucfsports_cfg.update({'subset': 0.1})
            self.eval_subeset = 1.

        # Initialize properties etc.
        self.epoch = 0
        self.phase = None
        self._datasets = {}
        self._dataloaders = {}
        self._scheduler = None
        self._optimizer = None
        self._model = None
        self.best_epoch = 0
        self.best_val_score = None
        self.is_best = False
        self.all_scalars = {}
        self._writer = None
        self._salicon_datasets = {}
        self._salicon_dataloaders = {}
        self._hollywood_datasets = {}
        self._hollywood_dataloaders = {}
        self._ucfsports_datasets = {}
        self._ucfsports_dataloaders = {}
        self.mit1003_finetuned = False

        # Clone repository to train dir
        self.new_instance = new_instance
        if new_instance:
            self.new_instance = False
            self.train_dir.mkdir(parents=True, exist_ok=True)
            self.save_cfg(self.train_dir)
            self.model.save_cfg(self.train_dir)
            self.copy_code()
            for src in self.data_sources:
                self.get_dataset('train', source=src).save_cfg(self.train_dir)

    def fit(self):
        """
        Train the model
        """

        # Print information about the trainer class to the terminal
        # pprint.pprint(self.asdict(), width=1)

        # Iterate over all epochs
        while self.epoch < self.num_epochs:

            # Runt the current epoch
            self.fit_epoch()

            # Save a checkpoint if applicable
            if (self.epoch >= self.chkpnt_warmup
                and (self.epoch + 1) % self.chkpnt_epochs == 0)\
                    or self.epoch == self.num_epochs - 1:
                self.save_chkpnt()

            self.epoch += 1

        # Save the training data (losses, etc.)
        self.export_scalars()

        return self.best_val_score

    def fit_epoch(self):
        """
        Run the training and evaluation phase for the current epoch
        """

        # Perform LR decay
        self.scheduler.step(epoch=self.epoch)
        lr = self.optimizer.param_groups[0]['lr']
        print(f"\nEpoch {self.epoch:3d}, lr {lr:.5f}")
        self.add_scalar('conv/lr', lr, self.epoch)

        # Run the training and validation phase
        for self.phase in self.phases:
            self.fit_phase()

    def fit_phase(self):
        """
        Run the current phase (training or validation)
        """
        sources = self.data_sources

        # Prepare book keeping
        running_losses = {src: 0. for src in sources}
        running_loss_summands = {
            src: [0. for _ in self.loss_weights] for src in sources}
        n_samples = {src: 0 for src in sources}

        # Shuffle the dataset batches
        dataloaders = {src: self.get_dataloader(self.phase, src)
                       for src in sources}
        all_batches = [src for src in chain.from_iterable(zip_longest(
            *[[src for _ in range(len(dataloaders[src]))] for src in sources]
            )) if src is not None]
        if self.shuffle_datasets:
            shuffle(all_batches)
        if self.epoch == 0:
            print(f"Number of batches: {len(all_batches)}")
            print(", ".join(f"{src}: {len(dataloaders[src])}"
                            for src in sources))

        # Set model train/eval mode
        self.model.train(self.phase == 'train')

        # Switch CNN gradients on/off and set CNN eval mode (for BN modules)
        if self.phase == 'train':
            cnn_grad = self.epoch >= self.train_cnn_after
            for param in self._model.cnn.parameters():
                param.requires_grad = cnn_grad
            if self.cnn_eval:
                self.model.cnn.eval()

        # Iterate over all batches
        data_iters = {src: iter(dataloaders[src]) for src in sources}
        for sample_idx, src in enumerate(all_batches):

            # Get the next batch
            sample = next(data_iters[src])
            target_size = (sample[-1][0][0].item(),
                           sample[-1][1][0].item())
            sample = sample[:-1]

            # Fit/evaluate the batch
            loss, loss_summands, batch_size = self.fit_sample(
                sample, grad_clip=self.grad_clip, target_size=target_size,
                source='SALICON' if src == 'MIT1003' else src)

            # Book keeping
            running_losses[src] += loss * batch_size
            running_loss_summands[src] = [
                r + l * batch_size
                for r, l in zip(running_loss_summands[src], loss_summands)]
            n_samples[src] += batch_size

        # Book keeping and writing to TensorboardX
        sources_eval = [src for src in sources if n_samples[src] > 0]
        for src in sources_eval:
            phase_loss = running_losses[src] / n_samples[src]
            phase_loss_summands = [
                loss_ / n_samples[src] for loss_ in running_loss_summands[src]]

            print(f'{src:9s}:   Phase: {self.phase}, loss: {phase_loss:.4f}, '
                  + ", ".join(f"loss_{idx}: {loss_:.4f}"
                              for idx, loss_ in enumerate(phase_loss_summands)))

            key = 'conv' if src == 'DHF1K' else src.lower()
            self.add_scalar(f'{key}/loss/{self.phase}', phase_loss, self.epoch)
            for idx, loss_ in enumerate(phase_loss_summands):
                self.add_scalar(f'{key}/loss_{idx}/{self.phase}', loss_,
                                self.epoch)

            if src == "DHF1K" and self.phase == 'valid' and\
                    self.epoch >= self.chkpnt_warmup:
                val_score = - phase_loss
                if self.best_val_score is None:
                    self.best_val_score = val_score
                elif val_score > self.best_val_score:
                    self.best_val_score = val_score
                    self.is_best = True
                    self.model.save_weights(self.train_dir, 'best')
                    with open(self.train_dir / 'best_epoch.dat', 'w') as f:
                        f.write(str(self.epoch))
                    with open(self.train_dir / 'best_val_loss.dat', 'w') as f:
                        f.write(str(val_score))
                else:
                    self.is_best = False

    def fit_sample(self, sample, grad_clip=None, **kwargs):
        """
        Take a sample containing a batch, and fit/evaluate the model
        """

        with torch.set_grad_enabled(self.phase == 'train'):
            _, x, sal, fix = sample

            # Add temporal dimension to image data
            if x.dim() == 4:
                x = x.unsqueeze(1)
                sal = sal.unsqueeze(1)
                fix = fix.unsqueeze(1)
            x = x.float().to(self.device)
            sal = sal.float().to(self.device)
            fix = fix.to(self.device)

            # Switch the gradients of unused modules off to prevent unnecessary
            # weight decay
            if self.phase == 'train':
                # Switch the RNN gradients off if this is a image batch
                rnn_grad = x.shape[1] != 1 or not self.model.bypass_rnn
                for param in chain(self._model.rnn.parameters(),
                                   self._model.post_rnn.parameters()):
                    param.requires_grad = rnn_grad

                # Switch the gradients of unused dataset-specific modules off
                for name, param in self.model.named_parameters():
                    for source in self.all_data_sources:
                        if source.lower() in name.lower():
                            param.requires_grad = source == kwargs['source']

            # Run forward pass
            pred_seq = self.model(x, **kwargs)

            # Compute the total loss
            loss_summands = self.loss_sequences(
                pred_seq, sal, fix, metrics=self.loss_metrics)
            loss_summands = [l.mean(1).mean(0) for l in loss_summands]
            loss = sum(weight * l for weight, l in
                              zip(self.loss_weights, loss_summands))

        # Run backward pass and optimization step
        if self.phase == 'train':
            self.optimizer.zero_grad()
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip)
            self.optimizer.step()

        return loss.item(), [l.item() for l in loss_summands], x.shape[0]

    @staticmethod
    def loss_sequences(pred_seq, sal_seq, fix_seq, metrics):
        """
        Compute the training losses
        """

        losses = []
        for this_metric in metrics:
            if this_metric == 'kld':
                losses.append(utils.kld_loss(pred_seq, sal_seq))
            if this_metric == 'nss':
                losses.append(utils.nss(pred_seq.exp(), fix_seq))
            if this_metric == 'cc':
                losses.append(utils.corr_coeff(pred_seq.exp(), sal_seq))
        return losses

    def run_inference(self, source, vid_nr, dataset=None, phase=None,
                      smooth_method=None, metrics=None, save_predictions=False,
                      return_predictions=False, seq_len_factor=0.5,
                      random_seed=27, n_aucs_maps=10, auc_portion=1.0,
                      model_domain=None, folder_suffix=None):

        if dataset is None:
            assert phase, "Must provide either dataset or phase"
            dataset = self.get_dataset(phase, source)

        if random_seed is not None:
            random.seed(random_seed)

        # Get the original resolution
        target_size = dataset.target_size_dict[vid_nr]

        # Set the keyword arguments for the forward pass
        model_kwargs = {
            'source': model_domain or source,
            'target_size': target_size}

        # Make sure that the model was trained on the selected domain
        if model_kwargs['source'] not in self.model.sources:
            print(f"\nWarning! Evaluation bn source {model_kwargs['source']} "
                  f"doesn't exist in model.\n  Using {self.model.sources[0]}.")
            model_kwargs['source'] = self.model.sources[0]

        # Select static or dynamic forward pass for Bypass-RNN
        model_kwargs.update(
            {'static': model_kwargs['source'] in ('SALICON', 'MIT300', 'MIT1003')})

        # Set additional parameters
        static_data = source in ('SALICON', 'MIT300', 'MIT1003')
        if static_data:
            smooth_method = None
            auc_portion = 1.
            n_images = 1
            frame_modulo = 1
        else:
            n_images = dataset.n_images_dict[vid_nr]
            frame_modulo = dataset.frame_modulo

        # Prepare the model
        self.model.to(self.device)
        self.model.eval()
        torch.cuda.empty_cache()

        # Prepare the prediction and target tensors
        results_size = (1, n_images, 1, *model_kwargs['target_size'])
        pred_seq = torch.full(results_size, 0, dtype=torch.float)
        if metrics is not None:
            sal_seq = torch.full(results_size, 0, dtype=torch.float)
            fix_seq = torch.full(results_size, 0, dtype=torch.uint8)
        else:
            sal_seq, fix_seq = None, None

        # Define input sequence length
        # seq_len = self.batch_size * self.get_dataset('train').seq_len * \
        #     seq_len_factor
        seq_len = int(12 * seq_len_factor)

        # Iterate over different offsets to create the interleaved predictions
        for offset in range(min(frame_modulo, n_images)):

            # Get the data
            if not static_data:
                sample = dataset.get_data(vid_nr, offset + 1)
            else:
                sample = dataset.get_data(vid_nr)

            # Preprocess the data
            sample = sample[:-1]
            if len(sample) >= 4:
                # if len(sample) == 5:
                #     sample = sample[:-1]
                frame_nrs, frame_seq, this_sal_seq, this_fix_seq = sample
                this_sal_seq = this_sal_seq.unsqueeze(0).float()
                this_fix_seq = this_fix_seq.unsqueeze(0)
                if frame_seq.dim() == 3:
                    frame_seq = frame_seq.unsqueeze(0)
                    this_sal_seq = this_sal_seq.unsqueeze(0)
                    this_fix_seq = this_fix_seq.unsqueeze(0)
            else:
                if metrics is not None:
                    raise ValueError(
                        "Labels needed for evaluation metrics but not provided"
                        "by dataset.")
                frame_nrs, frame_seq = sample
                this_sal_seq, this_fix_seq = None, None
                if frame_seq.dim() == 3:
                    frame_seq = frame_seq.unsqueeze(0)
            frame_seq = frame_seq.unsqueeze(0).float()
            frame_idx_array = [f_nr - 1 for f_nr in frame_nrs]
            frame_seq = frame_seq.to(self.device)

            # Run all sequences of the current offset
            h0 = [None]
            for start in range(0, len(frame_idx_array), seq_len):

                # Select the frames
                end = min(len(frame_idx_array), start + seq_len)
                this_frame_seq = frame_seq[:, start:end, :, :, :]
                this_frame_idx_array = frame_idx_array[start:end]

                # Forward pass
                this_pred_seq, h0 = self.model(
                    this_frame_seq, h0=h0, return_hidden=True,
                    **model_kwargs)

                # Insert the predictions into the prediction array
                this_pred_seq = this_pred_seq.cpu()
                pred_seq[:, this_frame_idx_array, :, :, :] =\
                    this_pred_seq

            # Keep the training targets if scores are to be computed
            if metrics is not None:
                sal_seq[:, frame_idx_array, :, :, :] = this_sal_seq
                fix_seq[:, frame_idx_array, :, :, :] = this_fix_seq

        # Assert non-empty predictions
        assert(torch.min(pred_seq.exp().sum(-1).sum(-1)) > 0)

        # Optionally smooth the interleaved sequences
        if smooth_method is not None:
            pred_seq = pred_seq.numpy()
            pred_seq = utils.smooth_sequence(pred_seq, smooth_method)
            pred_seq = torch.from_numpy(pred_seq).float()

        # optionally save the predictions
        if save_predictions:

            # Construct the output folder name
            folder_name = source
            if folder_suffix is not None:
                folder_name += f"{folder_suffix}"
            this_pred_dir = self.pred_dir / folder_name
            this_pred_dir.mkdir(exist_ok=True)

            # Construct a subfolder if applicable
            if source == 'DHF1K':
                this_pred_dir /= f"{vid_nr:04d}"
            elif source == 'MIT1003':
                this_pred_dir /= self.mit1003_dir.stem
            elif source == 'MIT300' and 'x_val_step' in self.salicon_cfg:
                this_pred_dir /= f"MIT300_xVal{self.salicon_cfg['x_val_step']}"
            elif source in ('Hollywood', 'UCFSports'):
                this_pred_dir /= dataset.get_annotation_dir(vid_nr).stem
            this_pred_dir.mkdir(exist_ok=True)

            # Iterate over the prediction frames
            for frame_idx, smap in enumerate(torch.unbind(pred_seq, dim=1)):

                # Define the filename
                if source == 'SALICON':
                    filename = f"COCO_test2014_{vid_nr:012d}.png"
                elif source == 'MIT300':
                    filename = dataset.samples[vid_nr][0]
                elif source == 'MIT1003':
                    filename = dataset.all_image_files[vid_nr]['img']
                elif source in ('Hollywood', 'UCFSports'):
                    filename = dataset.get_data_file(
                        vid_nr, frame_idx + 1, 'frame').name
                else:
                    filename = f"{frame_idx + 1:04d}.png"

                # Posporcess prediction
                smap = smap.exp()
                smap = torch.squeeze(smap)
                smap = utils.to_numpy(smap)

                # Optinally save numpy file in addition to the image
                if source == 'MIT300':
                    np.save(
                        this_pred_dir / filename.replace(".jpg", ""),
                        smap.copy())

                # Save prediction as image
                smap = (smap / np.amax(smap) * 255).astype(np.uint8)
                pred_file = this_pred_dir / filename
                cv2.imwrite(str(pred_file), smap,
                            [cv2.IMWRITE_JPEG_QUALITY, 100])

        # Optionally compute the scores
        if metrics is not None:

            # Compute the KLD, NSS and CC metrics
            vid_scores = []
            loss_sequences = self.loss_sequences(
                pred_seq, sal_seq, fix_seq, metrics=metrics)
            loss_sequences = [loss.numpy() for loss in loss_sequences]
            vid_scores += loss_sequences

            def other_maps():
                """Sample reference maps for s-AUC"""
                while True:
                    this_map = np.zeros(results_size[-2:])
                    video_nrs = random.sample(
                        dataset.n_images_dict.keys(), n_aucs_maps)
                    for map_idx, vid_nr in enumerate(video_nrs):
                        frame_nr = random.randint(
                            1, dataset.n_images_dict[vid_nr])
                        if static_data:
                            this_this_map = dataset.get_fixation_map(vid_nr)
                        else:
                            this_this_map = dataset.get_seq(
                                vid_nr, [frame_nr], 'fix').numpy()[0, 0, ...]
                        this_this_map = cv2.resize(
                            this_this_map, tuple(target_size[::-1]),
                            cv2.INTER_NEAREST
                        )
                        this_map += this_this_map

                    this_map = np.clip(this_map, 0, 1)
                    yield this_map

            # Compute the SIM, AUC-J, s-AUC metrics
            vid_scores += self.eval_sequences(
                pred_seq, sal_seq, fix_seq, metrics,
                other_maps=other_maps() if 'aucs' in metrics else None,
                auc_portion=auc_portion)

            # Average the scores over the frames
            mean_scores = np.array([np.mean(scores) for scores in vid_scores])

            if return_predictions:
                return pred_seq, mean_scores
            return mean_scores

        if return_predictions:
            return pred_seq

    @staticmethod
    def eval_sequences(pred_seq, sal_seq, fix_seq, metrics,
                       other_maps=None, auc_portion=1.):
        """
        Compute SIM, AUC-J and s-AUC scores
        """

        # process inputs
        metrics = [metric for metric in metrics
                   if metric in ('sim', 'aucj', 'aucs')]
        if 'aucs' in metrics:
            assert(other_maps is not None)

        # Preprocess sequences
        shape = pred_seq.shape
        new_shape = (-1, shape[-2], shape[-1])
        pred_seq = pred_seq.exp()
        pred_seq = pred_seq.detach().cpu().numpy().reshape(new_shape)
        sal_seq = sal_seq.detach().cpu().numpy().reshape(new_shape)
        fix_seq = fix_seq.detach().cpu().numpy().reshape(new_shape)

        # Optionally compute AUC-s for a subset of frames to reduce runtime
        if auc_portion < 1:
            auc_indices = set(
                random.sample(
                    range(shape[1]), max(1, int(auc_portion * shape[1]))))
        else:
            auc_indices = set(list(range(shape[1])))

        # Compute the metrics
        results = {metric: [] for metric in metrics}
        for idx, (pred, sal, fix) in enumerate(zip(pred_seq, sal_seq, fix_seq)):
            for this_metric in metrics:
                if this_metric == 'sim':
                    results['sim'].append(
                        salience_metrics.similarity(pred, sal))
                if this_metric == 'aucj':
                    if idx in auc_indices:
                        results['aucj'].append(
                            salience_metrics.auc_judd(pred, fix))
                if this_metric == 'aucs':
                    if idx in auc_indices:
                        other_map = next(other_maps)
                        results['aucs'].append(salience_metrics.auc_shuff_acl(
                            pred, fix, other_map))
        return [np.array(results[metric]) for metric in metrics]

    @property
    def pred_dir(self):
        """Directory to save predictions"""
        pred_dir = Path(os.environ["PRED_DIR"]) / f"{self.prefix}_{self.suffix}"
        pred_dir.mkdir(exist_ok=True, parents=True)
        return pred_dir

    def score_model(self, subset=1, source='DHF1K',
             metrics=('kld', 'nss', 'cc', 'sim', 'aucj', 'aucs'),
             smooth_method=None, seq_len_factor=2,
             random_seed=27, n_aucs_maps=10, auc_portion=0.5,
             model_domain=None, phase=None, load_weights=True,
             vid_nr_array=None):
        """
        Compute the evaluation scores of the model.

        For Hollywood-2 and UCF Sports, test set scores are computed since the
        ground truth is available.
        For all other datasets, the test set is held out and therefore
        validation set scores are computed.
        """

        if load_weights:
            # Load the best weights, if available, otherwise the weights of
            # the last epoch
            try:
                self.model.load_best_weights(self.train_dir)
                print('Best weights loaded')
            except FileNotFoundError:
                print('No best weights found')
                self.model.load_last_chkpnt(self.train_dir)
                print('Last checkpoint loaded')

        # Select the appropriate phase (see docstring) and get the dataset
        if phase is None:
            phase = 'eval' if source in ('DHF1K', 'SALICON', 'MIT1003')\
                else 'test'
        dataset = self.get_dataset(phase, source)

        if vid_nr_array is None:
            # Get list of sample numbers
            vid_nr_array = list(dataset.n_images_dict.keys())

        # Iterate over the videos/images and compute the scores
        scores = []
        tmr = utils.Timer(f'Evaluating {len(vid_nr_array)} {source} videos')
        random.seed(random_seed)
        with torch.no_grad():
            for vid_idx, vid_nr in enumerate(vid_nr_array):
                this_scores = self.run_inference(
                    source, vid_nr, dataset=dataset,
                    smooth_method=smooth_method, metrics=metrics,
                    seq_len_factor=seq_len_factor, n_aucs_maps=n_aucs_maps,
                    auc_portion=auc_portion, model_domain=model_domain)
                scores.append(this_scores)
                if vid_idx == 0:
                    print(f' Nr.   ( .../{len(vid_nr_array):4d}), ' +
                          ', '.join(f'{metric:5s}' for metric in metrics))
                print(f'{vid_nr:6d} ' +
                      f'({vid_idx + 1:4d}/{len(vid_nr_array):4d}), ' +
                      ', '.join(f'{score:.3f}' for score in this_scores))

        # Compute the average video scores
        tmr.finish()
        scores = np.array(scores)
        mean_scores = scores.mean(0)

        # In previous literature, scores were computed across all video frames,
        # which means that each videos contribution to the overall score is
        # weighted by its number of frames. The equivalent scores are denoted
        # below as weighted mean
        num_frames_array = [
            dataset.n_images_dict[vid_nr] for vid_nr in vid_nr_array]
        weighted_mean_scores = np.average(scores, 0, num_frames_array)

        # Print and save the scores
        print()
        print("Macro average (average of video averages) scores:")
        print(', '.join(f'{metric:5s}' for metric in metrics))
        print(', '.join(f'{score:.3f}' for score in mean_scores))
        print()
        print("Weighted average (per-frame average) scores:")
        print(', '.join(f'{metric:5s}' for metric in metrics))
        print(', '.join(f'{score:.3f}' for score in weighted_mean_scores))
        if subset == 1:
            dest_dir = self.mit1003_dir if source == 'MIT1003' \
                else self.train_dir
            if source in ('Hollywood', 'UCFSports'):
                source += "_resized"
            with open(dest_dir / f'{source}_eval_scores.json', 'w') as f:
                json.dump(scores, f, cls=utils.NumpyEncoder, indent=2)
            with open(dest_dir / f'{source}_eval_mean_scores.json', 'w')\
                    as f:
                json.dump(mean_scores, f, cls=utils.NumpyEncoder, indent=2)
            with open(dest_dir /
                      f'{source}_eval_weighted_mean_scores.json', 'w') as f:
                json.dump(weighted_mean_scores, f, cls=utils.NumpyEncoder,
                          indent=2)
            try:
                for score, metric in zip(mean_scores, metrics):
                    self.add_scalar(f'{source}_eval/{metric}', score, self.epoch)
                self.export_scalars(suffix=f"_eval-{source}")
            except AttributeError:
                print('add_scalar failed because tensorboard is closed.')

        return metrics, mean_scores, scores

    def generate_predictions(self, smooth_method=None, source='DHF1K',
                             phase='eval', load_weights=True, vid_nr_array=None,
                             **kwargs):
        """Generate predictions for submission and visualization"""

        if load_weights:
            # Load the best weights, if available, otherwise the weights of
            # the last epoch
            try:
                self.model.load_best_weights(self.train_dir)
                print('Best weights loaded')
            except FileNotFoundError:
                print('No best weights found')
                self.model.load_last_chkpnt(self.train_dir)
                print('Last checkpoint loaded')

        # Get the dataset
        dataset = self.get_dataset(phase, source)

        if vid_nr_array is None:
            # Get list of sample numbers
            if source == 'MIT300':
                vid_nr_array = list(range(300))
            else:
                vid_nr_array = list(dataset.n_images_dict.keys())

        tmr = utils.Timer(f'Predicting {len(vid_nr_array)} {source} videos')
        with torch.no_grad():
            for vid_nr in vid_nr_array:
                self.run_inference(
                    source, vid_nr, dataset=dataset,
                    smooth_method=smooth_method, save_predictions=True,
                    **kwargs)
        tmr.finish()

    def generate_predictions_from_path(
            self, folder_path, is_video, source=None, load_weights=True,
            **kwargs):

        # Process inputs
        if source is None:
            source = 'DHF1K' if is_video else 'SALICON'

        if source in ('MIT1003', 'MIT300'):
            try:
                self.model.load_weights(self.train_dir, "ft_mit1003")
                print("Fine-tuned MIT1003 weights loaded")
                load_weights = False
            except:
                print("No MIT1003 fine-tuned weights found.")
            source = 'SALICON'

        images_path = folder_path / 'images'
        torch.cuda.empty_cache()

        if load_weights:
            # Load the best weights, if available, otherwise the weights of
            # the last epoch
            try:
                self.model.load_best_weights(self.train_dir)
                print('Best weights loaded')
            except FileNotFoundError:
                print('No best weights found')
                self.model.load_last_chkpnt(self.train_dir)
                print('Last checkpoint loaded')

        with torch.no_grad():
            if is_video:
                frame_modulo = 5 if source == 'DHF1K' else 4
                dataset = data.FolderVideoDataset(
                    images_path, source=source, frame_modulo=frame_modulo)
                pred_dir = folder_path / 'saliency'
                pred_dir.mkdir(exist_ok=True)

                pred_seq = self.run_inference(
                    source, 0, dataset=dataset, phase=None,
                    return_predictions=True, folder_suffix=None, **kwargs)

                # Iterate over the prediction frames
                for frame_idx, smap in enumerate(torch.unbind(pred_seq, dim=1)):

                    # Postprocess prediction
                    smap = smap.exp()
                    smap = torch.squeeze(smap)
                    smap = utils.to_numpy(smap)

                    # Save prediction as image
                    filename = dataset.frame_files[frame_idx].name
                    smap = (smap / np.amax(smap) * 255).astype(np.uint8)
                    pred_file = pred_dir / filename
                    cv2.imwrite(
                        str(pred_file), smap, [cv2.IMWRITE_JPEG_QUALITY, 100])

            else:
                dataset = data.FolderImageDataset(images_path)
                pred_dir = folder_path / 'saliency'
                pred_dir.mkdir(exist_ok=True)

                for img_idx in range(len(dataset)):
                    pred_seq = self.run_inference(
                        source, img_idx, dataset=dataset, phase=None,
                        return_predictions=True, **kwargs)

                    smap = pred_seq[:, 0, ...]

                    # Posporcess prediction
                    smap = smap.exp()
                    smap = torch.squeeze(smap)
                    smap = utils.to_numpy(smap)

                    # Save prediction as image
                    filename = dataset.image_files[img_idx].name
                    smap = (smap / np.amax(smap) * 255).astype(np.uint8)
                    pred_file = pred_dir / filename
                    cv2.imwrite(
                        str(pred_file), smap, [cv2.IMWRITE_JPEG_QUALITY, 100])

    def fine_tune_mit(
            self, lr=0.01, num_epochs=8, lr_gamma=0.8, x_val_step=0,
            train_cnn_after=0):
        """Fine tune the model with the MIT1003 dataset for MIT300 submission"""

        # Set the fine tuning parameters
        self.num_epochs = num_epochs
        self.lr = lr
        self.lr_gamma = lr_gamma
        self.lr_scheduler = 'ExponentialLR'
        self.optim_algo = 'SGD'
        self.momentum = 0.9
        self.weight_decay = 1e-4
        self.grad_clip = None
        self.loss_weights = (1.,)
        self.loss_metrics = ('kld',)
        self.salicon_weight = 1.
        self.salicon_batch_size = 32
        self.data_sources = ('MIT1003',)
        self.shuffle_datasets = True
        self.cnn_lr_factor = 0.1
        self.train_cnn_after = train_cnn_after
        self.cnn_eval = True
        self.salicon_cfg.update({'x_val_step': x_val_step})
        self.mit1003_finetuned = True

        self.num_workers = 4

        # Load the best weights, if available, otherwise the weights of
        # the last epoch
        try:
            self.model.load_best_weights(self.train_dir)
            print('Best weights loaded')
        except FileNotFoundError:
            print('No best weights found')
            self.model.load_last_chkpnt(self.train_dir)

        # Run the fine tuning
        # pprint.pprint(self.asdict(), width=1)
        best_epoch = None
        best_val = None
        self._model.to(self.device)
        while self.epoch < self.num_epochs:
            self.scheduler.step(epoch=self.epoch)
            lr = self.optimizer.param_groups[0]['lr']
            print(f"\nEpoch {self.epoch:3d}, lr {lr:.5f}")
            for self.phase in self.phases:
                self.fit_phase()

            val_loss = self.all_scalars['mit1003']['loss']['valid'][self.epoch]
            if math.isnan(val_loss):
                best_epoch = 0
                best_val = 1000
                break

            val_score = - val_loss
            if self.best_val_score is None:
                self.best_val_score = val_score
            elif val_score > self.best_val_score:
                self.best_val_score = val_score
                best_epoch = self.epoch
                best_val = val_loss

            self.epoch += 1

        self.export_scalars()
        return best_val, best_epoch

    def get_dataset(self, phase, source="DHF1K"):
        """Get the Dataset instance for a phase and source"""

        if source not in self._datasets:
            self._datasets[source] = {}
        if phase not in self._datasets[source]:
            if source == "DHF1K":
                dataset_cls = data.get_dataset()
                config = self.data_cfg
            else:
                dataset_cls_name = f"{source}Dataset"
                dataset_cls = getattr(data, dataset_cls_name)
                if source in ('MIT300',):
                    config = {}
                elif source in ('MIT1003',):
                    config = getattr(self, f"salicon_cfg")
                else:
                    config = getattr(self, f"{source.lower()}_cfg")
            self._datasets[source][phase] = dataset_cls(
                phase=phase, **config)
            print(f'{source:10s} {phase} dataset loaded with'
                  f' {len(self._datasets[source][phase])} samples')
        return self._datasets[source][phase]

    def get_dataloader(self, phase, source="DHF1K"):
        """Get the DataLoader instance for a phase and source"""

        if source not in self._dataloaders:
            self._dataloaders[source] = {}
        if phase not in self._dataloaders[source]:
            dataset = self.get_dataset(phase, source)
            if source == "DHF1K":
                batch_size = self.batch_size
            elif source in ("Hollywood", "UCFSports"):
                batch_size = self.__getattribute__(
                    f"{source.lower()}_batch_size")
            elif phase == 'valid' and source == 'MIT1003':
                batch_size = 8
            elif source in ('SALICON', 'MIT1003'):
                batch_size = self.salicon_batch_size or len(dataset) //\
                    len(self.get_dataloader(phase))
                if batch_size > 8:
                    batch_size -= batch_size % 2
                batch_size = min(32, batch_size)
            else:
                raise ValueError(f'Unknown dataset source {source}')
            print(f"{source}, phase {phase} batch size: {batch_size}")
            self._dataloaders[source][phase] = data.get_dataloader(source)(
                dataset, batch_size=batch_size,
                shuffle=phase == 'train', num_workers=self.num_workers,
                drop_last=True,
            )
        return self._dataloaders[source][phase]

    @property
    def model(self):
        """Set the model and move it to self.device"""

        if self._model is None:
            model_cls = model.get_model()
            self._model = model_cls(**self.model_cfg)
            self._model.to(self.device)
        return self._model

    def measure_runtime(self):
        """Measure the model runtime on GPU and CPU"""

        # Turn off gradient computation
        with torch.no_grad():

            # Prepare the experiment
            self.model.eval()
            self.num_workers = 0
            dl = self.get_dataloader('test', 'DHF1K')
            sample = next(iter(dl))
            _, x, _ = sample
            x = x.float().cuda()
            print(x.shape)
            x_0 = x[:1, :1, ...].clone().contiguous()
            output, h0 = self.model(x_0, return_hidden=True)

            # Measure the average time to process single frames on the GPU
            times = []
            for t_idx in range(1, x.shape[1]):
                x_t = x[:1, t_idx:t_idx+1, ...].clone().contiguous()
                t0 = time.time()
                output, h0 = self.model(
                    x_t, h0=h0, return_hidden=True, static=False)
                dt = time.time() - t0
                times.append(dt)
            print()
            dt = sum(times) / len(times)
            print(f"Avg single-frame time: {dt:.4f} s ({1 / dt:.1f} fps)")
            dt = min(times)
            print(f"Min single-frame time: {dt:.4f} s ({1 / dt:.1f} fps)")
            dt = max(times)
            print(f"Max single-frame time: {dt:.4f} s ({1 / dt:.1f} fps)")

            # Measure the average time to process single frames on the GPU
            self.model.cpu()
            x = x.cpu()
            h0 = [h0_.cpu() for h0_ in h0]
            times = []
            torch.cuda.empty_cache()
            for t_idx in range(1, min(x.shape[1], 16)):
                x_t = x[:1, t_idx:t_idx+1, ...].clone().contiguous()
                t0 = time.time()
                output, h0 = self.model(
                    x_t, h0=h0, return_hidden=True, static=False)
                dt = time.time() - t0
                times.append(dt)
            print()
            dt = sum(times) / len(times)
            print("Avg single-frame CPU time: "
                  f"{dt:.4f} s ({1 / dt:.1f} fps)")
            dt = min(times)
            print(f"Min single-frame CPU time: {dt:.4f} s ({1 / dt:.1f} fps)")
            dt = max(times)
            print(f"Max single-frame CPU time: {dt:.4f} s ({1 / dt:.1f} fps)")

    def measure_model_size(self):
        """Measure the model size"""

        model_cls = model.get_model()
        this_model_cfg = copy.deepcopy(self.model_cfg)
        net = model_cls(verbose=0, **this_model_cfg)

        dest = self.train_dir
        torch.save(net, dest / 'net_full.pth')
        file_size = (dest / 'net_full.pth').stat().st_size / 1e6
        print("All data sources net size: "
              f"{file_size:.2f} MB")
        torch.save(net.state_dict(), dest / 'net_full_state_dict.pth')
        file_size = (dest / 'net_full_state_dict.pth').stat().st_size / 1e6
        print(f"All data sources net state dict size: {file_size:.2f} MB")

    def get_model_parameter_groups(self):
        """
        Get parameter groups.
        Output CNN parameters separately with reduced LR and weight decay.
        """
        def parameters_except_cnn():
            parameters = []
            adaptation = []
            for name, module in self.model.named_children():
                if name == 'cnn':
                    continue
                elif 'adaptation' in name:
                    adaptation += list(module.parameters())
                else:
                    parameters += list(module.parameters())
            return parameters, adaptation

        parameters, adaptation = parameters_except_cnn()

        for name, this_parameter in self.model.named_parameters():
            if 'gaussian' in name:
                parameters.append(this_parameter)

        return [
            {'params': parameters + adaptation},
            {'params': self.model.cnn.parameters(),
             'lr': self.lr * self.cnn_lr_factor,
             'weight_decay': self.cnn_weight_decay,
             },
        ]

    @property
    def optimizer(self):
        """Return the optimizer"""
        if self._optimizer is None:
            if self.optim_algo == 'SGD':
                self._optimizer = torch.optim.SGD(
                    self.get_model_parameter_groups(), lr=self.lr,
                    momentum=self.momentum, weight_decay=self.weight_decay)
            elif self.optim_algo == 'Adam':
                self._optimizer = torch.optim.Adam(
                    self.get_model_parameter_groups(), lr=self.lr,
                    weight_decay=self.weight_decay)
            elif self.optim_algo == 'RMSprop':
                self._optimizer = torch.optim.RMSprop(
                    self.get_model_parameter_groups(), lr=self.lr,
                    weight_decay=self.weight_decay, momentum=self.momentum)

        return self._optimizer

    @property
    def scheduler(self):
        """Return the learning rate scheduler"""
        if self._scheduler is None:
            if self.lr_scheduler == 'ExponentialLR':
                self._scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    self.optimizer, gamma=self.lr_gamma,
                    last_epoch=self.epoch - 1)
            else:
                raise ValueError(f"Unknown scheduler {self.lr_scheduler}")
        return self._scheduler

    @property
    def train_dir(self):
        """Return the directory to store the training data"""
        return Path(os.environ["TRAIN_DIR"]) / f"{self.prefix}_{self.suffix}"

    def copy_code(self):
        """Make a copy of the code to facilitate leading of older models"""

        source = Path(inspect.getfile(Trainer)).parent.parent
        destination = self.train_dir / 'code_copy'
        tracked_files = [
            '.gitignore',
            'unisal/__init__.py',
            'unisal/data.py',
            'unisal/model.py',
            'unisal/models/MobileNetV2.py',
            'unisal/models/__init__.py',
            'unisal/models/cgru.py',
            'unisal/models/weights/mobilenet_v2.pth.tar',
            'unisal/salience_metrics.py',
            'unisal/train.py',
            'unisal/utils.py',
            'run.py',
            'unisal/dhf1k_n_images.dat',
            'unisal/cache/img_size_dict.json',
            'unisal/cache/train_hollywood_register.json',
            'unisal/cache/train_ucfsports_register.json',
            'unisal/cache/test_hollywood_register.json',
            'unisal/cache/test_ucfsports_register.json',
        ]
        for file in tracked_files:
            subdir = Path(file).parent
            (destination / subdir).mkdir(exist_ok=True, parents=True)
            shutil.copy2(source / file, destination / subdir)

    def save_chkpnt(self):
        """Save model and trainer checkpoint"""
        print(f"Saving checkpoint at epoch {self.epoch}")
        chkpnt = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        chkpnt.update({key: self.__dict__[key] for key in (
            'epoch', 'best_epoch', 'best_val_score', 'all_scalars',)})
        torch.save(chkpnt, self.train_dir / f'chkpnt_epoch{self.epoch:04d}.pth')

    def load_checkpoint(self, file):
        """Load model and trainer checkpoint"""
        chkpnt = torch.load(file)
        self.model.load_state_dict(chkpnt['model_state_dict'])
        self.optimizer.load_state_dict(chkpnt['optimizer_state_dict'])
        self.__dict__.update({key: chkpnt[key] for key in (
            'epoch', 'best_epoch', 'best_val_score', 'all_scalars')})
        self.epoch += 1

    def load_last_chkpnt(self):
        """Load latest model checkpoint"""
        last_chkpnt = sorted(list(self.train_dir.glob('chkpnt_epoch*.pth')))[-1]
        self.load_checkpoint(last_chkpnt)

    def add_scalar(self, key, value, epoch, this_tboard=True):
        """Add a scalar to self.all_scalars and TensorboardX"""
        keys = key.split('/')
        this_dict = self.all_scalars
        for key_ in keys:
            if key_ not in this_dict:
                this_dict[key_] = {}
            this_dict = this_dict[key_]
        this_dict[epoch] = value

        if self.tboard and this_tboard:
            self.writer.add_scalar(key, value, epoch)

    @property
    def writer(self):
        """Return TensorboardX writer"""
        if self.tboard and self._writer is None:
            if self.data_sources == ('MIT1003',):
                log_dir = self.mit1003_dir
                log_dir.mkdir(exist_ok=True)
            else:
                log_dir = self.train_dir
            self._writer = SummaryWriter(log_dir=str(log_dir))
        return self._writer

    @property
    def mit1003_dir(self):
        """Return directory to fine tune on MIT1003"""
        if self.mit1003_finetuned:
            mit1003_dir = self.train_dir / f"MIT1003_lr{self.lr:.4f}_" \
                f"lrGamma{self.lr_gamma:.2f}_nEpochs{self.num_epochs}_" \
                f"TrainCNNAfter{self.train_cnn_after}_" \
                f"xVal{self.salicon_cfg['x_val_step']}"
        else:
            mit1003_dir = self.train_dir / f"MIT1003_" \
                f"xVal{self.salicon_cfg['x_val_step']}"
        mit1003_dir.mkdir(exist_ok=True)
        return mit1003_dir

    def export_scalars(self, suffix=''):
        """Save self.all_scalars"""
        if self.data_sources == ('MIT1003',):
            export_dir = self.mit1003_dir
        else:
            export_dir = self.train_dir
        with open(export_dir / f'all_scalars{suffix}.json', 'w') as f:
            json.dump(self.all_scalars, f, cls=utils.NumpyEncoder,
                      indent=2)

    def get_configs(self):
        """Get configurations of trainer, dataset and model instances"""
        return {
            'Trainer': self.asdict(),
            'Dataset': self.get_dataset('train', 'DHF1K').asdict(),
            'Model': self.model.asdict()
        }

    @property
    def train_id(self):
        return '/'.join(self.train_dir.parts[-2:])
