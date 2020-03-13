from collections import OrderedDict
import pprint
from functools import partial
from itertools import product

import torch
from torch import nn
import torch.nn.functional as F

from . import utils
from .models.cgru import ConvGRU
from .models.MobileNetV2 import MobileNetV2, InvertedResidual


def get_model():
    """Return the model class"""
    return UNISAL


class BaseModel(nn.Module):
    """Abstract model class with functionality to save and load weights"""

    def forward(self, *input):
        raise NotImplementedError

    def save_weights(self, directory, name):
        torch.save(self.state_dict(), directory / f'weights_{name}.pth')

    def load_weights(self, directory, name):
        self.load_state_dict(torch.load(directory / f'weights_{name}.pth'))

    def load_best_weights(self, directory):
        self.load_state_dict(torch.load(directory / f'weights_best.pth'))

    def load_epoch_checkpoint(self, directory, epoch):
        """Load state_dict from a Trainer checkpoint at a specific epoch"""
        chkpnt = torch.load(directory / f"chkpnt_epoch{epoch:04d}.pth")
        self.load_state_dict(chkpnt['model_state_dict'])

    def load_checkpoint(self, file):
        """Load state_dict from a specific Trainer checkpoint"""
        """Load """
        chkpnt = torch.load(file)
        self.load_state_dict(chkpnt['model_state_dict'])

    def load_last_chkpnt(self, directory):
        """Load state_dict from the last Trainer checkpoint"""
        last_chkpnt = sorted(list(directory.glob('chkpnt_epoch*.pth')))[-1]
        self.load_checkpoint(last_chkpnt)


# Set default backbone CNN kwargs
default_cnn_cfg = {
    'widen_factor': 1., 'pretrained': True, 'input_channel': 32,
    'last_channel': 1280}

# Set default RNN kwargs
default_rnn_cfg = {
    'kernel_size': (3, 3), 'gate_ksize': (3, 3),
    'dropout': (False, True, False), 'drop_prob': (0.2, 0.2, 0.2),
    'mobile': True,
}


class DomainBatchNorm2d(nn.Module):
    """
    Domain-specific 2D BatchNorm module.

    Stores a BN module for a given list of sources.
    During the forward pass, select the BN module based on self.this_source.
    """

    def __init__(self, num_features, sources, momenta=None, **kwargs):
        """
            num_features: Number of channels
            sources: List of sources
            momenta: List of BatchNorm momenta corresponding to the sources.
                Default is 0.1 for each source.
            kwargs: Other BatchNorm kwargs
        """
        super().__init__()
        self.sources = sources

        # Process momenta input
        if momenta is None:
            momenta = [0.1] * len(sources)
        self.momenta = momenta
        if 'momentum' in kwargs:
            del kwargs['momentum']

        # Instantiate the BN modules
        for src, mnt in zip(sources, self.momenta):
            self.__setattr__(f"bn_{src}", nn.BatchNorm2d(
                num_features, momentum=mnt, **kwargs))

        # Prepare the self.this_source attribute that will be updated at runtime
        # by the model
        self.this_source = None

    def forward(self, x):
        return self.__getattr__(f"bn_{self.this_source}")(x)


class UNISAL(BaseModel, utils.KwConfigClass):
    """
    UNISAL model. See paper for more information.

    Arguments:
        rnn_input_channels: Number of channels of the RNN input.
        rnn_hidden_channels: Number of channels of the RNN hidden state.
        cnn_cfg: Dictionary with kwargs for the backbone CNN.
        rnn_cfg: Dictionary with kwargs for the RNN.
        res_rnn: Whether to add the RNN features with a residual connection.
        bypass_rnn: Whether to bypass the RNN for static inputs.
            Requires res_rnn.
        drop_probs: Dropout probabilities for
            [backbone CNN outputs, Skip-2x and Skip-4x].
        gaussian_init: Method to initialize the learned Gaussian parameters.
            If "manual", 16 pre-defined Gaussians are initialized.
        n_gaussians: Number of Gaussians if gaussian_init is "random".
        smoothing_ksize: Size of the Smoothing kernel.
        bn_momentum: Momentum of the BatchNorm running estimates for dynamic
            batches.
        static_bn_momentum: Momentum of the BatchNorm running estimates for
            static batches.
        sources: List of datasets.
        ds_bn: Domain-specific BatchNorm (DSBN).
        ds_adaptation: Domain-specific Adaptation.
        ds_smoothing: Domain-specific Smoothing.
        ds_gaussians: Domain-specific Gaussian prior maps.
        verbose: Verbosity level.
    """

    def __init__(self,
                 rnn_input_channels=256, rnn_hidden_channels=256,
                 cnn_cfg=None,
                 rnn_cfg=None,
                 res_rnn=True,
                 bypass_rnn=True,
                 drop_probs=(0.0, 0.6, 0.6),
                 gaussian_init='manual',
                 n_gaussians=16,
                 smoothing_ksize=41,
                 bn_momentum=0.01,
                 static_bn_momentum=0.1,
                 sources=('DHF1K', 'Hollywood', 'UCFSports', 'SALICON'),
                 ds_bn=True,
                 ds_adaptation=True,
                 ds_smoothing=True,
                 ds_gaussians=True,
                 verbose=1,
                 ):
        super().__init__()

        # Check inputs
        assert(gaussian_init in ('random', 'manual'))
        # Bypass-RNN requires residual RNN connection
        if bypass_rnn:
            assert res_rnn

        # Manual Gaussian initialization generates 16 Gaussians
        if n_gaussians > 0 and gaussian_init == 'manual':
            n_gaussians = 16

        self.rnn_input_channels = rnn_input_channels
        self.rnn_hidden_channels = rnn_hidden_channels
        this_cnn_cfg = default_cnn_cfg.copy()
        this_cnn_cfg.update(cnn_cfg or {})
        self.cnn_cfg = this_cnn_cfg
        this_rnn_cfg = default_rnn_cfg.copy()
        this_rnn_cfg.update(rnn_cfg or {})
        self.rnn_cfg = this_rnn_cfg
        self.bypass_rnn = bypass_rnn
        self.res_rnn = res_rnn
        self.drop_probs = drop_probs
        self.gaussian_init = gaussian_init
        self.n_gaussians = n_gaussians
        self.smoothing_ksize = smoothing_ksize
        self.bn_momentum = bn_momentum
        self.sources = sources
        self.ds_bn = ds_bn
        self.static_bn_momentum = static_bn_momentum
        self.ds_adaptation = ds_adaptation
        self.ds_smoothing = ds_smoothing
        self.ds_gaussians = ds_gaussians
        self.verbose = verbose

        # Initialize backbone CNN
        self.cnn = MobileNetV2(**self.cnn_cfg)

        # Initialize Post-CNN module with optional dropout
        post_cnn = [
            ('inv_res', InvertedResidual(
                self.cnn.out_channels + n_gaussians,
                rnn_input_channels, 1, 1, bn_momentum=bn_momentum,
            ))
        ]
        if self.drop_probs[0] > 0:
            post_cnn.insert(0, (
                'dropout', nn.Dropout2d(self.drop_probs[0], inplace=False)
            ))
        self.post_cnn = nn.Sequential(OrderedDict(post_cnn))

        # Initialize Bypass-RNN if training on dynamic data
        if sources != ('SALICON',) or not self.bypass_rnn:
            self.rnn = ConvGRU(
                rnn_input_channels,
                hidden_channels=[rnn_hidden_channels],
                batchnorm=self.get_bn_module,
                **self.rnn_cfg)
            self.post_rnn = self.conv_1x1_bn(
                rnn_hidden_channels, rnn_input_channels)

        # Initialize first upsampling module US1
        self.upsampling_1 = nn.Sequential(OrderedDict([
            ('us1', self.upsampling(2)),
        ]))

        # Number of channels at the 2x scale
        channels_2x = 128

        # Initialize Skip-2x module
        self.skip_2x = self.make_skip_connection(
            self.cnn.feat_2x_channels, channels_2x, 2, self.drop_probs[1])

        # Initialize second upsampling module US2
        self.upsampling_2 = nn.Sequential(OrderedDict([
            ('inv_res', InvertedResidual(
                rnn_input_channels + channels_2x,
                channels_2x, 1, 2, batchnorm=self.get_bn_module)),
            ('us2', self.upsampling(2)),
        ]))

        # Number of channels at the 4x scale
        channels_4x = 64

        # Initialize Skip-4x module
        self.skip_4x = self.make_skip_connection(
            self.cnn.feat_4x_channels, channels_4x, 2, self.drop_probs[2])

        # Initialize Post-US2 module
        self.post_upsampling_2= nn.Sequential(OrderedDict([
            ('inv_res', InvertedResidual(
                channels_2x + channels_4x, channels_4x, 1, 2,
                batchnorm=self.get_bn_module)),
        ]))

        # Initialize domain-specific modules
        for source_str in self.sources:
            source_str = f'_{source_str}'.lower()

            # Initialize learned Gaussian priors parameters
            if n_gaussians > 0:
                self.set_gaussians(source_str)

            # Initialize Adaptation
            self.__setattr__(
                'adaptation' + (source_str if self.ds_adaptation else ''),
                nn.Sequential(*[
                    nn.Conv2d(channels_4x, 1, 1, bias=True)
                ]))

            # Initialize Smoothing
            smoothing = nn.Conv2d(
                1, 1, kernel_size=smoothing_ksize, padding=0, bias=False)
            with torch.no_grad():
                gaussian = self._make_gaussian_maps(
                    smoothing.weight.data,
                    torch.Tensor([[[0.5, -2]] * 2])
                )
                gaussian /= gaussian.sum()
                smoothing.weight.data = gaussian
            self.__setattr__(
                'smoothing' + (source_str if self.ds_smoothing else ''),
                smoothing)

        if self.verbose > 1:
            pprint.pprint(self.asdict(), width=1)

    @property
    def this_source(self):
        """Return current source for domain-specific BatchNorm."""
        return self._this_source

    @this_source.setter
    def this_source(self, source):
        """Set current source for domain-specific BatchNorm."""
        for module in self.modules():
            if isinstance(module, DomainBatchNorm2d):
                module.this_source = source
        self._this_source = source

    def get_bn_module(self, num_features, **kwargs):
        """Return BatchNorm class (domain-specific or domain-invariant)."""
        momenta = [self.bn_momentum if src != 'SALICON'
                   else self.static_bn_momentum for src in self.sources]
        if self.ds_bn:
            return DomainBatchNorm2d(
                num_features, self.sources, momenta=momenta, **kwargs)
        else:
            return nn.BatchNorm2d(num_features, **kwargs)

    # @staticmethod
    def upsampling(self, factor):
        """Return upsampling module."""
        return nn.Sequential(*[
            nn.Upsample(
                scale_factor=factor, mode='bilinear', align_corners=False),
        ])

    def set_gaussians(self, source_str, prefix='coarse_'):
        """Set Gaussian parameters."""
        suffix = source_str if self.ds_gaussians else ''
        self.__setattr__(
            prefix + 'gaussians' + suffix,
            self._initialize_gaussians(self.n_gaussians))

    def _initialize_gaussians(self, n_gaussians):
        """
        Return initialized Gaussian parameters.
        Dimensions: [idx, y/x, mu/logstd].
        """
        if self.gaussian_init == 'manual':
            gaussians = torch.Tensor([
                    list(product([0.25, 0.5, 0.75], repeat=2)) +
                    [(0.5, 0.25), (0.5, 0.5), (0.5, 0.75)] +
                    [(0.25, 0.5), (0.5, 0.5), (0.75, 0.5)] +
                    [(0.5, 0.5)],
                    [(-1.5, -1.5)] * 9 + [(0, -1.5)] * 3 + [(-1.5, 0)] * 3 +
                    [(0, 0)],
            ]).permute(1, 2, 0)

        elif self.gaussian_init == 'random':
            with torch.no_grad():
                gaussians = torch.stack([
                        torch.randn(
                            n_gaussians, 2, dtype=torch.float) * .1 + 0.5,
                        torch.randn(
                            n_gaussians, 2, dtype=torch.float) * .2 - 1,],
                    dim=2)

        else:
            raise NotImplementedError

        gaussians = nn.Parameter(gaussians, requires_grad=True)
        return gaussians

    @staticmethod
    def _make_gaussian_maps(x, gaussians, size=None, scaling=6.):
        """Construct prior maps from Gaussian parameters."""
        if size is None:
            size = x.shape[-2:]
            bs = x.shape[0]
        else:
            size = [size] * 2
            bs = 1
        dtype = x.dtype
        device = x.device

        gaussian_maps = []
        map_template = torch.ones(*size, dtype=dtype, device=device)
        meshgrids = torch.meshgrid(
            [torch.linspace(0, 1, size[0], dtype=dtype, device=device),
             torch.linspace(0, 1, size[1], dtype=dtype, device=device),])

        for gaussian_idx, yx_mu_logstd in enumerate(torch.unbind(gaussians)):
            map = map_template.clone()
            for mu_logstd, mgrid in zip(yx_mu_logstd, meshgrids):
                mu = mu_logstd[0]
                std = torch.exp(mu_logstd[1])
                map *= torch.exp(-((mgrid - mu) / std) ** 2 / 2)

            map *= scaling
            gaussian_maps.append(map)

        gaussian_maps = torch.stack(gaussian_maps)
        gaussian_maps = gaussian_maps.unsqueeze(0).expand(bs, -1, -1, -1)
        return gaussian_maps

    def _get_gaussian_maps(self, x, source_str, prefix='coarse_', **kwargs):
        """Return the constructed Gaussian prior maps."""
        suffix = source_str if self.ds_gaussians else ''
        gaussians = self.__getattr__(prefix + "gaussians" + suffix)
        gaussian_maps = self._make_gaussian_maps(x, gaussians, **kwargs)
        return gaussian_maps

    # @classmethod
    def make_skip_connection(self, input_channels, output_channels, expand_ratio, p,
                       inplace=False):
        """Return skip connection module."""
        hidden_channels = round(input_channels * expand_ratio)
        return nn.Sequential(OrderedDict([
            ('expansion', self.conv_1x1_bn(
                input_channels, hidden_channels)),
            ('dropout', nn.Dropout2d(p, inplace=inplace)),
            ('reduction', nn.Sequential(*[
                nn.Conv2d(hidden_channels, output_channels, 1),
                self.get_bn_module(output_channels),
            ])),
        ]))

    # @staticmethod
    def conv_1x1_bn(self, inp, oup):
        """Return pointwise convolution with BatchNorm and ReLU6."""
        return nn.Sequential(
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            self.get_bn_module(oup),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x, target_size=None, h0=None, return_hidden=False,
                source='DHF1K', static=None):
        """
        Forward pass.

        Arguments:
            x: Input batch of dimensions [batch, time, channel, h, w].
            target_size: (height, width) of the resized output.
            h0: Initial hidden state.
            return_hidden: Return [prediction, hidden_state].
            source: Data source of current batch. Must be in self.sources.
            static: Whether the current input is static. If None, this is
                inferred from the input dimensions or self.sources.
        """
        if target_size is None:
            target_size = x.shape[-2:]

        # Set the current source for the domain-specific BatchNorm modules
        self.this_source = source

        # Prepare other parameters
        source_str = f'_{source.lower()}'
        if static is None:
            static = x.shape[1] == 1 or self.sources == ('SALICON',)

        # Compute backbone CNN features and concatenate with Gaussian prior maps
        feat_seq_1x = []
        feat_seq_2x = []
        feat_seq_4x = []
        for t, img in enumerate(torch.unbind(x, dim=1)):
            im_feat_1x, im_feat_2x, im_feat_4x = self.cnn(img)

            im_feat_2x = self.skip_2x(im_feat_2x)
            im_feat_4x = self.skip_4x(im_feat_4x)

            if self.n_gaussians > 0:
                gaussian_maps = self._get_gaussian_maps(im_feat_1x, source_str)
                im_feat_1x = torch.cat((im_feat_1x, gaussian_maps), dim=1)

            im_feat_1x = self.post_cnn(im_feat_1x)
            feat_seq_1x.append(im_feat_1x)
            feat_seq_2x.append(im_feat_2x)
            feat_seq_4x.append(im_feat_4x)

        feat_seq_1x = torch.stack(feat_seq_1x, dim=1)

        # Bypass-RNN
        hidden, rnn_feat_seq, rnn_feat = (None,) * 3
        if not (static and self.bypass_rnn):
            rnn_feat_seq, hidden = self.rnn(feat_seq_1x, hidden=h0)

        # Decoder
        output_seq = []
        for idx, im_feat in enumerate(
                torch.unbind(feat_seq_1x, dim=1)):

            if not (static and self.bypass_rnn):
                rnn_feat = rnn_feat_seq[:, idx, ...]
                rnn_feat = self.post_rnn(rnn_feat)
                if self.res_rnn:
                    im_feat = im_feat + rnn_feat
                else:
                    im_feat = rnn_feat

            im_feat = self.upsampling_1(im_feat)
            im_feat = torch.cat((im_feat, feat_seq_2x[idx]), dim=1)
            im_feat = self.upsampling_2(im_feat)
            im_feat = torch.cat((im_feat, feat_seq_4x[idx]), dim=1)
            im_feat = self.post_upsampling_2(im_feat)

            im_feat = self.__getattr__(
                'adaptation' + (source_str if self.ds_adaptation else ''))(
                im_feat)

            im_feat = F.interpolate(
                im_feat, size=x.shape[-2:], mode='nearest')

            im_feat = F.pad(im_feat, [self.smoothing_ksize // 2] * 4,
                            mode='replicate')
            im_feat = self.__getattr__(
                'smoothing' + (source_str if self.ds_smoothing else ''))(
                im_feat)

            im_feat = F.interpolate(
                im_feat, size=target_size, mode='bilinear', align_corners=False)

            im_feat = utils.log_softmax(im_feat)
            output_seq.append(im_feat)
        output_seq = torch.stack(output_seq, dim=1)

        outputs = [output_seq]
        if return_hidden:
            outputs.append(hidden)
        if len(outputs) == 1:
            return outputs[0]
        return outputs
