import math
from collections import OrderedDict

import torch
from torch.distributions.bernoulli import Bernoulli
import torch.nn as nn
import torch.nn.functional as f
from torch.nn.parameter import Parameter
from torch.nn import init

# Inspired by:
# https://github.com/jacobkimmel/pytorch_convgru
# https://gist.github.com/halochou/acbd669af86ecb8f988325084ba7a749


class ConvGRUCell(nn.Module):
    """
    Generate a convolutional GRU cell.


    Arguments:
        input_ch: Number of channels of the input.
        hidden_ch: Number of channels of hidden state.
        kernel_size (tuple): Kernel size of the U and W operations.
        gate_ksize (tuple): Kernel size for the gates.
        bias: Add bias term to layers.
        norm: Normalization method. 'batch', 'instance' or ''.
        norm_momentum: BatchNorm momentum.
        affine_norm: Affine BatchNorm.
        batchnorm: External function that accepts a number of channels and
            returns a BatchNorm module (for DSBN). Overwrites norm and
            norm_momentum.
        drop_prob: Tuple of dropout probabilities for input, recurrent and
            output dropout.
        do_mode: If 'recurrent', the variational dropout is used, dropping out
            the same channels at every time step. If 'naive', different channels
            are dropped at each time step.
        r_bias, z_bias: Bias initialization for r and z gates.
        mobile: If True, MobileNet-style convolutions are used.
    """

    def __init__(self, input_ch, hidden_ch, kernel_size, gate_ksize=(1, 1),
                 bias=True, norm='', norm_momentum=0.1, affine_norm=True,
                 batchnorm=None, gain=1, drop_prob=(0., 0., 0.),
                 do_mode='recurrent', r_bias=0., z_bias=0., mobile=False,
                 **kwargs):
        super().__init__()

        self.input_ch = input_ch
        self.hidden_ch = hidden_ch
        self.kernel_size = kernel_size
        self.gate_ksize = gate_ksize
        self.mobile = mobile
        self.kwargs = {'init': 'xavier_uniform_'}
        self.kwargs.update(kwargs)

        # Process normalization arguments
        self.norm = norm
        self.norm_momentum = norm_momentum
        self.affine_norm = affine_norm
        self.batchnorm = batchnorm
        self.norm_kwargs = None
        if self.batchnorm is not None:
            self.norm = 'batch'
        elif self.norm:
            self.norm_kwargs = {
                'affine': self.affine_norm, 'track_running_stats': True,
                'momentum': self.norm_momentum}

        # Prepare normalization modules
        if self.norm:
            self.norm_r_x = self.get_norm_module(self.hidden_ch)
            self.norm_r_h = self.get_norm_module(self.hidden_ch)
            self.norm_z_x = self.get_norm_module(self.hidden_ch)
            self.norm_z_h = self.get_norm_module(self.hidden_ch)
            self.norm_out_x = self.get_norm_module(self.hidden_ch)
            self.norm_out_h = self.get_norm_module(self.hidden_ch)

        # Prepare dropout
        self.drop_prob = drop_prob
        self.do_mode = do_mode
        if self.do_mode == 'recurrent':
            # Prepare dropout masks if using recurrent dropout
            for idx, mask in self.yield_drop_masks():
                self.register_buffer(self.mask_name(idx), mask)
        elif self.do_mode != 'naive':
            raise ValueError('Unknown dropout mode ', self.do_mode)

        # Instantiate the main weight matrices
        self.w_r = self._conv2d(self.input_ch, self.gate_ksize, bias=False)
        self.u_r = self._conv2d(self.hidden_ch, self.gate_ksize, bias=False)
        self.w_z = self._conv2d(self.input_ch, self.gate_ksize, bias=False)
        self.u_z = self._conv2d(self.hidden_ch, self.gate_ksize, bias=False)
        self.w = self._conv2d(self.input_ch, self.kernel_size, bias=False)
        self.u = self._conv2d(self.hidden_ch, self.gate_ksize, bias=False)

        # Instantiate the optional biases and affine paramters
        self.bias = bias
        self.r_bias = r_bias
        self.z_bias = z_bias
        if self.bias or self.affine_norm:
            self.b_r = Parameter(torch.Tensor(self.hidden_ch, 1, 1))
            self.b_z = Parameter(torch.Tensor(self.hidden_ch, 1, 1))
            self.b_h = Parameter(torch.Tensor(self.hidden_ch, 1, 1))
        if self.affine_norm:
            self.a_r_x = Parameter(torch.Tensor(self.hidden_ch, 1, 1))
            self.a_r_h = Parameter(torch.Tensor(self.hidden_ch, 1, 1))
            self.a_z_x = Parameter(torch.Tensor(self.hidden_ch, 1, 1))
            self.a_z_h = Parameter(torch.Tensor(self.hidden_ch, 1, 1))
            self.a_h_x = Parameter(torch.Tensor(self.hidden_ch, 1, 1))
            self.a_h_h = Parameter(torch.Tensor(self.hidden_ch, 1, 1))

        self.gain = gain
        self.set_weights()

    def set_weights(self):
        """Initialize the parameters"""
        def gain_from_ksize(ksize):
            n = ksize[0] * ksize[1] * self.hidden_ch
            return math.sqrt(2. / n)
        with torch.no_grad():
            if not self.mobile:
                if self.gain < 0:
                    gain_1 = gain_from_ksize(self.kernel_size)
                    gain_2 = gain_from_ksize(self.gate_ksize)
                else:
                    gain_1 = gain_2 = self.gain
                init_fn = getattr(init, self.kwargs['init'])
                init_fn(self.w_r.weight, gain=gain_2)
                init_fn(self.u_r.weight, gain=gain_2)
                init_fn(self.w_z.weight, gain=gain_2)
                init_fn(self.u_z.weight, gain=gain_2)
                init_fn(self.w.weight, gain=gain_1)
                init_fn(self.u.weight, gain=gain_2)
            if self.bias or self.affine_norm:
                self.b_r.data.fill_(self.r_bias)
                self.b_z.data.fill_(self.z_bias)
                self.b_h.data.zero_()
            if self.affine_norm:
                self.a_r_x.data.fill_(1)
                self.a_r_h.data.fill_(1)
                self.a_z_x.data.fill_(1)
                self.a_z_h.data.fill_(1)
                self.a_h_x.data.fill_(1)
                self.a_h_h.data.fill_(1)

    def forward(self, x, h_tm1):
        # Initialize hidden state if necessary
        if h_tm1 is None:
            h_tm1 = self._init_hidden(x, cuda=x.is_cuda)

        # Compute gate components
        r_x = self.w_r(self.apply_dropout(x, 0, 0))
        r_h = self.u_r(self.apply_dropout(h_tm1, 1, 0))
        z_x = self.w_z(self.apply_dropout(x, 0, 1))
        z_h = self.u_z(self.apply_dropout(h_tm1, 1, 1))
        h_x = self.w(self.apply_dropout(x, 0, 2))
        h_h = self.u(self.apply_dropout(h_tm1, 1, 2))

        if self.norm:
            # Apply normalization
            r_x = self.norm_r_x(r_x)
            r_h = self.norm_r_h(r_h)
            z_x = self.norm_z_x(z_x)
            z_h = self.norm_z_h(z_h)
            h_x = self.norm_out_x(h_x)
            h_h = self.norm_out_h(h_h)

            if self.affine_norm:
                # Apply affine transformation
                r_x = r_x * self.a_r_x
                r_h = r_h * self.a_r_h
                z_x = z_x * self.a_z_x
                z_h = z_h * self.a_z_h
                h_x = h_x * self.a_h_x
                h_h = h_h * self.a_h_h

        # Compute gates with optinal bias
        if self.bias or self.affine_norm:
            r = torch.sigmoid(r_x + r_h + self.b_r)
            z = torch.sigmoid(z_x + z_h + self.b_z)
        else:
            r = torch.sigmoid(r_x + r_h)
            z = torch.sigmoid(z_x + z_h)

        # Compute new hidden state
        if self.bias or self.affine_norm:
            h = torch.tanh(h_x + r * h_h + self.b_h)
        else:
            h = torch.tanh(h_x + r * h_h)
        h = (1 - z) * h_tm1 + z * h

        # Optionally apply output dropout
        y = self.apply_dropout(h, 2, 0)

        return y, h

    @staticmethod
    def mask_name(idx):
        return 'drop_mask_{}'.format(idx)

    def set_drop_masks(self):
        """Set the dropout masks for the current sequence"""
        for idx, mask in self.yield_drop_masks():
            setattr(self, self.mask_name(idx), mask)

    def yield_drop_masks(self):
        """Iterator over recurrent dropout masks"""
        n_masks = (3, 3, 1)
        n_channels = (self.input_ch, self.hidden_ch, self.hidden_ch)
        for idx, p in enumerate(self.drop_prob):
            if p > 0:
                yield (idx, self.generate_do_mask(
                    p, n_masks[idx], n_channels[idx]))

    @staticmethod
    def generate_do_mask(p, n, ch):
        """Generate a dropout mask for recurrent dropout"""
        with torch.no_grad():
            mask = Bernoulli(torch.full((n, ch), 1 - p)).sample() / (1 - p)
            mask = mask.requires_grad_(False).cuda()
            return mask

    def apply_dropout(self, x, idx, sub_idx):
        """Apply recurrent or naive dropout"""
        if self.training and self.drop_prob[idx] > 0 and idx != 2:
            if self.do_mode == 'recurrent':
                x = x.clone() * torch.reshape(
                    getattr(self, self.mask_name(idx))
                    [sub_idx, :], (1, -1, 1, 1))
            elif self.do_mode == 'naive':
                x = f.dropout2d(
                    x, self.drop_prob[idx], self.training, inplace=False)
        else:
            x = x.clone()
        return x

    def get_norm_module(self, channels):
        """Return normalization module instance"""
        norm_module = None
        if self.batchnorm is not None:
            norm_module = self.batchnorm(channels)
        elif self.norm == 'instance':
            norm_module = nn.InstanceNorm2d(channels, **self.norm_kwargs)
        elif self.norm == 'batch':
            norm_module = nn.BatchNorm2d(channels, **self.norm_kwargs)
        return norm_module

    def _conv2d(self, in_channels, kernel_size, bias=True):
        """
        Return convolutional layer.
        Supports standard convolutions and MobileNet-style convolutions.
        """
        padding = tuple(k_size // 2 for k_size in kernel_size)
        if not self.mobile or kernel_size == (1, 1):
            return nn.Conv2d(in_channels, self.hidden_ch, kernel_size,
                             padding=padding, bias=bias)
        else:
            return nn.Sequential(OrderedDict([
                ('conv_dw', nn.Conv2d(
                    in_channels, in_channels, kernel_size=kernel_size,
                    padding=padding, groups=in_channels, bias=False)),
                ('sep_bn', self.get_norm_module(in_channels)),
                ('sep_relu', nn.ReLU6()),
                ('conv_sep', nn.Conv2d(
                    in_channels, self.hidden_ch, 1, bias=bias)),
            ]))

    def _init_hidden(self, input_, cuda=True):
        """Initialize the hidden state"""
        batch_size, _, height, width = input_.data.size()
        prev_state = torch.zeros(
            batch_size, self.hidden_ch, height, width)
        if cuda:
            prev_state = prev_state.cuda()
        return prev_state


class ConvGRU(nn.Module):

    def __init__(self, input_channels=None, hidden_channels=None,
                 kernel_size=(3, 3), gate_ksize=(1, 1),
                 dropout=(False, False, False), drop_prob=(0.5, 0.5, 0.5),
                 **kwargs):
        """
        Generates a multi-layer convolutional GRU.
        Preserves spatial dimensions across cells, only altering depth.

        Arguments:
            input_channels: Number of channels of the input.
            hidden_channels (list): List of hidden channels for each layer.
            kernel_size (tuple): Kernel size of the U and W operations.
            gate_ksize (tuple): Kernel size for the gates.
            dropout: Tuple of Booleans for input, recurrent and output dropout.
            drop_prob: Tuple of dropout probabilities for each selected dropout.
            kwargs: Additional parameters for the cGRU cells.
        """

        super().__init__()

        kernel_size = tuple(kernel_size)
        gate_ksize = tuple(gate_ksize)
        dropout = tuple(dropout)
        drop_prob = tuple(drop_prob)

        assert len(hidden_channels) > 0
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.num_layers = len(hidden_channels)
        self._check_kernel_size_consistency(kernel_size)
        self._check_kernel_size_consistency(gate_ksize)
        self.kernel_size = self._extend_for_multilayer(kernel_size)
        self.gate_ksize = self._extend_for_multilayer(gate_ksize)
        self.dropout = self._extend_for_multilayer(dropout)
        drop_prob = self._extend_for_multilayer(drop_prob)
        self.drop_prob = [tuple(dp_ if do_ else 0. for dp_, do_ in zip(dp, do))
                          for dp, do in zip(drop_prob, self.dropout)]
        self.kwargs = kwargs

        cell_list = []
        for idx in range(self.num_layers):
            if idx < self.num_layers - 1:
                # Switch output dropout off for hidden layers.
                # Otherwise it would confict with input dropout.
                this_drop_prob = self.drop_prob[idx][:2] + (0.,)
            else:
                this_drop_prob = self.drop_prob[idx]
            cell_list.append(ConvGRUCell(
                self.input_channels[idx], self.hidden_channels[idx],
                self.kernel_size[idx], drop_prob=this_drop_prob,
                gate_ksize=self.gate_ksize[idx], **kwargs))
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden=None):
        """
        Args:
            input_tensor:
                5-D Tensor of shape (b, t, c, h, w)
            hidden:
                optional initial hiddens state

        Returns:
            outputs
        """
        if not hidden:
            hidden = [None] * self.num_layers

        outputs = []
        iterator = torch.unbind(input_tensor, dim=1)

        for t, x in enumerate(iterator):
            for layer_idx in range(self.num_layers):
                if self.cell_list[layer_idx].do_mode == 'recurrent'\
                        and t == 0:
                    self.cell_list[layer_idx].set_drop_masks()
                (x, h) = self.cell_list[layer_idx](x, hidden[layer_idx])
                hidden[layer_idx] = h.clone()
            outputs.append(x.clone())
        outputs = torch.stack(outputs, dim=1)

        return outputs, hidden

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and
                 all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    def _extend_for_multilayer(self, param):
        if not isinstance(param, list):
            param = [param] * self.num_layers
        else:
            assert(len(param) == self.num_layers)
        return param
