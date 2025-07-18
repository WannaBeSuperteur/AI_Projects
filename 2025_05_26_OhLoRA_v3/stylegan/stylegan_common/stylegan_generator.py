# Modified implementation from https://github.com/genforce/genforce/blob/master/models/stylegan_generator.py


# python3.7
"""Contains the implementation of generator described in StyleGAN.

Paper: https://arxiv.org/pdf/1812.04948.pdf

Official TensorFlow implementation: https://github.com/NVlabs/stylegan
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sync_op import all_gather

__all__ = ['StyleGANGenerator']

# Resolutions allowed.
_RESOLUTIONS_ALLOWED = [8, 16, 32, 64, 128, 256, 512, 1024]

# Initial resolution.
_INIT_RES = 4

# Fused-scale options allowed.
_FUSED_SCALE_ALLOWED = [True, False, 'auto']

# Minimal resolution for `auto` fused-scale strategy.
_AUTO_FUSED_SCALE_MIN_RES = 128

# Default gain factor for weight scaling.
_WSCALE_GAIN = np.sqrt(2.0)
_STYLEMOD_WSCALE_GAIN = 1.0


class StyleGANGenerator(nn.Module):
    """Defines the generator network in StyleGAN.

    NOTE: The synthesized images are with `RGB` channel order and pixel range
    [-1, 1].

    Settings for the mapping network:

    (1) z_space_dim: Dimension of the input latent space, Z.
        (default: 512 -> change to 128 is not compatible for pre-trained model)
    (2) w_space_dim: Dimension of the output latent space, W.
        (default: 512 -> change to 128 is not compatible for pre-trained model)
    (3) label_size: Size of the additional label for conditional generation.
        (default: 0)
    (4）mapping_layers: Number of layers of the mapping network. (default: 8)
    (5) mapping_fmaps: Number of hidden channels of the mapping network.
        (default: 512)
    (6) mapping_lr_mul: Learning rate multiplier for the mapping network.
        (default: 0.01)
    (7) repeat_w: Repeat w-code for different layers.

    Settings for the synthesis network:

    (1) resolution: The resolution of the output image.
    (2) image_channels: Number of channels of the output image. (default: 3)
    (3) final_tanh: Whether to use `tanh` to control the final pixel range.
        (default: False)
    (4) const_input: Whether to use a constant in the first convolutional layer.
        (default: True)
    (5) fused_scale: Whether to fused `upsample` and `conv2d` together,
        resulting in `conv2d_transpose`. (default: `auto`)
    (6) use_wscale: Whether to use weight scaling. (default: True)
    (7) noise_type: Type of noise added to the convolutional results at each
        layer. (default: `spatial`)
    (8) fmaps_base: Factor to control number of feature maps for each layer.
        (default: 16 << 10)
    (9) fmaps_max: Maximum number of feature maps in each layer. (default: 512)
    """

    def __init__(self,
                 resolution,
                 z_space_dim=512,       # 512 in Original code
                 w_space_dim=512,       # 512 in Original code
                 label_size=7,          # originally (eyes, hair_color, hair_length, mouth, ..., back_std) property
                 mapping_layers=8,
                 mapping_fmaps=512,
                 mapping_lr_mul=0.01,
                 repeat_w=True,
                 image_channels=3,
                 final_tanh=False,
                 const_input=True,
                 fused_scale='auto',
                 use_wscale=True,
                 noise_type='spatial',
                 fmaps_base=16 << 10,
                 fmaps_max=512):
        """Initializes with basic settings.

        Raises:
            ValueError: If the `resolution` is not supported, or `fused_scale`
                is not supported.
        """
        super().__init__()

        if resolution not in _RESOLUTIONS_ALLOWED:
            raise ValueError(f'Invalid resolution: `{resolution}`!\n'
                             f'Resolutions allowed: {_RESOLUTIONS_ALLOWED}.')
        if fused_scale not in _FUSED_SCALE_ALLOWED:
            raise ValueError(f'Invalid fused-scale option: `{fused_scale}`!\n'
                             f'Options allowed: {_FUSED_SCALE_ALLOWED}.')

        self.init_res = _INIT_RES
        self.resolution = resolution
        self.z_space_dim = z_space_dim
        self.w_space_dim = w_space_dim
        self.label_size = label_size
        self.mapping_layers = mapping_layers
        self.mapping_fmaps = mapping_fmaps
        self.mapping_lr_mul = mapping_lr_mul
        self.repeat_w = repeat_w
        self.image_channels = image_channels
        self.final_tanh = final_tanh
        self.const_input = const_input
        self.fused_scale = fused_scale
        self.use_wscale = use_wscale
        self.noise_type = noise_type
        self.fmaps_base = fmaps_base
        self.fmaps_max = fmaps_max

        self.num_layers = int(np.log2(self.resolution // self.init_res * 2)) * 2

        if self.repeat_w:
            self.mapping_space_dim = self.w_space_dim
        else:
            self.mapping_space_dim = self.w_space_dim * self.num_layers
        self.mapping = MappingModule(input_space_dim=self.z_space_dim,
                                     hidden_space_dim=self.mapping_fmaps,
                                     final_space_dim=self.mapping_space_dim,
                                     label_size=self.label_size,
                                     num_layers=self.mapping_layers,
                                     use_wscale=self.use_wscale,
                                     lr_mul=self.mapping_lr_mul)

        self.truncation = TruncationModule(w_space_dim=self.w_space_dim,
                                           num_layers=self.num_layers,
                                           repeat_w=self.repeat_w)

        self.synthesis = SynthesisModule(resolution=self.resolution,
                                         init_resolution=self.init_res,
                                         w_space_dim=self.w_space_dim,
                                         image_channels=self.image_channels,
                                         final_tanh=self.final_tanh,
                                         const_input=self.const_input,
                                         fused_scale=self.fused_scale,
                                         use_wscale=self.use_wscale,
                                         noise_type=self.noise_type,
                                         fmaps_base=self.fmaps_base,
                                         fmaps_max=self.fmaps_max)

        self.pth_to_tf_var_mapping = {}
        for key, val in self.mapping.pth_to_tf_var_mapping.items():
            self.pth_to_tf_var_mapping[f'mapping.{key}'] = val
        for key, val in self.truncation.pth_to_tf_var_mapping.items():
            self.pth_to_tf_var_mapping[f'truncation.{key}'] = val
        for key, val in self.synthesis.pth_to_tf_var_mapping.items():
            self.pth_to_tf_var_mapping[f'synthesis.{key}'] = val

    def set_space_of_latent(self, space_of_latent='w'):
        """Sets the space to which the latent code belong.

        This function is particually used for choosing how to inject the latent
        code into the convolutional layers. The original generator will take a
        W-Space code and apply it for style modulation after an affine
        transformation. But, sometimes, it may need to directly feed an already
        affine-transformed code into the convolutional layer, e.g., when
        training an encoder for GAN inversion. We term the transformed space as
        Style Space (or Y-Space). This function is designed to tell the
        convolutional layers how to use the input code.

        Args:
            space_of_latent: The space to which the latent code belong. Case
                insensitive. (default: 'w')
        """
        for module in self.modules():
            if isinstance(module, StyleModLayer):
                setattr(module, 'space_of_latent', space_of_latent)

    def forward(self,
                z,
                label,  # originally (eyes, hair_color, hair_length, mouth, pose, back_mean, back_std) property score
                lod=None,
                w_moving_decay=0.995,
                style_mixing_prob=0.0,
                trunc_psi=None,
                trunc_layers=None,
                randomize_noise=False,
                **_unused_kwargs):
        mapping_results = self.mapping(z, label)
        w = mapping_results['w']

        if self.training and w_moving_decay < 1:
            batch_w_avg = all_gather(w).mean(dim=0)
            self.truncation.w_avg.copy_(
                self.truncation.w_avg * w_moving_decay +
                batch_w_avg * (1 - w_moving_decay))

        if self.training and style_mixing_prob > 0:
            new_z = torch.randn_like(z)
            new_w = self.mapping(new_z, label)['w']
            lod = self.synthesis.lod.cpu().tolist() if lod is None else lod
            current_layers = self.num_layers - int(lod) * 2
            if np.random.uniform() < style_mixing_prob:
                mixing_cutoff = np.random.randint(1, current_layers)
                w = self.truncation(w)
                new_w = self.truncation(new_w)
                w[:, mixing_cutoff:] = new_w[:, mixing_cutoff:]

        wp = self.truncation(w, trunc_psi, trunc_layers)
        synthesis_results = self.synthesis(wp, lod, randomize_noise)

        return {**mapping_results, **synthesis_results}


class MappingModule(nn.Module):
    """Implements the latent space mapping module.

    Basically, this module executes several dense layers in sequence.
    """

    def __init__(self,
                 input_space_dim=512,
                 hidden_space_dim=512,
                 final_space_dim=512,
                 label_convert_dim=16,
                 label_size=7,  # originally (eyes, hair_color, hair_length, mouth, ..., back_std) property score
                 num_layers=8,
                 normalize_input=True,
                 use_wscale=True,
                 lr_mul=0.01):
        super().__init__()

        self.input_space_dim = input_space_dim
        self.hidden_space_dim = hidden_space_dim
        self.final_space_dim = final_space_dim
        self.label_size = label_size
        self.num_layers = num_layers
        self.normalize_input = normalize_input
        self.use_wscale = use_wscale
        self.lr_mul = lr_mul

        self.norm = PixelNormLayer() if self.normalize_input else nn.Identity()

        self.pth_to_tf_var_mapping = {}
        for i in range(num_layers):
            dim_mul = 2 if label_size else 1
            in_channels = (input_space_dim * dim_mul if i == 0 else
                           hidden_space_dim)
            out_channels = (final_space_dim if i == (num_layers - 1) else
                            hidden_space_dim)
            self.add_module(f'dense{i}',
                            DenseBlock(in_channels=input_space_dim + label_convert_dim if i == 0 else in_channels,
                                       out_channels=out_channels,
                                       use_wscale=self.use_wscale,
                                       lr_mul=self.lr_mul))
            self.pth_to_tf_var_mapping[f'dense{i}.weight'] = f'Dense{i}/weight'
            self.pth_to_tf_var_mapping[f'dense{i}.bias'] = f'Dense{i}/bias'
        if label_size:
            self.label_weight = nn.Parameter(
                torch.randn(label_size, label_convert_dim))
            self.pth_to_tf_var_mapping[f'label_weight'] = f'LabelConcat/weight'

    def forward(self, z, label):
        if z.ndim != 2 or z.shape[1] != self.input_space_dim:
            raise ValueError(f'Input latent code should be with shape '
                             f'[batch_size, input_dim], where '
                             f'`input_dim` equals to {self.input_space_dim}!\n'
                             f'But `{z.shape}` is received!')

        if label.ndim != 2 or label.shape != (z.shape[0], self.label_size):
            raise ValueError(f'Input label should be with shape '
                             f'[batch_size, label_size], where '
                             f'`batch_size` equals to that of '
                             f'latent codes ({z.shape[0]}) and '
                             f'`label_size` equals to {self.label_size}!\n'
                             f'But `{label.shape}` is received!')

        embedding = torch.matmul(label, self.label_weight)
        z = torch.cat((z, embedding), dim=1)

        z = self.norm(z)
        w = z
        for i in range(self.num_layers):
            w = self.__getattr__(f'dense{i}')(w)
        results = {
            'z': z,
            'label': label,
            'w': w,
        }
        if self.label_size:
            results['embedding'] = embedding
        return results


class TruncationModule(nn.Module):
    """Implements the truncation module.

    Truncation is executed as follows:

    For layers in range [0, truncation_layers), the truncated w-code is computed
    as

    w_new = w_avg + (w - w_avg) * truncation_psi

    To disable truncation, please set
    (1) truncation_psi = 1.0 (None) OR
    (2) truncation_layers = 0 (None)

    NOTE: The returned tensor is layer-wise style codes.
    """

    def __init__(self, w_space_dim, num_layers, repeat_w=True):
        super().__init__()

        self.num_layers = num_layers
        self.w_space_dim = w_space_dim
        self.repeat_w = repeat_w

        if self.repeat_w:
            self.register_buffer('w_avg', torch.zeros(w_space_dim))
        else:
            self.register_buffer('w_avg', torch.zeros(num_layers * w_space_dim))
        self.pth_to_tf_var_mapping = {'w_avg': 'dlatent_avg'}

    def forward(self, w, trunc_psi=None, trunc_layers=None):
        if w.ndim == 2:
            if self.repeat_w and w.shape[1] == self.w_space_dim:
                w = w.view(-1, 1, self.w_space_dim)
                wp = w.repeat(1, self.num_layers, 1)
            else:
                assert w.shape[1] == self.w_space_dim * self.num_layers
                wp = w.view(-1, self.num_layers, self.w_space_dim)
        else:
            wp = w
        assert wp.ndim == 3
        assert wp.shape[1:] == (self.num_layers, self.w_space_dim)

        trunc_psi = 1.0 if trunc_psi is None else trunc_psi
        trunc_layers = 0 if trunc_layers is None else trunc_layers
        if trunc_psi < 1.0 and trunc_layers > 0:
            layer_idx = np.arange(self.num_layers).reshape(1, -1, 1)
            coefs = np.ones_like(layer_idx, dtype=np.float32)
            coefs[layer_idx < trunc_layers] *= trunc_psi
            coefs = torch.from_numpy(coefs).to(wp)
            w_avg = self.w_avg.view(1, -1, self.w_space_dim)
            wp = w_avg + (wp - w_avg) * coefs
        return wp


class SynthesisModule(nn.Module):
    """Implements the image synthesis module.

    Basically, this module executes several convolutional layers in sequence.
    """

    def __init__(self,
                 resolution=1024,
                 init_resolution=4,
                 w_space_dim=512,
                 image_channels=3,
                 final_tanh=False,
                 const_input=True,
                 fused_scale='auto',
                 use_wscale=True,
                 noise_type='spatial',
                 fmaps_base=16 << 10,
                 fmaps_max=512):
        super().__init__()

        self.init_res = init_resolution
        self.init_res_log2 = int(np.log2(self.init_res))
        self.resolution = resolution
        self.final_res_log2 = int(np.log2(self.resolution))
        self.w_space_dim = w_space_dim
        self.image_channels = image_channels
        self.final_tanh = final_tanh
        self.const_input = const_input
        self.fused_scale = fused_scale
        self.use_wscale = use_wscale
        self.noise_type = noise_type
        self.fmaps_base = fmaps_base
        self.fmaps_max = fmaps_max

        self.num_layers = (self.final_res_log2 - self.init_res_log2 + 1) * 2

        # Level of detail (used for progressive training).
        self.register_buffer('lod', torch.zeros(()))
        self.pth_to_tf_var_mapping = {'lod': 'lod'}

        for res_log2 in range(self.init_res_log2, self.final_res_log2 + 1):
            res = 2 ** res_log2
            block_idx = res_log2 - self.init_res_log2

            # First convolution layer for each resolution.
            layer_name = f'layer{2 * block_idx}'
            if res == self.init_res:
                if self.const_input:
                    self.add_module(layer_name,
                                    ConvBlock(in_channels=self.get_nf(res),
                                              out_channels=self.get_nf(res),
                                              resolution=self.init_res,
                                              w_space_dim=self.w_space_dim,
                                              position='const_init',
                                              use_wscale=self.use_wscale,
                                              noise_type=self.noise_type))
                    tf_layer_name = 'Const'
                    self.pth_to_tf_var_mapping[f'{layer_name}.const'] = (
                        f'{res}x{res}/{tf_layer_name}/const')
                else:
                    self.add_module(layer_name,
                                    ConvBlock(in_channels=self.w_space_dim,
                                              out_channels=self.get_nf(res),
                                              resolution=self.init_res,
                                              w_space_dim=self.w_space_dim,
                                              kernel_size=self.init_res,
                                              padding=self.init_res - 1,
                                              use_wscale=self.use_wscale,
                                              noise_type=self.noise_type))
                    tf_layer_name = 'Dense'
                    self.pth_to_tf_var_mapping[f'{layer_name}.weight'] = (
                        f'{res}x{res}/{tf_layer_name}/weight')
            else:
                if self.fused_scale == 'auto':
                    fused_scale = (res >= _AUTO_FUSED_SCALE_MIN_RES)
                else:
                    fused_scale = self.fused_scale
                self.add_module(layer_name,
                                ConvBlock(in_channels=self.get_nf(res // 2),
                                          out_channels=self.get_nf(res),
                                          resolution=res,
                                          w_space_dim=self.w_space_dim,
                                          upsample=True,
                                          fused_scale=fused_scale,
                                          use_wscale=self.use_wscale,
                                          noise_type=self.noise_type))
                tf_layer_name = 'Conv0_up'
                self.pth_to_tf_var_mapping[f'{layer_name}.weight'] = (
                    f'{res}x{res}/{tf_layer_name}/weight')
            self.pth_to_tf_var_mapping[f'{layer_name}.bias'] = (
                f'{res}x{res}/{tf_layer_name}/bias')
            self.pth_to_tf_var_mapping[f'{layer_name}.style.weight'] = (
                f'{res}x{res}/{tf_layer_name}/StyleMod/weight')
            self.pth_to_tf_var_mapping[f'{layer_name}.style.bias'] = (
                f'{res}x{res}/{tf_layer_name}/StyleMod/bias')
            self.pth_to_tf_var_mapping[f'{layer_name}.apply_noise.weight'] = (
                f'{res}x{res}/{tf_layer_name}/Noise/weight')
            self.pth_to_tf_var_mapping[f'{layer_name}.apply_noise.noise'] = (
                f'noise{2 * block_idx}')

            # Second convolution layer for each resolution.
            layer_name = f'layer{2 * block_idx + 1}'
            self.add_module(layer_name,
                            ConvBlock(in_channels=self.get_nf(res),
                                      out_channels=self.get_nf(res),
                                      resolution=res,
                                      w_space_dim=self.w_space_dim,
                                      use_wscale=self.use_wscale,
                                      noise_type=self.noise_type))
            tf_layer_name = 'Conv' if res == self.init_res else 'Conv1'
            self.pth_to_tf_var_mapping[f'{layer_name}.weight'] = (
                f'{res}x{res}/{tf_layer_name}/weight')
            self.pth_to_tf_var_mapping[f'{layer_name}.bias'] = (
                f'{res}x{res}/{tf_layer_name}/bias')
            self.pth_to_tf_var_mapping[f'{layer_name}.style.weight'] = (
                f'{res}x{res}/{tf_layer_name}/StyleMod/weight')
            self.pth_to_tf_var_mapping[f'{layer_name}.style.bias'] = (
                f'{res}x{res}/{tf_layer_name}/StyleMod/bias')
            self.pth_to_tf_var_mapping[f'{layer_name}.apply_noise.weight'] = (
                f'{res}x{res}/{tf_layer_name}/Noise/weight')
            self.pth_to_tf_var_mapping[f'{layer_name}.apply_noise.noise'] = (
                f'noise{2 * block_idx + 1}')

            # Output convolution layer for each resolution.
            self.add_module(f'output{block_idx}',
                            ConvBlock(in_channels=self.get_nf(res),
                                      out_channels=self.image_channels,
                                      resolution=res,
                                      w_space_dim=self.w_space_dim,
                                      position='last',
                                      kernel_size=1,
                                      padding=0,
                                      use_wscale=self.use_wscale,
                                      wscale_gain=1.0,
                                      activation_type='linear'))
            self.pth_to_tf_var_mapping[f'output{block_idx}.weight'] = (
                f'ToRGB_lod{self.final_res_log2 - res_log2}/weight')
            self.pth_to_tf_var_mapping[f'output{block_idx}.bias'] = (
                f'ToRGB_lod{self.final_res_log2 - res_log2}/bias')

        self.upsample = UpsamplingLayer()
        self.final_activate = nn.Tanh() if final_tanh else nn.Identity()

    def get_nf(self, res):
        """Gets number of feature maps according to current resolution."""
        return min(self.fmaps_base // res, self.fmaps_max)

    def forward(self, wp, lod=None, randomize_noise=False):
        lod = self.lod.cpu().tolist() if lod is None else lod
        if lod + self.init_res_log2 > self.final_res_log2:
            raise ValueError(f'Maximum level-of-detail (lod) is '
                             f'{self.final_res_log2 - self.init_res_log2}, '
                             f'but `{lod}` is received!')

        results = {'wp': wp}
        for res_log2 in range(self.init_res_log2, self.final_res_log2 + 1):
            current_lod = self.final_res_log2 - res_log2
            if lod < current_lod + 1:
                block_idx = res_log2 - self.init_res_log2
                if block_idx == 0:
                    if self.const_input:
                        x, style = self.layer0(None, wp[:, 0], randomize_noise)
                    else:
                        x = wp[:, 0].view(-1, self.w_space_dim, 1, 1)
                        x, style = self.layer0(x, wp[:, 0], randomize_noise)
                else:
                    x, style = self.__getattr__(f'layer{2 * block_idx}')(
                        x, wp[:, 2 * block_idx], randomize_noise)
                results[f'style{2 * block_idx:02d}'] = style
                x, style = self.__getattr__(f'layer{2 * block_idx + 1}')(
                    x, wp[:, 2 * block_idx + 1], randomize_noise)
                results[f'style{2 * block_idx + 1:02d}'] = style
            if current_lod - 1 < lod <= current_lod:
                image = self.__getattr__(f'output{block_idx}')(x, None)
            elif current_lod < lod < current_lod + 1:
                alpha = np.ceil(lod) - lod
                image = (self.__getattr__(f'output{block_idx}')(x, None) * alpha
                         + self.upsample(image) * (1 - alpha))
            elif lod >= current_lod + 1:
                image = self.upsample(image)
        results['image'] = self.final_activate(image)
        return results


class PixelNormLayer(nn.Module):
    """Implements pixel-wise feature vector normalization layer."""

    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.eps = epsilon

    def forward(self, x):
        norm = torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.eps)
        return x / norm


class InstanceNormLayer(nn.Module):
    """Implements instance normalization layer."""

    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.eps = epsilon

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError(f'The input tensor should be with shape '
                             f'[batch_size, channel, height, width], '
                             f'but `{x.shape}` is received!')
        x = x - torch.mean(x, dim=[2, 3], keepdim=True)
        norm = torch.sqrt(
            torch.mean(x ** 2, dim=[2, 3], keepdim=True) + self.eps)
        return x / norm


class UpsamplingLayer(nn.Module):
    """Implements the upsampling layer.

    Basically, this layer can be used to upsample feature maps with nearest
    neighbor interpolation.
    """

    def __init__(self, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        if self.scale_factor <= 1:
            return x
        return F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')


class Blur(torch.autograd.Function):
    """Defines blur operation with customized gradient computation."""

    @staticmethod
    def forward(ctx, x, kernel):
        ctx.save_for_backward(kernel)
        y = F.conv2d(input=x,
                     weight=kernel,
                     bias=None,
                     stride=1,
                     padding=1,
                     groups=x.shape[1])
        return y

    @staticmethod
    def backward(ctx, dy):
        kernel, = ctx.saved_tensors
        dx = F.conv2d(input=dy,
                      weight=kernel.flip((2, 3)),
                      bias=None,
                      stride=1,
                      padding=1,
                      groups=dy.shape[1])
        return dx, None, None


class BlurLayer(nn.Module):
    """Implements the blur layer."""

    def __init__(self,
                 channels,
                 kernel=(1, 2, 1),
                 normalize=True):
        super().__init__()
        kernel = np.array(kernel, dtype=np.float32).reshape(1, -1)
        kernel = kernel.T.dot(kernel)
        if normalize:
            kernel /= np.sum(kernel)
        kernel = kernel[np.newaxis, np.newaxis]
        kernel = np.tile(kernel, [channels, 1, 1, 1])
        self.register_buffer('kernel', torch.from_numpy(kernel))

    def forward(self, x):
        return Blur.apply(x, self.kernel)


class NoiseApplyingLayer(nn.Module):
    """Implements the noise applying layer."""

    def __init__(self, resolution, channels, noise_type='spatial'):
        super().__init__()
        self.noise_type = noise_type.lower()
        self.res = resolution
        self.channels = channels
        if self.noise_type == 'spatial':
            self.register_buffer('noise', torch.randn(1, 1, self.res, self.res))
            self.weight = nn.Parameter(torch.zeros(self.channels))
        elif self.noise_type == 'channel':
            self.register_buffer('noise', torch.randn(1, self.channels, 1, 1))
            self.weight = nn.Parameter(torch.zeros(self.res, self.res))
        else:
            raise NotImplementedError(f'Not implemented noise type: '
                                      f'`{self.noise_type}`!')

    def forward(self, x, randomize_noise=False):
        if x.ndim != 4:
            raise ValueError(f'The input tensor should be with shape '
                             f'[batch_size, channel, height, width], '
                             f'but `{x.shape}` is received!')
        if randomize_noise:
            if self.noise_type == 'spatial':
                noise = torch.randn(x.shape[0], 1, self.res, self.res).to(x)
            elif self.noise_type == 'channel':
                noise = torch.randn(x.shape[0], self.channels, 1, 1).to(x)
        else:
            noise = self.noise

        if self.noise_type == 'spatial':
            x = x + noise * self.weight.view(1, self.channels, 1, 1)
        elif self.noise_type == 'channel':
            x = x + noise * self.weight.view(1, 1, self.res, self.res)
        return x


class StyleModLayer(nn.Module):
    """Implements the style modulation layer."""

    def __init__(self,
                 w_space_dim,
                 out_channels,
                 use_wscale=True):
        super().__init__()
        self.w_space_dim = w_space_dim
        self.out_channels = out_channels

        weight_shape = (self.out_channels * 2, self.w_space_dim)
        wscale = _STYLEMOD_WSCALE_GAIN / np.sqrt(self.w_space_dim)
        if use_wscale:
            self.weight = nn.Parameter(torch.randn(*weight_shape))
            self.wscale = wscale
        else:
            self.weight = nn.Parameter(torch.randn(*weight_shape) * wscale)
            self.wscale = 1.0

        self.bias = nn.Parameter(torch.zeros(self.out_channels * 2))
        self.space_of_latent = 'w'

    def forward_style(self, w):
        """Gets style code from the given input.

        More specifically, if the input is from W-Space, it will be projected by
        an affine transformation. If it is from the Style Space (Y-Space), no
        operation is required.

        NOTE: For codes from Y-Space, we use slicing to make sure the dimension
        is correct, in case that the code is padded before fed into this layer.
        """
        if self.space_of_latent == 'w':
            if w.ndim != 2 or w.shape[1] != self.w_space_dim:
                raise ValueError(f'The input tensor should be with shape '
                                 f'[batch_size, w_space_dim], where '
                                 f'`w_space_dim` equals to '
                                 f'{self.w_space_dim}!\n'
                                 f'But `{w.shape}` is received!')
            style = F.linear(w,
                             weight=self.weight * self.wscale,
                             bias=self.bias)
        elif self.space_of_latent == 'y':
            if w.ndim != 2 or w.shape[1] < 2 * self.out_channels:
                raise ValueError(f'The input tensor should be with shape '
                                 f'[batch_size, y_space_dim], where '
                                 f'`y_space_dim` equals to '
                                 f'{2 * self.out_channels}!\n'
                                 f'But `{w.shape}` is received!')
            style = w[:, :2 * self.out_channels]
        return style

    def forward(self, x, w):
        style = self.forward_style(w)
        style_split = style.view(-1, 2, self.out_channels, 1, 1)
        x = x * (style_split[:, 0] + 1) + style_split[:, 1]
        return x, style


class ConvBlock(nn.Module):
    """Implements the normal convolutional block.

    Basically, this block executes upsampling layer (if needed), convolutional
    layer, blurring layer, noise applying layer, activation layer, instance
    normalization layer, and style modulation layer in sequence.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 resolution,
                 w_space_dim,
                 position=None,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 add_bias=True,
                 upsample=False,
                 fused_scale=False,
                 use_wscale=True,
                 wscale_gain=_WSCALE_GAIN,
                 lr_mul=1.0,
                 activation_type='lrelu',
                 noise_type='spatial'):
        """Initializes with block settings.

        Args:
            in_channels: Number of channels of the input tensor.
            out_channels: Number of channels of the output tensor.
            resolution: Resolution of the output tensor.
            w_space_dim: Dimension of W space for style modulation.
            position: Position of the layer. `const_init`, `last` would lead to
                different behavior. (default: None)
            kernel_size: Size of the convolutional kernels. (default: 3)
            stride: Stride parameter for convolution operation. (default: 1)
            padding: Padding parameter for convolution operation. (default: 1)
            add_bias: Whether to add bias onto the convolutional result.
                (default: True)
            upsample: Whether to upsample the input tensor before convolution.
                (default: False)
            fused_scale: Whether to fused `upsample` and `conv2d` together,
                resulting in `conv2d_transpose`. (default: False)
            use_wscale: Whether to use weight scaling. (default: True)
            wscale_gain: Gain factor for weight scaling. (default: _WSCALE_GAIN)
            lr_mul: Learning multiplier for both weight and bias. (default: 1.0)
            activation_type: Type of activation. Support `linear` and `lrelu`.
                (default: `lrelu`)
            noise_type: Type of noise added to the feature map after the
                convolution (if needed). Support `spatial` and `channel`.
                (default: `spatial`)

        Raises:
            NotImplementedError: If the `activation_type` is not supported.
        """
        super().__init__()

        self.position = position

        if add_bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
            self.bscale = lr_mul
        else:
            self.bias = None

        if activation_type == 'linear':
            self.activate = nn.Identity()
        elif activation_type == 'lrelu':
            self.activate = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            raise NotImplementedError(f'Not implemented activation function: '
                                      f'`{activation_type}`!')

        if self.position != 'last':
            self.apply_noise = NoiseApplyingLayer(
                resolution, out_channels, noise_type=noise_type)
            self.normalize = InstanceNormLayer()
            self.style = StyleModLayer(w_space_dim, out_channels, use_wscale)

        if self.position == 'const_init':
            self.const = nn.Parameter(
                torch.ones(1, in_channels, resolution, resolution))
            return

        self.blur = BlurLayer(out_channels) if upsample else nn.Identity()

        if upsample and not fused_scale:
            self.upsample = UpsamplingLayer()
        else:
            self.upsample = nn.Identity()

        if upsample and fused_scale:
            self.use_conv2d_transpose = True
            self.stride = 2
            self.padding = 1
        else:
            self.use_conv2d_transpose = False
            self.stride = stride
            self.padding = padding

        weight_shape = (out_channels, in_channels, kernel_size, kernel_size)
        fan_in = kernel_size * kernel_size * in_channels
        wscale = wscale_gain / np.sqrt(fan_in)
        if use_wscale:
            self.weight = nn.Parameter(torch.randn(*weight_shape) / lr_mul)
            self.wscale = wscale * lr_mul
        else:
            self.weight = nn.Parameter(
                torch.randn(*weight_shape) * wscale / lr_mul)
            self.wscale = lr_mul

    def forward(self, x, w, randomize_noise=False):
        if self.position != 'const_init':
            x = self.upsample(x)
            weight = self.weight * self.wscale
            if self.use_conv2d_transpose:
                weight = F.pad(weight, (1, 1, 1, 1, 0, 0, 0, 0), 'constant', 0)
                weight = (weight[:, :, 1:, 1:] + weight[:, :, :-1, 1:] +
                          weight[:, :, 1:, :-1] + weight[:, :, :-1, :-1])
                weight = weight.permute(1, 0, 2, 3)
                x = F.conv_transpose2d(x,
                                       weight=weight,
                                       bias=None,
                                       stride=self.stride,
                                       padding=self.padding)
            else:
                x = F.conv2d(x,
                             weight=weight,
                             bias=None,
                             stride=self.stride,
                             padding=self.padding)
            x = self.blur(x)
        else:
            x = self.const.repeat(w.shape[0], 1, 1, 1)

        bias = self.bias * self.bscale if self.bias is not None else None

        if self.position == 'last':
            if bias is not None:
                x = x + bias.view(1, -1, 1, 1)
            return x

        x = self.apply_noise(x, randomize_noise)
        if bias is not None:
            x = x + bias.view(1, -1, 1, 1)
        x = self.activate(x)
        x = self.normalize(x)
        x, style = self.style(x, w)
        return x, style


class DenseBlock(nn.Module):
    """Implements the dense block.

    Basically, this block executes fully-connected layer and activation layer.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 add_bias=True,
                 use_wscale=True,
                 wscale_gain=_WSCALE_GAIN,
                 lr_mul=1.0,
                 activation_type='lrelu'):
        """Initializes with block settings.

        Args:
            in_channels: Number of channels of the input tensor.
            out_channels: Number of channels of the output tensor.
            add_bias: Whether to add bias onto the fully-connected result.
                (default: True)
            use_wscale: Whether to use weight scaling. (default: True)
            wscale_gain: Gain factor for weight scaling. (default: _WSCALE_GAIN)
            lr_mul: Learning multiplier for both weight and bias. (default: 1.0)
            activation_type: Type of activation. Support `linear` and `lrelu`.
                (default: `lrelu`)

        Raises:
            NotImplementedError: If the `activation_type` is not supported.
        """
        super().__init__()
        weight_shape = (out_channels, in_channels)
        wscale = wscale_gain / np.sqrt(in_channels)
        if use_wscale:
            self.weight = nn.Parameter(torch.randn(*weight_shape) / lr_mul)
            self.wscale = wscale * lr_mul
        else:
            self.weight = nn.Parameter(
                torch.randn(*weight_shape) * wscale / lr_mul)
            self.wscale = lr_mul

        if add_bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
            self.bscale = lr_mul
        else:
            self.bias = None

        if activation_type == 'linear':
            self.activate = nn.Identity()
        elif activation_type == 'lrelu':
            self.activate = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            raise NotImplementedError(f'Not implemented activation function: '
                                      f'`{activation_type}`!')

    def forward(self, x):
        if x.ndim != 2:
            x = x.view(x.shape[0], -1)
        bias = self.bias * self.bscale if self.bias is not None else None
        x = F.linear(x, weight=self.weight * self.wscale, bias=bias)
        x = self.activate(x)
        return x


class StyleGANGeneratorForV6(nn.Module):
    """Defines the generator network in StyleGAN.

    NOTE: The synthesized images are with `RGB` channel order and pixel range
    [-1, 1].

    Settings for the mapping network:

    (1) z_space_dim: Dimension of the input latent space, Z.
        (default: 512 -> change to 128 is not compatible for pre-trained model)
    (2) w_space_dim: Dimension of the output latent space, W.
        (default: 512 -> change to 128 is not compatible for pre-trained model)
    (3) label_size: Size of the additional label for conditional generation.
        (default: 0)
    (4）mapping_layers: Number of layers of the mapping network. (default: 8)
    (5) mapping_fmaps: Number of hidden channels of the mapping network.
        (default: 512)
    (6) mapping_lr_mul: Learning rate multiplier for the mapping network.
        (default: 0.01)
    (7) repeat_w: Repeat w-code for different layers.

    Settings for the synthesis network:

    (1) resolution: The resolution of the output image.
    (2) image_channels: Number of channels of the output image. (default: 3)
    (3) final_tanh: Whether to use `tanh` to control the final pixel range.
        (default: False)
    (4) const_input: Whether to use a constant in the first convolutional layer.
        (default: True)
    (5) fused_scale: Whether to fused `upsample` and `conv2d` together,
        resulting in `conv2d_transpose`. (default: `auto`)
    (6) use_wscale: Whether to use weight scaling. (default: True)
    (7) noise_type: Type of noise added to the convolutional results at each
        layer. (default: `spatial`)
    (8) fmaps_base: Factor to control number of feature maps for each layer.
        (default: 16 << 10)
    (9) fmaps_max: Maximum number of feature maps in each layer. (default: 512)
    """

    def __init__(self,
                 resolution,
                 z_space_dim=512,       # 512 in Original code
                 w_space_dim=512,       # 512 in Original code
                 label_size=3,          # (eyes, mouth, pose) property
                 mapping_layers=8,
                 mapping_fmaps=512,
                 mapping_lr_mul=0.01,
                 repeat_w=True,
                 image_channels=3,
                 final_tanh=False,
                 const_input=True,
                 fused_scale='auto',
                 use_wscale=True,
                 noise_type='spatial',
                 fmaps_base=16 << 10,
                 fmaps_max=512):
        """Initializes with basic settings.

        Raises:
            ValueError: If the `resolution` is not supported, or `fused_scale`
                is not supported.
        """
        super().__init__()

        if resolution not in _RESOLUTIONS_ALLOWED:
            raise ValueError(f'Invalid resolution: `{resolution}`!\n'
                             f'Resolutions allowed: {_RESOLUTIONS_ALLOWED}.')
        if fused_scale not in _FUSED_SCALE_ALLOWED:
            raise ValueError(f'Invalid fused-scale option: `{fused_scale}`!\n'
                             f'Options allowed: {_FUSED_SCALE_ALLOWED}.')

        self.init_res = _INIT_RES
        self.resolution = resolution
        self.z_space_dim = z_space_dim
        self.w_space_dim = w_space_dim
        self.label_size = label_size
        self.mapping_layers = mapping_layers
        self.mapping_fmaps = mapping_fmaps
        self.mapping_lr_mul = mapping_lr_mul
        self.repeat_w = repeat_w
        self.image_channels = image_channels
        self.final_tanh = final_tanh
        self.const_input = const_input
        self.fused_scale = fused_scale
        self.use_wscale = use_wscale
        self.noise_type = noise_type
        self.fmaps_base = fmaps_base
        self.fmaps_max = fmaps_max

        self.num_layers = int(np.log2(self.resolution // self.init_res * 2)) * 2

        if self.repeat_w:
            self.mapping_space_dim = self.w_space_dim
        else:
            self.mapping_space_dim = self.w_space_dim * self.num_layers
        self.mapping = MappingModuleForV6(input_space_dim=self.z_space_dim,
                                          hidden_space_dim=self.mapping_fmaps,
                                          final_space_dim=self.mapping_space_dim,
                                          label_size=self.label_size,
                                          num_layers=self.mapping_layers,
                                          use_wscale=self.use_wscale,
                                          lr_mul=self.mapping_lr_mul)

        self.truncation = TruncationModuleForV6(w_space_dim=self.w_space_dim,
                                                num_layers=self.num_layers,
                                                repeat_w=self.repeat_w)

        self.synthesis = SynthesisModule(resolution=self.resolution,
                                         init_resolution=self.init_res,
                                         w_space_dim=self.w_space_dim,
                                         image_channels=self.image_channels,
                                         final_tanh=self.final_tanh,
                                         const_input=self.const_input,
                                         fused_scale=self.fused_scale,
                                         use_wscale=self.use_wscale,
                                         noise_type=self.noise_type,
                                         fmaps_base=self.fmaps_base,
                                         fmaps_max=self.fmaps_max)

        self.pth_to_tf_var_mapping = {}
        for key, val in self.mapping.pth_to_tf_var_mapping.items():
            self.pth_to_tf_var_mapping[f'mapping.{key}'] = val
        for key, val in self.truncation.pth_to_tf_var_mapping.items():
            self.pth_to_tf_var_mapping[f'truncation.{key}'] = val
        for key, val in self.synthesis.pth_to_tf_var_mapping.items():
            self.pth_to_tf_var_mapping[f'synthesis.{key}'] = val

    def set_space_of_latent(self, space_of_latent='w'):
        """Sets the space to which the latent code belong.

        This function is particually used for choosing how to inject the latent
        code into the convolutional layers. The original generator will take a
        W-Space code and apply it for style modulation after an affine
        transformation. But, sometimes, it may need to directly feed an already
        affine-transformed code into the convolutional layer, e.g., when
        training an encoder for GAN inversion. We term the transformed space as
        Style Space (or Y-Space). This function is designed to tell the
        convolutional layers how to use the input code.

        Args:
            space_of_latent: The space to which the latent code belong. Case
                insensitive. (default: 'w')
        """
        for module in self.modules():
            if isinstance(module, StyleModLayer):
                setattr(module, 'space_of_latent', space_of_latent)

    def forward(self,
                z,
                label,  # (eyes, mouth, pose) property score
                lod=None,
                w_moving_decay=0.995,
                style_mixing_prob=0.0,  # originally 0.9
                trunc_psi=None,
                trunc_layers=None,
                randomize_noise=False,
                **_unused_kwargs):
        mapping_results = self.mapping(z, label)
        w = mapping_results['w']

        if self.training and w_moving_decay < 1:
            batch_w_avg = all_gather(w).mean(dim=0)
            self.truncation.w_avg.copy_(
                self.truncation.w_avg * w_moving_decay +
                batch_w_avg * (1 - w_moving_decay))

        if self.training and style_mixing_prob > 0:
            new_z = torch.randn_like(z)
            new_w = self.mapping(new_z, label)['w']
            lod = self.synthesis.lod.cpu().tolist() if lod is None else lod
            current_layers = self.num_layers - int(lod) * 2
            if np.random.uniform() < style_mixing_prob:
                mixing_cutoff = np.random.randint(1, current_layers)
                w = self.truncation(w)
                new_w = self.truncation(new_w)
                w[:, mixing_cutoff:] = new_w[:, mixing_cutoff:]

        wp = self.truncation(w, trunc_psi, trunc_layers)
        synthesis_results = self.synthesis(wp, lod, randomize_noise)

        return {**mapping_results, **synthesis_results}


class MappingModuleForV6(nn.Module):
    """Implements the latent space mapping module.

    Basically, this module executes several dense layers in sequence.
    """

    def __init__(self,
                 input_space_dim=512,
                 hidden_space_dim=512,
                 final_space_dim=512,
                 label_convert_dim=16,
                 label_size=3,  # (eyes, mouth, pose) property score
                 num_layers=8,
                 normalize_input=True,
                 use_wscale=True,
                 lr_mul=0.01):
        super().__init__()

        self.input_space_dim = input_space_dim
        self.hidden_space_dim = hidden_space_dim
        self.final_space_dim = final_space_dim
        self.label_size = label_size
        self.num_layers = num_layers
        self.normalize_input = normalize_input
        self.use_wscale = use_wscale
        self.lr_mul = lr_mul

        self.norm = PixelNormLayer() if self.normalize_input else nn.Identity()

        self.pth_to_tf_var_mapping = {}
        for i in range(num_layers):
            dim_mul = 2 if label_size else 1
            in_channels = (input_space_dim * dim_mul if i == 0 else
                           hidden_space_dim)
            out_channels = (final_space_dim if i == (num_layers - 1) else
                            hidden_space_dim)
            self.add_module(f'dense{i}',
                            DenseBlock(in_channels=input_space_dim + label_convert_dim if i == 0 else in_channels,
                                       out_channels=out_channels,
                                       use_wscale=self.use_wscale,
                                       lr_mul=self.lr_mul))
            self.pth_to_tf_var_mapping[f'dense{i}.weight'] = f'Dense{i}/weight'
            self.pth_to_tf_var_mapping[f'dense{i}.bias'] = f'Dense{i}/bias'
        if label_size:
            self.label_weight = nn.Parameter(
                torch.randn(label_size, label_convert_dim))
            self.pth_to_tf_var_mapping[f'label_weight'] = f'LabelConcat/weight'

    def forward(self, z, label):
        if z.ndim != 2 or z.shape[1] != self.input_space_dim:
            raise ValueError(f'Input latent code should be with shape '
                             f'[batch_size, input_dim], where '
                             f'`input_dim` equals to {self.input_space_dim}!\n'
                             f'But `{z.shape}` is received!')

        if label.ndim != 2 or label.shape != (z.shape[0], self.label_size):
            raise ValueError(f'Input label should be with shape '
                             f'[batch_size, label_size], where '
                             f'`batch_size` equals to that of '
                             f'latent codes ({z.shape[0]}) and '
                             f'`label_size` equals to {self.label_size}!\n'
                             f'But `{label.shape}` is received!')

        embedding = torch.matmul(label, self.label_weight)
        z = torch.cat((z, embedding), dim=1)

        z = self.norm(z)
        w = z
        for i in range(self.num_layers):
            w = self.__getattr__(f'dense{i}')(w)
        results = {
            'z': z,
            'label': label,
            'w': w,
        }
        if self.label_size:
            results['embedding'] = embedding
        return results


class TruncationModuleForV6(nn.Module):
    """Implements the truncation module.

    Truncation is executed as follows:

    For layers in range [0, truncation_layers), the truncated w-code is computed
    as

    w_new = w_avg + (w - w_avg) * truncation_psi

    To disable truncation, please set
    (1) truncation_psi = 1.0 (None) OR
    (2) truncation_layers = 0 (None)

    NOTE: The returned tensor is layer-wise style codes.
    """

    def __init__(self, w_space_dim, num_layers, repeat_w=True):
        super().__init__()

        self.num_layers = num_layers
        self.w_space_dim = w_space_dim
        self.repeat_w = repeat_w

        if self.repeat_w:
            self.register_buffer('w_avg', torch.zeros(w_space_dim))
        else:
            self.register_buffer('w_avg', torch.zeros(num_layers * w_space_dim))
        self.pth_to_tf_var_mapping = {'w_avg': 'dlatent_avg'}

    def forward(self, w, trunc_psi=None, trunc_layers=None):
        if w.ndim == 2:
            if self.repeat_w and w.shape[1] == self.w_space_dim:
                w = w.view(-1, 1, self.w_space_dim)
                wp = w.repeat(1, self.num_layers, 1)
            else:
                assert w.shape[1] == self.w_space_dim * self.num_layers
                wp = w.view(-1, self.num_layers, self.w_space_dim)
        else:
            wp = w
        assert wp.ndim == 3
        assert wp.shape[1:] == (self.num_layers, self.w_space_dim)

        trunc_psi = 1.0 if trunc_psi is None else trunc_psi
        trunc_layers = 0 if trunc_layers is None else trunc_layers
        if trunc_psi < 1.0 and trunc_layers > 0:
            layer_idx = np.arange(self.num_layers).reshape(1, -1, 1)
            coefs = np.ones_like(layer_idx, dtype=np.float32)
            coefs[layer_idx < trunc_layers] *= trunc_psi
            coefs = torch.from_numpy(coefs).to(wp)
            w_avg = self.w_avg.view(1, -1, self.w_space_dim)
            wp = w_avg + (wp - w_avg) * coefs
        return wp