# -*- coding: utf-8 -*-

"""HiFi-GAN Modules.

This code is based on https://github.com/jik876/hifi-gan.

"""

import copy
import logging
import random

import numpy as np
import torch
import torch.nn.functional as F

from parallel_wavegan.layers import HiFiGANResidualBlock as ResidualBlock
from parallel_wavegan.utils import read_hdf5



class DiscreteSymbolSpkEmbHiFiGANGenerator(torch.nn.Module):
    """HiFiGAN generator module."""

    def __init__(
        self,
        in_channels=512,
        out_channels=1,
        channels=512,
        num_embs=100,
        spk_emb_dim=128,
        concat_spk_emb=False,
        kernel_size=7,
        upsample_scales=(8, 8, 2, 2),
        upsample_kernel_sizes=(16, 16, 4, 4),
        resblock_kernel_sizes=(3, 7, 11),
        resblock_dilations=[(1, 3, 5), (1, 3, 5), (1, 3, 5)],
        use_additional_convs=True,
        bias=True,
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.1},
        use_weight_norm=True,
    ):
        """Initialize HiFiGANGenerator module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            channels (int): Number of hidden representation channels.
            kernel_size (int): Kernel size of initial and final conv layer.
            upsample_scales (list): List of upsampling scales.
            upsample_kernel_sizes (list): List of kernal sizes for upsampling layers.
            resblock_kernal_sizes (list): List of kernal sizes for residual blocks.
            resblock_dilations (list): List of dilation list for residual blocks.
            use_additional_convs (bool): Whether to use additional conv layers in residual blocks.
            bias (bool): Whether to add bias parameter in convolution layers.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.
            use_weight_norm (bool): Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.

        """
        super().__init__()

        # define id embedding
        self.emb = torch.nn.Embedding(
            num_embeddings=num_embs, embedding_dim=in_channels
        )

        self.spk_emb_dim = spk_emb_dim
        self.concat_spk_emb = concat_spk_emb
        if not concat_spk_emb:
            assert in_channels == spk_emb_dim
        else:
            in_channels = in_channels + spk_emb_dim

        # check hyperparameters are valid
        assert kernel_size % 2 == 1, "Kernal size must be odd number."
        assert len(upsample_scales) == len(upsample_kernel_sizes)
        assert len(resblock_dilations) == len(resblock_kernel_sizes)

        # define modules
        self.num_upsamples = len(upsample_kernel_sizes)
        self.num_blocks = len(resblock_kernel_sizes)
        self.input_conv = torch.nn.Conv1d(
            in_channels,
            channels,
            kernel_size,
            1,
            padding=(kernel_size - 1) // 2,
        )
        self.upsamples = torch.nn.ModuleList()
        self.blocks = torch.nn.ModuleList()
        for i in range(len(upsample_kernel_sizes)):
            self.upsamples += [
                torch.nn.Sequential(
                    getattr(torch.nn, nonlinear_activation)(
                        **nonlinear_activation_params
                    ),
                    torch.nn.ConvTranspose1d(
                        channels // (2 ** i),
                        channels // (2 ** (i + 1)),
                        upsample_kernel_sizes[i],
                        upsample_scales[i],
                        padding=(upsample_kernel_sizes[i] - upsample_scales[i]) // 2,
                    ),
                )
            ]
            for j in range(len(resblock_kernel_sizes)):
                self.blocks += [
                    ResidualBlock(
                        kernel_size=resblock_kernel_sizes[j],
                        channels=channels // (2 ** (i + 1)),
                        dilations=resblock_dilations[j],
                        bias=bias,
                        use_additional_convs=use_additional_convs,
                        nonlinear_activation=nonlinear_activation,
                        nonlinear_activation_params=nonlinear_activation_params,
                    )
                ]
        self.output_conv = torch.nn.Sequential(
            # NOTE(kan-bayashi): follow official implementation but why
            #   using different slope parameter here? (0.1 vs. 0.01)
            torch.nn.LeakyReLU(),
            torch.nn.Conv1d(
                channels // (2 ** (i + 1)),
                out_channels,
                kernel_size,
                1,
                padding=(kernel_size - 1) // 2,
            ),
            torch.nn.Tanh(),
        )

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

        # reset parameters
        self.reset_parameters()

    def forward(self, c):
        """Calculate forward propagation.

        Args:
            c (Tensor): Input tensor (B, N, T).

        Returns:
            Tensor: Output tensor (B, out_channels, T).

        """
        # convert idx to embedding
        assert c.size(1) == self.spk_emb_dim + 1
        c_idx, g = c.split([1, self.spk_emb_dim], dim=1)
        c = self.emb(c_idx.long().squeeze(1)).transpose(1, 2)  # (B, C, T)
        g = g[:, :, 0]

        # integrate global embedding
        if not self.concat_spk_emb:
            c = c + g.unsqueeze(2)
        else:
            g = g.unsqueeze(1).expand(-1, c.size(1), -1)
            c = torch.cat([c, g], dim=-1)

        c = self.input_conv(c)
        for i in range(self.num_upsamples):
            c = self.upsamples[i](c)
            cs = 0.0  # initialize
            for j in range(self.num_blocks):
                cs += self.blocks[i * self.num_blocks + j](c)
            c = cs / self.num_blocks
        c = self.output_conv(c)

        return c

    def reset_parameters(self):
        """Reset parameters.

        This initialization follows the official implementation manner.
        https://github.com/jik876/hifi-gan/blob/master/models.py

        """

        def _reset_parameters(m):
            if isinstance(m, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
                m.weight.data.normal_(0.0, 0.01)
                logging.debug(f"Reset parameters in {m}.")

        self.apply(_reset_parameters)

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m):
            try:
                logging.debug(f"Weight norm is removed from {m}.")
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(
                m, torch.nn.ConvTranspose1d
            ):
                torch.nn.utils.weight_norm(m)
                logging.debug(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    def inference(self, c, g=None, normalize_before=False):
        """Perform inference.

        Args:
            c (Union[Tensor, ndarray]): Input tensor (T, 1).
            g (Union[Tensor, ndarray]): Input tensor (1, Dim).

        Returns:
            Tensor: Output tensor (T ** prod(upsample_scales), out_channels).

        """
        assert not normalize_before, "No statistics are used."
        if not isinstance(c, torch.Tensor):
            c = torch.tensor(c, dtype=torch.long).to(next(self.parameters()).device)
        if g is not None:
            spkemb = torch.tensor(np.repeat(g, c.shape[0], axis=0), device=c.device)
            c = torch.cat([c, spkemb], axis=1)

        c = self.forward(c.transpose(1, 0).unsqueeze(0))  # (1, Dim+1, T)
        return c.squeeze(0).transpose(1, 0)

    def inference_w_xvec(self, c, g, normalize_before=False):
        """Perform inference give xvector.

        Args:
            c (Union[Tensor, ndarray]): Input tensor (T, 1).
            g (Tensor): Xvector Tensor (1, dim, T)

        Returns:
            Tensor: Output tensor (T ** prod(upsample_scales), out_channels).

        """
        assert not normalize_before, "No statistics are used."
        assert g is not None

        if not isinstance(c, torch.Tensor):
            c = torch.tensor(c, dtype=torch.long).to(next(self.parameters()).device)

        c = self.emb(c.transpose(1, 0).long())  # (T, 1) -> (1, T) -> (1, T, D)
        c = c.transpose(1, 2)  # (T, 1) -> (1, T) -> (1, T, D) -> (1, D, T)
        # integrate global embedding
        if not self.concat_spk_emb:
            c = c + g
        else:
            c = torch.cat([c, g], dim=1)

        c = self.input_conv(c)
        for i in range(self.num_upsamples):
            c = self.upsamples[i](c)
            cs = 0.0  # initialize
            for j in range(self.num_blocks):
                cs += self.blocks[i * self.num_blocks + j](c)
            c = cs / self.num_blocks
        c = self.output_conv(c)

        return c.squeeze(0).transpose(1, 0)
