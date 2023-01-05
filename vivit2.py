from einops import rearrange, reduce, repeat
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np

import utils
from transformer import PatchEmbed, TransformerContainer, get_sine_cosine_pos_emb
from weight_init import (trunc_normal_,constant_init_, init_from_vit_pretrain_, 
	init_from_mae_pretrain_, init_from_kinetics_pretrain_)
import math
from functools import partial
from pytorchvideo.layers.utils import round_width, set_attributes
from pytorchvideo.layers import MultiScaleBlock, SpatioTemporalClsPositionalEncoding
from pytorchvideo.models.vision_transformers import MultiscaleVisionTransformers

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import CNNEncoder
from .transformer import FeatureTransformer
from .matching import (global_correlation_softmax, local_correlation_softmax, local_correlation_with_flow,
                       global_correlation_softmax_stereo, local_correlation_softmax_stereo,
                       correlation_softmax_depth)
from .attention import SelfAttnPropagation
from .geometry import flow_warp, compute_flow_with_depth_pose
from .reg_refine import BasicUpdateBlock
from .utils import normalize_img, feature_add_position, upsample_flow_with_mask


class UniMatch(nn.Module):
    def __init__(self,
                 num_scales=1,
                 feature_channels=128,
                 upsample_factor=8,
                 num_head=1,
                 ffn_dim_expansion=4,
                 num_transformer_layers=6,
                 reg_refine=False,  # optional local regression refinement
                 task='flow',
                 ):
        super(UniMatch, self).__init__()

        self.feature_channels = feature_channels
        self.num_scales = num_scales
        self.upsample_factor = upsample_factor
        self.reg_refine = reg_refine

        # CNN
        self.backbone = CNNEncoder(output_dim=feature_channels, num_output_scales=num_scales)

        # Transformer
        self.transformer = FeatureTransformer(num_layers=num_transformer_layers,
                                              d_model=feature_channels,
                                              nhead=num_head,
                                              ffn_dim_expansion=ffn_dim_expansion,
                                              )

        # propagation with self-attn
        self.feature_flow_attn = SelfAttnPropagation(in_channels=feature_channels)

        if not self.reg_refine or task == 'depth':
            # convex upsampling simiar to RAFT
            # concat feature0 and low res flow as input
            self.upsampler = nn.Sequential(nn.Conv2d(2 + feature_channels, 256, 3, 1, 1),
                                           nn.ReLU(inplace=True),
                                           nn.Conv2d(256, upsample_factor ** 2 * 9, 1, 1, 0))
            # thus far, all the learnable parameters are task-agnostic

        if reg_refine:
            # optional task-specific local regression refinement
            self.refine_proj = nn.Conv2d(128, 256, 1)
            self.refine = BasicUpdateBlock(corr_channels=(2 * 4 + 1) ** 2,
                                           downsample_factor=upsample_factor,
                                           flow_dim=2 if task == 'flow' else 1,
                                           bilinear_up=task == 'depth',
                                           )

    def extract_feature(self, img0, img1):
        concat = torch.cat((img0, img1), dim=0)  # [2B, C, H, W]
        features = self.backbone(concat)  # list of [2B, C, H, W], resolution from high to low

        # reverse: resolution from low to high
        features = features[::-1]

        feature0, feature1 = [], []

        for i in range(len(features)):
            feature = features[i]
            chunks = torch.chunk(feature, 2, 0)  # tuple
            feature0.append(chunks[0])
            feature1.append(chunks[1])

        return feature0, feature1

    def upsample_flow(self, flow, feature, bilinear=False, upsample_factor=8,
                      is_depth=False):
        if bilinear:
            multiplier = 1 if is_depth else upsample_factor
            up_flow = F.interpolate(flow, scale_factor=upsample_factor,
                                    mode='bilinear', align_corners=True) * multiplier
        else:
            concat = torch.cat((flow, feature), dim=1)
            mask = self.upsampler(concat)
            up_flow = upsample_flow_with_mask(flow, mask, upsample_factor=self.upsample_factor,
                                              is_depth=is_depth)

        return up_flow

    def forward(self, img0, img1,
                attn_type=None,
                attn_splits_list=None,
                corr_radius_list=None,
                prop_radius_list=None,
                num_reg_refine=1,
                pred_bidir_flow=False,
                task='flow',
                intrinsics=None,
                pose=None,  # relative pose transform
                min_depth=1. / 0.5,  # inverse depth range
                max_depth=1. / 10,
                num_depth_candidates=64,
                depth_from_argmax=False,
                pred_bidir_depth=False,
                **kwargs,
                ):

        if pred_bidir_flow:
            assert task == 'flow'

        if task == 'depth':
            assert self.num_scales == 1  # multi-scale depth model is not supported yet

        results_dict = {}
        flow_preds = []

        if task == 'flow':
            # stereo and depth tasks have normalized img in dataloader
            img0, img1 = normalize_img(img0, img1)  # [B, 3, H, W]

        # list of features, resolution low to high
        feature0_list, feature1_list = self.extract_feature(img0, img1)  # list of features

        flow = None

        if task != 'depth':
            assert len(attn_splits_list) == len(corr_radius_list) == len(prop_radius_list) == self.num_scales
        else:
            assert len(attn_splits_list) == len(prop_radius_list) == self.num_scales == 1

        for scale_idx in range(self.num_scales):
            feature0, feature1 = feature0_list[scale_idx], feature1_list[scale_idx]

            if pred_bidir_flow and scale_idx > 0:
                # predicting bidirectional flow with refinement
                feature0, feature1 = torch.cat((feature0, feature1), dim=0), torch.cat((feature1, feature0), dim=0)

            feature0_ori, feature1_ori = feature0, feature1

            upsample_factor = self.upsample_factor * (2 ** (self.num_scales - 1 - scale_idx))

            if task == 'depth':
                # scale intrinsics
                intrinsics_curr = intrinsics.clone()
                intrinsics_curr[:, :2] = intrinsics_curr[:, :2] / upsample_factor

            if scale_idx > 0:
                assert task != 'depth'  # not supported for multi-scale depth model
                flow = F.interpolate(flow, scale_factor=2, mode='bilinear', align_corners=True) * 2

            if flow is not None:
                assert task != 'depth'
                flow = flow.detach()

                if task == 'stereo':
                    # construct flow vector for disparity
                    # flow here is actually disparity
                    zeros = torch.zeros_like(flow)  # [B, 1, H, W]
                    # NOTE: reverse disp, disparity is positive
                    displace = torch.cat((-flow, zeros), dim=1)  # [B, 2, H, W]
                    feature1 = flow_warp(feature1, displace)  # [B, C, H, W]
                elif task == 'flow':
                    feature1 = flow_warp(feature1, flow)  # [B, C, H, W]
                else:
                    raise NotImplementedError

            attn_splits = attn_splits_list[scale_idx]
            if task != 'depth':
                corr_radius = corr_radius_list[scale_idx]
            prop_radius = prop_radius_list[scale_idx]

            # add position to features
            feature0, feature1 = feature_add_position(feature0, feature1, attn_splits, self.feature_channels)

            # Transformer
            feature0, feature1 = self.transformer(feature0, feature1,
                                                  attn_type=attn_type,
                                                  attn_num_splits=attn_splits,
                                                  )

            # correlation and softmax
            if task == 'depth':
                # first generate depth candidates
                b, _, h, w = feature0.size()
                depth_candidates = torch.linspace(min_depth, max_depth, num_depth_candidates).type_as(feature0)
                depth_candidates = depth_candidates.view(1, num_depth_candidates, 1, 1).repeat(b, 1, h,
                                                                                               w)  # [B, D, H, W]

                flow_pred = correlation_softmax_depth(feature0, feature1,
                                                      intrinsics_curr,
                                                      pose,
                                                      depth_candidates=depth_candidates,
                                                      depth_from_argmax=depth_from_argmax,
                                                      pred_bidir_depth=pred_bidir_depth,
                                                      )[0]

            else:
                if corr_radius == -1:  # global matching
                    if task == 'flow':
                        flow_pred = global_correlation_softmax(feature0, feature1, pred_bidir_flow)[0]
                    elif task == 'stereo':
                        flow_pred = global_correlation_softmax_stereo(feature0, feature1)[0]
                    else:
                        raise NotImplementedError
                else:  # local matching
                    if task == 'flow':
                        flow_pred = local_correlation_softmax(feature0, feature1, corr_radius)[0]
                    elif task == 'stereo':
                        flow_pred = local_correlation_softmax_stereo(feature0, feature1, corr_radius)[0]
                    else:
                        raise NotImplementedError

            # flow or residual flow
            flow = flow + flow_pred if flow is not None else flow_pred

            if task == 'stereo':
                flow = flow.clamp(min=0)  # positive disparity

            # upsample to the original resolution for supervison at training time only
            if self.training:
                flow_bilinear = self.upsample_flow(flow, None, bilinear=True, upsample_factor=upsample_factor,
                                                   is_depth=task == 'depth')
                flow_preds.append(flow_bilinear)

            # flow propagation with self-attn
            if (pred_bidir_flow or pred_bidir_depth) and scale_idx == 0:
                feature0 = torch.cat((feature0, feature1), dim=0)  # [2*B, C, H, W] for propagation

            flow = self.feature_flow_attn(feature0, flow.detach(),
                                          local_window_attn=prop_radius > 0,
                                          local_window_radius=prop_radius,
                                          )

            # bilinear exclude the last one
            if self.training and scale_idx < self.num_scales - 1:
                flow_up = self.upsample_flow(flow, feature0, bilinear=True,
                                             upsample_factor=upsample_factor,
                                             is_depth=task == 'depth')
                flow_preds.append(flow_up)

            if scale_idx == self.num_scales - 1:
                if not self.reg_refine:
                    # upsample to the original image resolution

                    if task == 'stereo':
                        flow_pad = torch.cat((-flow, torch.zeros_like(flow)), dim=1)  # [B, 2, H, W]
                        flow_up_pad = self.upsample_flow(flow_pad, feature0)
                        flow_up = -flow_up_pad[:, :1]  # [B, 1, H, W]
                    elif task == 'depth':
                        depth_pad = torch.cat((flow, torch.zeros_like(flow)), dim=1)  # [B, 2, H, W]
                        depth_up_pad = self.upsample_flow(depth_pad, feature0,
                                                          is_depth=True).clamp(min=min_depth, max=max_depth)
                        flow_up = depth_up_pad[:, :1]  # [B, 1, H, W]
                    else:
                        flow_up = self.upsample_flow(flow, feature0)

                    flow_preds.append(flow_up)
                else:
                    # task-specific local regression refinement
                    # supervise current flow
                    if self.training:
                        flow_up = self.upsample_flow(flow, feature0, bilinear=True,
                                                     upsample_factor=upsample_factor,
                                                     is_depth=task == 'depth')
                        flow_preds.append(flow_up)

                    assert num_reg_refine > 0
                    for refine_iter_idx in range(num_reg_refine):
                        flow = flow.detach()

                        if task == 'stereo':
                            zeros = torch.zeros_like(flow)  # [B, 1, H, W]
                            # NOTE: reverse disp, disparity is positive
                            displace = torch.cat((-flow, zeros), dim=1)  # [B, 2, H, W]
                            correlation = local_correlation_with_flow(
                                feature0_ori,
                                feature1_ori,
                                flow=displace,
                                local_radius=4,
                            )  # [B, (2R+1)^2, H, W]
                        elif task == 'depth':
                            if pred_bidir_depth and refine_iter_idx == 0:
                                intrinsics_curr = intrinsics_curr.repeat(2, 1, 1)
                                pose = torch.cat((pose, torch.inverse(pose)), dim=0)

                                feature0_ori, feature1_ori = torch.cat((feature0_ori, feature1_ori),
                                                                       dim=0), torch.cat((feature1_ori,
                                                                                          feature0_ori), dim=0)

                            flow_from_depth = compute_flow_with_depth_pose(1. / flow.squeeze(1),
                                                                           intrinsics_curr,
                                                                           extrinsics_rel=pose,
                                                                           )

                            correlation = local_correlation_with_flow(
                                feature0_ori,
                                feature1_ori,
                                flow=flow_from_depth,
                                local_radius=4,
                            )  # [B, (2R+1)^2, H, W]

                        else:
                            correlation = local_correlation_with_flow(
                                feature0_ori,
                                feature1_ori,
                                flow=flow,
                                local_radius=4,
                            )  # [B, (2R+1)^2, H, W]

                        proj = self.refine_proj(feature0)

                        net, inp = torch.chunk(proj, chunks=2, dim=1)

                        net = torch.tanh(net)
                        inp = torch.relu(inp)

                        net, up_mask, residual_flow = self.refine(net, inp, correlation, flow.clone(),
                                                                  )

                        if task == 'depth':
                            flow = (flow - residual_flow).clamp(min=min_depth, max=max_depth)
                        else:
                            flow = flow + residual_flow

                        if task == 'stereo':
                            flow = flow.clamp(min=0)  # positive

                        if self.training or refine_iter_idx == num_reg_refine - 1:
                            if task == 'depth':
                                if refine_iter_idx < num_reg_refine - 1:
                                    # bilinear upsampling
                                    flow_up = self.upsample_flow(flow, feature0, bilinear=True,
                                                                 upsample_factor=upsample_factor,
                                                                 is_depth=True)
                                else:
                                    # last one convex upsampling
                                    # NOTE: clamp depth due to the zero padding in the unfold in the convex upsampling
                                    # pad depth to 2 channels as flow
                                    depth_pad = torch.cat((flow, torch.zeros_like(flow)), dim=1)  # [B, 2, H, W]
                                    depth_up_pad = self.upsample_flow(depth_pad, feature0,
                                                                      is_depth=True).clamp(min=min_depth,
                                                                                           max=max_depth)
                                    flow_up = depth_up_pad[:, :1]  # [B, 1, H, W]

                            else:
                                flow_up = upsample_flow_with_mask(flow, up_mask, upsample_factor=self.upsample_factor,
                                                                  is_depth=task == 'depth')

                            flow_preds.append(flow_up)

        if task == 'stereo':
            for i in range(len(flow_preds)):
                flow_preds[i] = flow_preds[i].squeeze(1)  # [B, H, W]

        # convert inverse depth to depth
        if task == 'depth':
            for i in range(len(flow_preds)):
                flow_preds[i] = 1. / flow_preds[i].squeeze(1)  # [B, H, W]

        results_dict.update({'flow_preds': flow_preds})

        return results_dict


class ClassificationHead(nn.Module):
	"""Classification head for Video Transformer.
	
	Args:
		num_classes (int): Number of classes to be classified.
		in_channels (int): Number of channels in input feature.
		init_std (float): Std value for Initiation. Defaults to 0.02.
		kwargs (dict, optional): Any keyword argument to be used to initialize
			the head.
	"""

	def __init__(self,
				 num_classes,
				 in_channels,		
				 init_std=0.02,
				 eval_metrics='finetune',
				 **kwargs):
		super().__init__()
		self.init_std = init_std
		self.eval_metrics = eval_metrics
		self.cls_head = nn.Linear(in_channels, num_classes)
		
		self.init_weights(self.cls_head)

	def init_weights(self, module):
		if hasattr(module, 'weight') and module.weight is not None:
			if self.eval_metrics == 'finetune':
				trunc_normal_(module.weight, std=self.init_std)
			else:
				module.weight.data.normal_(mean=0.0, std=0.01)
		if hasattr(module, 'bias') and module.bias is not None:
			constant_init_(module.bias, constant_value=0)

	def forward(self, x):
		cls_score = self.cls_head(x)
		return cls_score

class ViViT(nn.Module):
	"""ViViT. A PyTorch impl of `ViViT: A Video Vision Transformer`
		<https://arxiv.org/abs/2103.15691>

	Args:
		num_frames (int): Number of frames in the video.
		img_size (int | tuple): Size of input image.
		patch_size (int): Size of one patch.
		pretrained (str | None): Name of pretrained model. Default: None.
		embed_dims (int): Dimensions of embedding. Defaults to 768.
		num_heads (int): Number of parallel attention heads. Defaults to 12.
		num_transformer_layers (int): Number of transformer layers. Defaults to 12.
		in_channels (int): Channel num of input features. Defaults to 3.
		dropout_p (float): Probability of dropout layer. Defaults to 0..
		tube_size (int): Dimension of the kernel size in Conv3d. Defaults to 2.
		conv_type (str): Type of the convolution in PatchEmbed layer. Defaults to Conv3d.
		attention_type (str): Type of attentions in TransformerCoder. Choices
			are 'divided_space_time', 'fact_encoder' and 'joint_space_time'.
			Defaults to 'fact_encoder'.
		norm_layer (dict): Config for norm layers. Defaults to nn.LayerNorm.
		copy_strategy (str): Copy or Initial to zero towards the new additional layer.
		extend_strategy (str): How to initialize the weights of Conv3d from pre-trained Conv2d.
		use_learnable_pos_emb (bool): Whether to use learnable position embeddings.
		return_cls_token (bool): Whether to use cls_token to predict class label.
	"""
	supported_attention_types = [
		'fact_encoder', 'joint_space_time', 'divided_space_time'
	]

	def __init__(self,
				 num_frames,
				 img_size=224,
				 patch_size=16,
				 pretrain_pth=None,
				 weights_from='imagenet',
				 embed_dims=768,
				 num_heads=12,
				 num_transformer_layers=12,
				 in_channels=3,
				 dropout_p=0.,
				 tube_size=2,
				 conv_type='Conv3d',
				 attention_type='fact_encoder',
				 norm_layer=nn.LayerNorm,
				 copy_strategy='repeat',
				 extend_strategy='temporal_avg',
				 use_learnable_pos_emb=True,
				 return_cls_token=True,
				 **kwargs):
		super().__init__()
		assert attention_type in self.supported_attention_types, (
			f'Unsupported Attention Type {attention_type}!')

		num_frames = num_frames//tube_size
		self.num_frames = num_frames
		self.pretrain_pth = pretrain_pth
		self.weights_from = weights_from
		self.embed_dims = embed_dims
		self.num_transformer_layers = num_transformer_layers
		self.attention_type = attention_type
		self.conv_type = conv_type
		self.copy_strategy = copy_strategy
		self.extend_strategy = extend_strategy
		self.tube_size = tube_size
		self.num_time_transformer_layers = 0
		self.use_learnable_pos_emb = use_learnable_pos_emb
		self.return_cls_token = return_cls_token
		gmgflow = UniMatch(feature_channels=128,
                     num_scales=1,
                     upsample_factor=8,
                     num_head=num_heads,
                     ffn_dim_expansion=4,
                     num_transformer_layers=6,
                     reg_refine=False,
                     task='flow').to(device)


		#tokenize & position embedding
		self.patch_embed = PatchEmbed(
			img_size=img_size,
			patch_size=patch_size,
			in_channels=in_channels,
			embed_dims=embed_dims,
			tube_size=tube_size,
			conv_type=conv_type)
		num_patches = self.patch_embed.num_patches
		
		if self.attention_type == 'divided_space_time':
			# Divided Space Time Attention - Model 3
			operator_order = ['time_attn','space_attn','ffn']
			container = TransformerContainer(
				num_transformer_layers=num_transformer_layers,
				embed_dims=embed_dims,
				num_heads=num_heads,
				num_frames=num_frames,
				norm_layer=norm_layer,
				hidden_channels=embed_dims*4,
				operator_order=operator_order)

			transformer_layers = container
		elif self.attention_type == 'joint_space_time':
			# Joint Space Time Attention - Model 1
			operator_order = ['self_attn','ffn']
			container = TransformerContainer(
				num_transformer_layers=num_transformer_layers,
				embed_dims=embed_dims,
				num_heads=num_heads,
				num_frames=num_frames,
				norm_layer=norm_layer,
				hidden_channels=embed_dims*4,
				operator_order=operator_order)
			
			transformer_layers = container
		else:
			# Divided Space Time Transformer Encoder - Model 2
			transformer_layers = nn.ModuleList([])
			self.num_time_transformer_layers = 4
			
			spatial_transformer = TransformerContainer(
				num_transformer_layers=num_transformer_layers,
				embed_dims=embed_dims,
				num_heads=num_heads,
				num_frames=num_frames,
				norm_layer=norm_layer,
				hidden_channels=embed_dims*4,
				operator_order=['self_attn','ffn'],
    			name="spatial")
			
			temporal_transformer = TransformerContainer(
				num_transformer_layers=self.num_time_transformer_layers,
				embed_dims=embed_dims,
				num_heads=num_heads,
				num_frames=num_frames,
				norm_layer=norm_layer,
				hidden_channels=embed_dims*4,
				operator_order=['self_attn','ffn'],
    			name="temporal")

			transformer_layers.append(spatial_transformer)
			transformer_layers.append(temporal_transformer)
 
		self.transformer_layers = transformer_layers
		self.norm = norm_layer(embed_dims, eps=1e-6)
		
		self.cls_token = nn.Parameter(torch.zeros(1,1,embed_dims))
		# whether to add one cls_token in temporal pos_enb
		if attention_type == 'fact_encoder':
			num_frames = num_frames + 1
			num_patches = num_patches + 1
			self.use_cls_token_temporal = False
		else:
			self.use_cls_token_temporal = operator_order[-2] == 'time_attn'
			if self.use_cls_token_temporal:
				num_frames = num_frames + 1
			else:
				num_patches = num_patches + 1

		if use_learnable_pos_emb:
			self.pos_embed = nn.Parameter(torch.zeros(1,num_patches,embed_dims))
			self.time_embed = nn.Parameter(torch.zeros(1,num_frames,embed_dims))
		else:
			self.pos_embed = get_sine_cosine_pos_emb(num_patches,embed_dims)
			self.time_embed = get_sine_cosine_pos_emb(num_frames,embed_dims)
		self.drop_after_pos = nn.Dropout(p=dropout_p)
		self.drop_after_time = nn.Dropout(p=dropout_p)

		self.init_weights()

	def init_weights(self):
		if self.use_learnable_pos_emb:
			#trunc_normal_(self.pos_embed, std=.02)
			#trunc_normal_(self.time_embed, std=.02)
			nn.init.trunc_normal_(self.pos_embed, std=.02)
			nn.init.trunc_normal_(self.time_embed, std=.02)
		trunc_normal_(self.cls_token, std=.02)
		
		if self.pretrain_pth is not None:
			if self.weights_from == 'imagenet':
				init_from_vit_pretrain_(self,
										self.pretrain_pth,
										self.conv_type,
										self.attention_type,
										self.copy_strategy,
										self.extend_strategy, 
										self.tube_size, 
										self.num_time_transformer_layers)
			elif self.weights_from == 'kinetics':
				init_from_kinetics_pretrain_(self,
											 self.pretrain_pth)
			else:
				raise TypeError(f'not support the pretrained weight {self.pretrain_pth}')

	@torch.jit.ignore
	def no_weight_decay_keywords(self):
		return {'pos_embed', 'cls_token', 'mask_token'}

	def prepare_tokens(self, x):
		#Tokenize
		b = x.shape[0]

		x = self.patch_embed(x)
		
		
		# Add Position Embedding
		cls_tokens = repeat(self.cls_token, 'b ... -> (repeat b) ...', repeat=x.shape[0])
		if self.use_cls_token_temporal:
			if self.use_learnable_pos_emb:
				x = x + self.pos_embed
			else:
				x = x + self.pos_embed.type_as(x).detach()
			x = torch.cat((cls_tokens, x), dim=1)
		else:
			x = torch.cat((cls_tokens, x), dim=1)
			if self.use_learnable_pos_emb:
				x = x + self.pos_embed
			else:
				x = x + self.pos_embed.type_as(x).detach()
		x = self.drop_after_pos(x)

		# Add Time Embedding
		if self.attention_type != 'fact_encoder':
			cls_tokens = x[:b, 0, :].unsqueeze(1)
			if self.use_cls_token_temporal:
				x = rearrange(x[:, 1:, :], '(b t) p d -> (b p) t d', b=b)
				cls_tokens = repeat(cls_tokens,
									'b ... -> (repeat b) ...',
									repeat=x.shape[0]//b)
				x = torch.cat((cls_tokens, x), dim=1)
				if self.use_learnable_pos_emb:
					x = x + self.time_embed
				else:
					x = x + self.time_embed.type_as(x).detach()
				cls_tokens = x[:b, 0, :].unsqueeze(1)
				x = rearrange(x[:, 1:, :], '(b p) t d -> b (p t) d', b=b)
				x = torch.cat((cls_tokens, x), dim=1)
			else:
				x = rearrange(x[:, 1:, :], '(b t) p d -> (b p) t d', b=b)
				if self.use_learnable_pos_emb:
					x = x + self.time_embed
				else:
					x = x + self.time_embed.type_as(x).detach()
				x = rearrange(x, '(b p) t d -> b (p t) d', b=b)
				x = torch.cat((cls_tokens, x), dim=1)
			x = self.drop_after_time(x)
		
		return x, cls_tokens, b

	def forward(self, x):
		
		x, cls_tokens, b = self.prepare_tokens(x)
		
		
		if self.attention_type != 'fact_encoder':
			x = self.transformer_layers(x)
		else:
			# fact encoder - CRNN style
			spatial_transformer, temporal_transformer, = *self.transformer_layers,
			x = spatial_transformer(x)
			
			# Add Time Embedding
			cls_tokens = x[:b, 0, :].unsqueeze(1)
			x = rearrange(x[:, 1:, :], '(b t) p d -> b t p d', b=b)
			x = reduce(x, 'b t p d -> b t d', 'mean')
			x = torch.cat((cls_tokens, x), dim=1)
			if self.use_learnable_pos_emb:
				x = x + self.time_embed
			else:
				x = x + self.time_embed.type_as(x).detach()
			x = self.drop_after_time(x)
			x = temporal_transformer(x)

		x = self.norm(x)
		# Return Class Token
		if self.return_cls_token:
			return x[:, 0]
		else:
			return x[:, 1:].mean(1)

	def get_last_selfattention(self, x):
		x, cls_tokens, b = self.prepare_tokens(x)
		
		if self.attention_type != 'fact_encoder':
			x = self.transformer_layers(x, return_attention=True)
		else:
			# fact encoder - CRNN style
			spatial_transformer, temporal_transformer, = *self.transformer_layers,
			x = spatial_transformer(x)
			
			# Add Time Embedding
			cls_tokens = x[:b, 0, :].unsqueeze(1)
			x = rearrange(x[:, 1:, :], '(b t) p d -> b t p d', b=b)
			x = reduce(x, 'b t p d -> b t d', 'mean')
			x = torch.cat((cls_tokens, x), dim=1)
			if self.use_learnable_pos_emb:
				x = x + self.time_embed
			else:
				x = x + self.time_embed.type_as(x).detach()
			x = self.drop_after_time(x)
			
			x = temporal_transformer(x, return_attention=True)
		return x


# --------------------------------------------------------
# Written by pytorchvideo offical repo (https://github.com/facebookresearch/pytorchvideo)
# Modified by MX
# --------------------------------------------------------
class PatchEmbeding(nn.Module):
	"""
	Transformer basic patch embedding module. Performs patchifying input, flatten and
	and transpose.
	The builder can be found in `create_patch_embed`.
	"""

	def __init__(
		self,
		*,
		patch_model=None,
	):
		super().__init__()
		set_attributes(self, locals())
		assert self.patch_model is not None

	def forward(self, x):
		x = self.patch_model(x)
		# B C (T) H W -> B (T)HW C
		return x.flatten(2).transpose(1, 2)


def create_conv_patch_embed(
	*,
	in_channels,
	out_channels,
	conv_kernel_size=(1, 16, 16),
	conv_stride=(1, 4, 4),
	conv_padding=(1, 7, 7),
	conv_bias=True,
	conv=nn.Conv3d,
):
	"""
	Creates the transformer basic patch embedding. It performs Convolution, flatten and
	transpose.
	Args:
		in_channels (int): input channel size of the convolution.
		out_channels (int): output channel size of the convolution.
		conv_kernel_size (tuple): convolutional kernel size(s).
		conv_stride (tuple): convolutional stride size(s).
		conv_padding (tuple): convolutional padding size(s).
		conv_bias (bool): convolutional bias. If true, adds a learnable bias to the
			output.
		conv (callable): Callable used to build the convolution layer.
	Returns:
		(nn.Module): transformer patch embedding layer.
	"""
	conv_module = conv(
		in_channels=in_channels,
		out_channels=out_channels,
		kernel_size=conv_kernel_size,
		stride=conv_stride,
		padding=conv_padding,
		bias=conv_bias,
	)
	return PatchEmbeding(patch_model=conv_module)


def create_multiscale_vision_transformers(
	*,
	spatial_size,
	temporal_size,
	cls_embed_on=True,
	sep_pos_embed=True,
	depth=16,
	norm="layernorm",
	# Patch embed config.
	input_channels=3,
	patch_embed_dim=96,
	conv_patch_embed_kernel=(3, 7, 7),
	conv_patch_embed_stride=(2, 4, 4),
	conv_patch_embed_padding=(1, 3, 3),
	enable_patch_embed_norm=False,
	use_2d_patch=False,
	# Attention block config.
	num_heads=1,
	mlp_ratio=4.0,
	qkv_bias=True,
	dropout_rate_block=0.0,
	droppath_rate_block=0.0,
	pooling_mode="conv",
	pool_first=False,
	residual_pool=False,
	depthwise_conv=True,
	bias_on=True,
	separate_qkv=True,
	embed_dim_mul=None,
	atten_head_mul=None,
	pool_q_stride_size=None,
	pool_kv_stride_size=None,
	pool_kv_stride_adaptive=None,
	pool_kvq_kernel=None,
	head=None,
) -> nn.Module:
	"""
	Build Multiscale Vision Transformers (MViT) for recognition. A Vision Transformer
	(ViT) is a specific case of MViT that only uses a single scale attention block.
	"""

	if use_2d_patch:
		assert temporal_size == 1, "If use_2d_patch, temporal_size needs to be 1."
	if pool_kv_stride_adaptive is not None:
		assert (
			pool_kv_stride_size is None
		), "pool_kv_stride_size should be none if pool_kv_stride_adaptive is set."
	if norm == "layernorm":
		norm_layer = partial(nn.LayerNorm, eps=1e-6)
		block_norm_layer = partial(nn.LayerNorm, eps=1e-6)
		attn_norm_layer = partial(nn.LayerNorm, eps=1e-6)
	else:
		raise NotImplementedError("Only supports layernorm.")

	if isinstance(spatial_size, int):
		spatial_size = (spatial_size, spatial_size)

	conv_patch_op = nn.Conv2d if use_2d_patch else nn.Conv3d
	norm_patch_embed = norm_layer(patch_embed_dim) if enable_patch_embed_norm else None

	patch_embed = None
	input_dims = [temporal_size, spatial_size[0], spatial_size[1]]
	input_stirde = (
		(1,) + tuple(conv_patch_embed_stride)
		if use_2d_patch
		else conv_patch_embed_stride
	)

	patch_embed_shape = (
		[input_dims[i] // input_stirde[i] for i in range(len(input_dims))]
	)

	cls_positional_encoding = SpatioTemporalClsPositionalEncoding(
		embed_dim=patch_embed_dim,
		patch_embed_shape=patch_embed_shape,
		sep_pos_embed=sep_pos_embed,
		has_cls=cls_embed_on,
	)

	dpr = [
		x.item() for x in torch.linspace(0, droppath_rate_block, depth)
	]  # stochastic depth decay rule

	if dropout_rate_block > 0.0:
		pos_drop = nn.Dropout(p=dropout_rate_block)

	dim_mul, head_mul = torch.ones(depth + 1), torch.ones(depth + 1)
	if embed_dim_mul is not None:
		for i in range(len(embed_dim_mul)):
			dim_mul[embed_dim_mul[i][0]] = embed_dim_mul[i][1]
	if atten_head_mul is not None:
		for i in range(len(atten_head_mul)):
			head_mul[atten_head_mul[i][0]] = atten_head_mul[i][1]

	mvit_blocks = nn.ModuleList()

	pool_q = [[] for i in range(depth)]
	pool_kv = [[] for i in range(depth)]
	stride_q = [[] for i in range(depth)]
	stride_kv = [[] for i in range(depth)]

	if pool_q_stride_size is not None:
		for i in range(len(pool_q_stride_size)):
			stride_q[pool_q_stride_size[i][0]] = pool_q_stride_size[i][1:]
			if pool_kvq_kernel is not None:
				pool_q[pool_q_stride_size[i][0]] = pool_kvq_kernel
			else:
				pool_q[pool_q_stride_size[i][0]] = [
					s + 1 if s > 1 else s for s in pool_q_stride_size[i][1:]
				]

	# If POOL_KV_STRIDE_ADAPTIVE is not None, initialize POOL_KV_STRIDE.
	if pool_kv_stride_adaptive is not None:
		_stride_kv = pool_kv_stride_adaptive
		pool_kv_stride_size = []
		for i in range(depth):
			if len(stride_q[i]) > 0:
				_stride_kv = [
					max(_stride_kv[d] // stride_q[i][d], 1)
					for d in range(len(_stride_kv))
				]
			pool_kv_stride_size.append([i] + _stride_kv)

	if pool_kv_stride_size is not None:
		for i in range(len(pool_kv_stride_size)):
			stride_kv[pool_kv_stride_size[i][0]] = pool_kv_stride_size[i][1:]
			if pool_kvq_kernel is not None:
				pool_kv[pool_kv_stride_size[i][0]] = pool_kvq_kernel
			else:
				pool_kv[pool_kv_stride_size[i][0]] = [
					s + 1 if s > 1 else s for s in pool_kv_stride_size[i][1:]
				]

	for i in range(depth):
		num_heads = round_width(num_heads, head_mul[i], min_width=1, divisor=1)
		patch_embed_dim = round_width(patch_embed_dim, dim_mul[i], divisor=num_heads)
		dim_out = round_width(
			patch_embed_dim,
			dim_mul[i + 1],
			divisor=round_width(num_heads, head_mul[i + 1]),
		)

		mvit_blocks.append(
			MultiScaleBlock(
				dim=patch_embed_dim,
				dim_out=dim_out,
				num_heads=num_heads,
				mlp_ratio=mlp_ratio,
				qkv_bias=qkv_bias,
				dropout_rate=dropout_rate_block,
				droppath_rate=dpr[i],
				norm_layer=block_norm_layer,
				#attn_norm_layer=attn_norm_layer,
				kernel_q=pool_q[i],
				kernel_kv=pool_kv[i],
				stride_q=stride_q[i],
				stride_kv=stride_kv[i],
				pool_mode=pooling_mode,
				has_cls_embed=cls_embed_on,
				pool_first=pool_first,
				#residual_pool=residual_pool,
				#bias_on=bias_on,
				#depthwise_conv=depthwise_conv,
				#separate_qkv=separate_qkv,
			)
		)

	embed_dim = dim_out
	norm_embed = None if norm_layer is None else norm_layer(embed_dim)
	head_model = None

	return MultiscaleVisionTransformers(
		patch_embed=patch_embed,
		cls_positional_encoding=cls_positional_encoding,
		pos_drop=pos_drop if dropout_rate_block > 0.0 else None,
		norm_patch_embed=norm_patch_embed,
		blocks=mvit_blocks,
		norm_embed=norm_embed,
		head=head_model,
	)


class MaskFeat(nn.Module):
	"""
	Multiscale Vision Transformers
	Haoqi Fan, Bo Xiong, Karttikeya Mangalam, Yanghao Li, Zhicheng Yan, Jitendra Malik, Christoph Feichtenhofer
	https://arxiv.org/abs/2104.11227
	"""

	def __init__(self,
				 img_size=224,
				 num_frames=16,
				 input_channels=3,
				 feature_dim=10,
				 patch_embed_dim=96,
				 conv_patch_embed_kernel=(3, 7, 7),
				 conv_patch_embed_stride=(2, 4, 4),
				 conv_patch_embed_padding=(1, 3, 3),
				 embed_dim_mul=[[1, 2.0], [3, 2.0], [14, 2.0]],
				 atten_head_mul=[[1, 2.0], [3, 2.0], [14, 2.0]],
				 pool_q_stride_size=[[1, 1, 2, 2], [3, 1, 2, 2], [14, 1, 2, 2]],
				 pool_kv_stride_adaptive=[1, 8, 8],
				 pool_kvq_kernel=[3, 3, 3],
				 head=None,
				 pretrain_pth=None,
				 **kwargs):
		super().__init__()
		self.num_frames = num_frames
		self.img_size = img_size
		self.stride = conv_patch_embed_stride
		self.downsample_rate = 2 ** len(pool_q_stride_size)
		self.embed_dims = 2**len(embed_dim_mul) * patch_embed_dim
		# Get mvit from pytorchvideo
		self.patch_embed = (
			create_conv_patch_embed(
				in_channels=input_channels,
				out_channels=patch_embed_dim,
				conv_kernel_size=conv_patch_embed_kernel,
				conv_stride=conv_patch_embed_stride,
				conv_padding=conv_patch_embed_padding,
				conv=nn.Conv3d,
			)
		)
		self.mvit = create_multiscale_vision_transformers(
			spatial_size=img_size, 
			temporal_size=num_frames,
			embed_dim_mul=embed_dim_mul,
			atten_head_mul=atten_head_mul,
			pool_q_stride_size=pool_q_stride_size,
			pool_kv_stride_adaptive=pool_kv_stride_adaptive,
			pool_kvq_kernel=pool_kvq_kernel,
			head=head)
		in_features = self.mvit.norm_embed.normalized_shape[0]
		out_features = feature_dim # the dimension of the predict features
		self.decoder_pred = nn.Linear(in_features, feature_dim, bias=True)
		# mask token
		self.mask_token = nn.Parameter(torch.zeros(1, 1, patch_embed_dim))

		# init weights
		w = self.patch_embed.patch_model.weight.data
		nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
		nn.init.xavier_uniform_(self.decoder_pred.weight)
		nn.init.constant_(self.decoder_pred.bias, 0)
		nn.init.trunc_normal_(self.mask_token, std=.02)
		
		if pretrain_pth is not None:		
			self.init_weights(pretrain_pth)
	
	def init_weights(self, pretrain_pth):
		init_from_kinetics_pretrain_(self, pretrain_pth)
	
	@torch.jit.ignore
	def no_weight_decay_keywords(self):
		return {'pos_embed', 'cls_token', 'mask_token'}

	def forward(self, x, target_x, mask, cube_marker, visualize=False):
		x = self.forward_features(x, mask)
		x = self.decoder_pred(x)
		x = x[:, 1:, :]
		
		# reshape to original x
		x = rearrange(x, 'b (t h w) (dt dc) -> b (t dt) h w dc', 
			dt=self.stride[0],
			t=self.num_frames//self.stride[0],
			h=self.img_size//(self.stride[1]*self.downsample_rate),
			w=self.img_size//(self.stride[2]*self.downsample_rate))
		
		# find the center frame of the mask cube
		mask = repeat(mask, 'b t h w -> b (t dt) h w', dt=self.stride[0])
		center_index = torch.zeros(self.num_frames, device=mask.device).to(torch.bool)
		for i, mark_item in enumerate(cube_marker): # loop for batch 
			for marker in mark_item: # loop for video sample
				start_frame, span_frame = marker
				center_index[start_frame*self.stride[0] + span_frame*self.stride[0]//2] = 1 # center index extends to 16
			mask[i, ~center_index] = 0
			center_index.zero_() # reset for new sample
		
		# compute loss on mask regions in center frame
		loss = (x - target_x) ** 2
		loss = loss.mean(dim=-1)
		loss = (loss * mask).sum() / (mask.sum() + 1e-5)
		
		# visulize(need to update)
		if visualize:
			mask_preds = x[:, center_index]
			mask_preds = rearrange(mask_preds, 'b t h w (dh dw c o) -> b t (h dh) (w dw) c o', dh=2,dw=2,c=3,o=9) # need to unnormalize
			return x, loss, mask_preds, center_index
		else:
			return x, loss
	
	def forward_features(self, x, mask=None):
		x = self.patch_embed(x.transpose(1,2))
		# apply mask tokens
		B, L, C = x.shape
		if mask is not None:
			mask_token = self.mask_token.expand(B, L, -1)
			dense_mask = repeat(mask, 'b t h w -> b t (h dh) (w dw)', dh=self.downsample_rate, dw=self.downsample_rate) # nearest-neighbor resize
			w = dense_mask.flatten(1).unsqueeze(-1).type_as(mask_token)
			x = x * (1 - w) + mask_token * w
		# forward network
		x = self.mvit(x)
		return x


def parse_args():
	parser = argparse.ArgumentParser(description='lr receiver')

	# Model
	parser.add_argument(
		'-arch', type=str, default='mvit',
		help='the choosen model arch from [timesformer, vivit]')
	# Training/Optimization parameters
	parser.add_argument(
		'-optim_type', type=str, default='adamw',
		help='the optimizer using in the training')
	parser.add_argument(
		'-lr', type=float, default=0.0005,
		help='the initial learning rate')
	parser.add_argument(
		'-layer_decay', type=float, default=1,
		help='the value of layer_decay')
	parser.add_argument(
		'-weight_decay', type=float, default=0.05, 
		help="""Initial value of the weight decay. With ViT, a smaller value at the beginning of training works well.""")
	
	args = parser.parse_args()
	
	return args

if __name__ == '__main__':
	# Unit test for model runnable experiment and Hog prediction
	import random
	import numpy as np
	#from mask_generator import CubeMaskGenerator
	import data_transform as T
	from dataset import DecordInit, extract_hog_features, temporal_sampling, denormalize, show_processed_image
	#from skimage import io, draw
	#from skimage.feature import _hoghistogram
	#from torchvision.transforms import ToPILImage
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	
	model = MaskFeat(pool_q_stride_size=[[1, 1, 2, 2], [3, 1, 2, 2]], feature_dim=2*2*2*3*9)
	for name, param in model.decoder_pred.named_parameters():
		param.requires_grad = False
	from optimizer import build_optimizer
	import argparse
	hparams = parse_args()
	optimizer = build_optimizer(hparams, model, is_pretrain=False)
	print(optimizer)
	
	'''
	model = TimeSformer(num_frames=4,
						img_size=224,
						patch_size=16,
						pretrained='./pretrain_model/pretrain_mae_vit_base_mask_0.75_400e.pth',
						attention_type='divided_space_time',
						use_learnable_pos_emb=True,
						return_cls_token=True)
	'''
	'''
	model = ViViT(num_frames=4, 
				  img_size=224,
				  patch_size=16,
				  pretrained='./pretrain_model/pretrain_mae_vit_base_mask_0.75_400e.pth',
				  attention_type='divided_space_time',
				  use_learnable_pos_emb=False,
				  return_cls_token=False)
	'''
	# To be reproducable
	'''
	import random 
	SEED = 0
	torch.random.manual_seed(SEED)
	np.random.seed(SEED)
	random.seed(SEED)
	'''
	'''
	# 1. laod pretrained model
	from weight_init import replace_state_dict
	model_name = 'maskfeat'
	model = MaskFeat(pool_q_stride_size=[[1, 1, 2, 2], [3, 1, 2, 2]], feature_dim=2*2*2*3*9)
	state_dict = torch.load(f'./{model_name}_model.pth')['state_dict']
	
	replace_state_dict(state_dict)
	missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
	utils.print_on_rank_zero(missing_keys, unexpected_keys)
	model.eval()
	model = model.to(device)
	
	# 2. prepare data
	#input = torch.rand(2,16,3,224,224)
	#target_x = torch.rand(2,16,14,14,2*2*3*9)
	path = './test.mp4'
	mask_generator = CubeMaskGenerator(input_size=(8,14,14),min_num_patches=16)

	v_decoder = DecordInit()
	v_reader = v_decoder(path)
	# Sampling video frames
	total_frames = len(v_reader)

	align_transform = T.Compose([
		T.RandomResizedCrop(size=(224, 224), area_range=(0.9, 1.0), interpolation=3), #InterpolationMode.BICUBIC
		])
	mean = (0.485, 0.456, 0.406) 
	std = (0.229, 0.224, 0.225) 
	aug_transform = T.Compose([T.ToTensor(),T.Normalize(mean,std)])
	temporal_sample = T.TemporalRandomCrop(16*4)
	start_frame_ind, end_frame_ind = temporal_sample(total_frames)
	frame_indice = np.linspace(0, end_frame_ind-start_frame_ind-1, 
							   16, dtype=int)
	video = v_reader.get_batch(frame_indice).asnumpy()
	del v_reader
	
	# Video align transform: T C H W
	with torch.no_grad():
		video = torch.from_numpy(video).permute(0,3,1,2)
		align_transform.randomize_parameters()
		video = align_transform(video)
		unnorm_video = video
	label = np.stack(list(map(extract_hog_features, video.permute(0,2,3,1).numpy())), axis=0) # T H W C -> T H' W' C'
	#_, hog_image = hog(video.permute(0,2,3,1).numpy()[0][:,:,2], orientations=9, pixels_per_cell=(8, 8), cells_per_block=(1, 1), block_norm='L2', feature_vector=False, visualize=True)
	mask, cube_marker = mask_generator() # T' H' W'

	video = aug_transform(video)
	label = torch.from_numpy(label)
	mask = torch.from_numpy(mask)
	video = video.to(device)
	label = label.to(device)
	mask = mask.to(device)
	print(video.shape, label.shape, mask.shape, cube_marker)
	
	# 3. mask hog prediction
	with torch.no_grad():
		out, loss, mask_preds, center_index = model(video.unsqueeze(0), label.unsqueeze(0), mask.unsqueeze(0), [cube_marker], visualize=True)
	print(out.shape, loss.item(), mask_preds.shape, center_index)
	# (1)visualize the de-norm hog prediction map of target frame
	eps=1e-5
	mask_preds = mask_preds[:,:,:,:,0,:][0][0].detach().cpu().numpy()
	real_norm = get_hog_norm(unnorm_video.permute(0,2,3,1).numpy()[center_index][0,:,:,0], orientations=9, pixels_per_cell=(8, 8), cells_per_block=(1, 1))
	print(real_norm.shape) # [28,28,1]
	mask_preds = mask_preds * real_norm
	save_path = f'./hog_pred.jpg'
	visualize_hog(mask_preds, save_path)

	# (2)visualize the original target frame
	save_path = f'./real_img.jpg'
	io.imsave(save_path, unnorm_video.permute(0,2,3,1).numpy()[center_index][0,:,:,0])
	
	# (3)visualize the target masked frame
	mask = repeat(mask, 't h w -> (t dt) (h dh) (w dw)', dt=2, dh=16, dw=16).detach().cpu() # [16, 224, 224]
	mask = 1 - mask
	img_mask = unnorm_video.permute(0,2,3,1)[center_index][0,:,:,0] * mask[center_index][0]
	save_path = f"./mask_img.jpg"
	io.imsave(save_path, img_mask.numpy())
	'''