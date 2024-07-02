import copy
import math
import logging
import numpy as np
import os
import pickle

import timm
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import MultiheadAttention
from collections import OrderedDict

from transformers import ASTModel, ASTConfig, ASTFeatureExtractor
from einops import rearrange

from diffusion_policy.common.pytorch_util import replace_submodules
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.model.vision.audio_encoder import AudioEncoder

logger = logging.getLogger(__name__)

class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)

def get_embedding_dim_from_shape_meta(obs_shape_meta, v_feature_dim, a_feature_dim):
    embedding_dim = 0
    for key, attr in obs_shape_meta.items():
        if 'camera' in key:
            embedding_dim += v_feature_dim * attr['horizon']
        elif 'mic' in key:
            embedding_dim += a_feature_dim
        else:
            embedding_dim += attr['shape'][0] * attr['horizon']
    
    return embedding_dim

class ManiWAVObsEncoder(ModuleAttrMixin):
    def __init__(self,
            shape_meta: dict,
            fusion_mode: str,
            num_heads: int,
            position_encoding: str,
            vision_encoder_cfg: dict,
            audio_encoder_cfg: dict
        ):
        """
        Assumes rgb input: B,T,C,H,W
        Assumes audio input: B,T,D
        Assumes low_dim input: B,T,D
        """
        super().__init__()
        
        rgb_keys = list()
        low_dim_keys = list()
        audio_keys = list()
        key_model_map = nn.ModuleDict()
        key_transform_map = nn.ModuleDict()
        key_proj_map = nn.ModuleDict()
        key_shape_map = dict()

        vision_model = timm.create_model(
            model_name=vision_encoder_cfg.model_name,
            pretrained=vision_encoder_cfg.pretrained,
            global_pool=vision_encoder_cfg.global_pool, # '' means no pooling
            num_classes=0            # remove classification layer
        )
        if vision_encoder_cfg.frozen:
            assert vision_encoder_cfg.pretrained
            for param in vision_model.parameters():
                param.requires_grad = False

        if audio_encoder_cfg.model_name == 'ast':
            config = ASTConfig()
            config.max_length = audio_encoder_cfg.max_length
            config.num_mel_bins = audio_encoder_cfg.num_mel_bins
            audio_model = ASTModel(config)
            self.ast_feature_extractor = ASTFeatureExtractor(
                num_mel_bins=audio_encoder_cfg.num_mel_bins, 
                max_length=audio_encoder_cfg.max_length,
                do_normalize=False,
                # mean=audio_encoder_cfg.norm_spec.mean,
                # std=audio_encoder_cfg.norm_spec.std
            )
        elif audio_encoder_cfg.model_name.startswith('resnet'):
            audio_model = timm.create_model(
                model_name=audio_encoder_cfg.model_name,
                pretrained=audio_encoder_cfg.pretrained,
                global_pool=audio_encoder_cfg.global_pool,
                num_classes=0
            )

        # get normalization stats
        if audio_encoder_cfg.norm_spec.is_norm: # normalize to -1 and 1
            if os.path.exists(f'{audio_encoder_cfg.norm_spec.stats_dir}/spec_stats.pkl'):
                with open(f'{audio_encoder_cfg.norm_spec.stats_dir}/spec_stats.pkl', 'rb') as f:
                    stats = pickle.load(f)
                    audio_encoder_cfg.norm_spec.min = stats['mic_0']['min']
                    audio_encoder_cfg.norm_spec.max = stats['mic_0']['max']
        
        feature_dim = None
        if vision_encoder_cfg.model_name.startswith('resnet'):
            # the last layer is nn.Identity() because num_classes is 0
            # second last layer is AdaptivePool2d, which is also identity because global_pool is empty
            if vision_encoder_cfg.downsample_ratio == 32:
                modules = list(vision_model.children())[:-2]
                vision_model = torch.nn.Sequential(*modules)
                feature_dim = 512
            elif vision_encoder_cfg.downsample_ratio == 16:
                modules = list(vision_model.children())[:-3]
                vision_model = torch.nn.Sequential(*modules)
                feature_dim = 256
            else:
                raise NotImplementedError(f"Unsupported downsample_ratio: {vision_encoder_cfg.downsample_ratio}")
        elif vision_encoder_cfg.model_name.startswith('convnext'):
            # the last layer is nn.Identity() because num_classes is 0
            # second last layer is AdaptivePool2d, which is also identity because global_pool is empty
            if vision_encoder_cfg.downsample_ratio == 32:
                modules = list(model.children())[:-2]
                model = torch.nn.Sequential(*modules)
                feature_dim = 1024
            else:
                raise NotImplementedError(f"Unsupported downsample_ratio: {vision_encoder_cfg.downsample_ratio}")
        elif vision_encoder_cfg.model_name.startswith('vit'):
            feature_dim = 768
        self.v_feature_dim = feature_dim
        
        feature_dim = None       
        if audio_encoder_cfg.model_name.startswith('resnet'):
            # the last layer is nn.Identity() because num_classes is 0
            # second last layer is AdaptivePool2d, which is also identity because global_pool is empty
            if audio_encoder_cfg.downsample_ratio == 32:
                modules = list(audio_model.children())[:-2]
                audio_model = torch.nn.Sequential(*modules)
                feature_dim = 512
            elif audio_encoder_cfg.downsample_ratio == 16:
                modules = list(audio_model.children())[:-3]
                audio_model = torch.nn.Sequential(*modules)
                feature_dim = 256
            else:
                raise NotImplementedError(f"Unsupported downsample_ratio: {audio_encoder_cfg.downsample_ratio}")
            
            audio_model = AudioEncoder(audio_model, norm_spec=audio_encoder_cfg.norm_spec)
        elif audio_encoder_cfg.model_name == 'ast':
            feature_dim = 768
        self.a_feature_dim = feature_dim
            
        if vision_encoder_cfg.use_group_norm:
            vision_model = replace_submodules(
                root_module=vision_model,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=(x.num_features // 16) if (x.num_features % 16 == 0) else (x.num_features // 8), 
                    num_channels=x.num_features)
                )
        if audio_encoder_cfg.use_group_norm:
            audio_model = replace_submodules(
                root_module=audio_model,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=(x.num_features // 16) if (x.num_features % 16 == 0) else (x.num_features // 8), 
                    num_channels=x.num_features)
                )
        
        image_shape = None
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                assert image_shape is None or image_shape == shape[1:]
                image_shape = shape[1:]
        
        vision_transforms = vision_encoder_cfg.transforms
        if vision_transforms is not None and not isinstance(vision_transforms[0], torch.nn.Module):
            assert vision_transforms[0].type == 'RandomCrop'
            ratio = vision_transforms[0].ratio
            vision_transforms = [
                torchvision.transforms.RandomCrop(size=int(image_shape[0] * ratio)),
                torchvision.transforms.Resize(size=image_shape[0], antialias=True)
            ] + vision_transforms[1:]
        vision_transform = nn.Identity() if vision_transforms is None else torch.nn.Sequential(*vision_transforms)
        audio_transform = nn.Identity() if audio_encoder_cfg.transforms is None else torch.nn.Sequential(*audio_encoder_cfg.transforms)

        obs_shape_meta = shape_meta['obs']
        if fusion_mode == 'mlp':
            self.mlp = nn.Sequential(
                nn.Linear(self.v_feature_dim*3, self.v_feature_dim*2),
                nn.ReLU(),
                nn.Linear(self.v_feature_dim*2, self.v_feature_dim)
            )
        elif fusion_mode == 'modality-attention':
            self.transformer_encoder = torch.nn.TransformerEncoderLayer(
                d_model=self.v_feature_dim*obs_shape_meta['camera0_rgb']['horizon'],
                nhead=num_heads,
                dim_feedforward=2048,
                batch_first=True,
                dropout=0.0
            )
            self.audio_proj_layer = nn.Linear(self.a_feature_dim, self.v_feature_dim*obs_shape_meta['camera0_rgb']['horizon'])
            self.projection_layer = nn.Linear(self.v_feature_dim*obs_shape_meta['camera0_rgb']['horizon'] * 2, self.v_feature_dim)
        
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            key_shape_map[key] = shape
            if type == 'rgb':
                rgb_keys.append(key)

                this_model = vision_model if vision_encoder_cfg.share_rgb_model else copy.deepcopy(vision_model)
                key_model_map[key] = this_model

                this_transform = vision_transform
                key_transform_map[key] = this_transform
            elif type == 'audio':
                audio_keys.append(key)
                                
                this_model = audio_model if audio_encoder_cfg.share_audio_model else copy.deepcopy(audio_model)
                this_model.audio_key = key
                key_model_map[key] = this_model
                key_proj_map[key] = nn.Linear(self.a_feature_dim, self.v_feature_dim)
                
                this_transform = audio_transform
                key_transform_map[key] = this_transform
                
            elif type == 'low_dim':
                low_dim_keys.append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")
            
        rgb_keys = sorted(rgb_keys)
        audio_keys = sorted(audio_keys)
        low_dim_keys = sorted(low_dim_keys)

        self.shape_meta = shape_meta
        self.fusion_mode = fusion_mode
        self.key_model_map = key_model_map
        self.key_transform_map = key_transform_map
        self.rgb_keys = rgb_keys
        self.low_dim_keys = low_dim_keys
        self.audio_keys = audio_keys
        self.key_shape_map = key_shape_map
        self.vision_encoder_cfg = vision_encoder_cfg
        self.audio_encoder_cfg = audio_encoder_cfg
        
        feature_map_shape = [x // self.vision_encoder_cfg.downsample_ratio for x in image_shape]
        if vision_encoder_cfg.model_name.startswith('vit'):
            if vision_encoder_cfg.feature_aggregation == 'all_tokens':
                # Use all tokens from ViT
                pass
            elif vision_encoder_cfg.feature_aggregation is not None:
                logger.warn(f'vit will use the CLS token. feature_aggregation ({vision_encoder_cfg.feature_aggregation}) is ignored!')
                vision_encoder_cfg.feature_aggregation = None
        if vision_encoder_cfg.feature_aggregation == 'soft_attention':
            self.attention = nn.Sequential(
                nn.Linear(self.v_feature_dim, 1, bias=False),
                nn.Softmax(dim=1)
            )
        elif vision_encoder_cfg.feature_aggregation == 'spatial_embedding':
            self.spatial_embedding = torch.nn.Parameter(torch.randn(feature_map_shape[0] * feature_map_shape[1], feature_dim))
        elif vision_encoder_cfg.feature_aggregation == 'attention_pool_2d':
            self.attention_pool_2d = AttentionPool2d(
                spacial_dim=feature_map_shape[0],
                embed_dim=self.v_feature_dim,
                num_heads=self.v_feature_dim // 64,
                output_dim=self.v_feature_dim
            )
        
        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )


    def aggregate_feature(self, feature, model_name, agg_mode):
        if model_name.startswith('vit') or model_name.startswith('ast'):
            if agg_mode == 'all_tokens':
                return feature
            elif agg_mode is None: # use the CLS token
                return feature[:, 0, :]
        
        if len(feature.shape) == 2:
            return feature
        
        # resnet
        assert len(feature.shape) == 4
        if agg_mode == 'attention_pool_2d':
            return self.attention_pool_2d(feature)

        feature = torch.flatten(feature, start_dim=-2) # B, 512, 7*7
        feature = torch.transpose(feature, 1, 2) # B, 7*7, 512

        if agg_mode == 'avg':
            return torch.mean(feature, dim=[1])
        elif agg_mode == 'max':
            return torch.amax(feature, dim=[1])
        elif agg_mode == 'soft_attention':
            weight = self.attention(feature)
            return torch.sum(feature * weight, dim=1)
        elif agg_mode == 'spatial_embedding':
            return torch.mean(feature * self.spatial_embedding, dim=1)
        else:
            assert agg_mode is None
            return feature

    def forward(self, obs_dict):
        features = list()
        modality_features = list()
        modality_feature_maps = list()
        low_dim_features = list()
        batch_size = next(iter(obs_dict.values())).shape[0]
        
        # process rgb input
        for key in self.rgb_keys:
            img = obs_dict[key]
            B, T = img.shape[:2]
            assert B == batch_size
            assert img.shape[2:] == self.key_shape_map[key]
            img = img.reshape(B*T, *img.shape[2:])
            img = self.key_transform_map[key](img)
            raw_feature = self.key_model_map[key](img)
            feature = self.aggregate_feature(raw_feature, 
                                             model_name=self.vision_encoder_cfg.model_name,
                                             agg_mode=self.vision_encoder_cfg.feature_aggregation)
            # assert len(feature.shape) == 2 and feature.shape[0] == B * T
            features.append(feature.reshape(B, -1))
            if self.fusion_mode == 'modality-attention':
                modality_features.append(feature.reshape(B, -1))
            
        # process audio input
        for key in self.audio_keys:
            audio = obs_dict[key]
            B, T = audio.shape[:2]
            assert B == batch_size
            assert audio.shape[2:] == self.key_shape_map[key]

            if self.audio_encoder_cfg.model_name == 'ast':
                audio = audio.reshape(B, audio.shape[-1]*T)
                audio = self.key_transform_map[key][0](audio)
                audio = self.ast_feature_extractor(audio.cpu().numpy(), sampling_rate=16000)['input_values']
                if self.audio_encoder_cfg.norm_spec.is_norm: # normalize to -1 and 1
                    audio = (np.array(audio) - self.audio_encoder_cfg.norm_spec.min) \
                        / (self.audio_encoder_cfg.norm_spec.max - self.audio_encoder_cfg.norm_spec.min)
                    audio = audio * 2 - 1
                audio = torch.tensor(np.array(audio)).to(self.device)
                audio = self.key_transform_map[key][1:](audio)
                if self.audio_encoder_cfg.mask_robot:
                    audio[:, :, :8] = 0
            else:
                audio = audio.reshape(B, 1, audio.shape[-1]*T)
                audio = self.key_transform_map[key](audio)
            
            if self.audio_encoder_cfg.model_name == 'ast':
                raw_feature = self.key_model_map[key](audio, output_hidden_states=True)
                raw_feature = raw_feature.last_hidden_state
            else:
                raw_feature = self.key_model_map[key](audio)
            feature = self.aggregate_feature(raw_feature,
                                             model_name=self.audio_encoder_cfg.model_name,
                                             agg_mode=self.audio_encoder_cfg.feature_aggregation)
            features.append(feature.reshape(B, -1))
            if self.fusion_mode == 'modality-attention':
                if feature.shape[-1] != self.v_feature_dim * self.shape_meta['obs']['camera0_rgb']['horizon']:
                    feature = self.audio_proj_layer(feature.reshape(B, -1))
                modality_features.append(feature)
                
        # process lowdim input
        for key in self.low_dim_keys:
            data = obs_dict[key]
            B, T = data.shape[:2]
            assert B == batch_size
            assert data.shape[2:] == self.key_shape_map[key]
            features.append(data.reshape(B, -1))
            low_dim_features.append(data.reshape(B, -1))
        
        # concatenate all features
        if self.fusion_mode == 'concat':
            result = torch.cat(features, dim=-1)
        # [Baseline]: Play it by ear https://arxiv.org/abs/2205.14850
        elif self.fusion_mode == 'mlp':
            result = self.mlp(torch.cat(features[:2], dim=-1))
            result = torch.concat([result, torch.cat(low_dim_features, dim=-1)], dim=1)
        elif self.fusion_mode == 'modality-attention':
            # fuse vision and audio with attention layer, and then concatenate with low_dim
            in_embeds = torch.stack(modality_features, dim=0).permute(1, 0, 2)  # [batch, 2, D]
            out_embeds = self.transformer_encoder(in_embeds)  # [batch, 2, D]
            # out_embeds += in_embeds
            result = torch.concat([out_embeds[:, i] for i in range(out_embeds.shape[1])], dim=1)
            result = self.projection_layer(result)
            result = torch.concat([result, torch.cat(low_dim_features, dim=-1)], dim=1)
        return result


    @torch.no_grad()
    def output_shape(self):
        example_obs_dict = dict()
        obs_shape_meta = self.shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            this_obs = torch.zeros(
                (1, attr['horizon']) + shape, 
                dtype=self.dtype,
                device=self.device)
            example_obs_dict[key] = this_obs
        example_output = self.forward(example_obs_dict)
        assert len(example_output.shape) == 2
        assert example_output.shape[0] == 1
        
        return example_output.shape


if __name__=='__main__':
    timm_obs_encoder = ManiWAVObsEncoder(
        shape_meta=None,
        model_name='resnet18.a1_in1k',
        pretrained=False,
        global_pool='avg',
        transforms=None
    )