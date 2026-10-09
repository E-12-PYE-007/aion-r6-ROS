"""
Reference Async
"""

import torch
import torch.nn as nn
from typing import Optional
from efficientnet_pytorch import EfficientNet

from ag_vla.vint_self_attention import MultiLayerDecoder_trans

NUM_ACTIONS_CHUNK = 8

class Edge_adapter(nn.Module):
    def __init__(
        self,
        obs_encoding_size: Optional[int] = 512,
        mha_num_attention_heads: Optional[int] = 2,
        mha_num_attention_layers: Optional[int] = 2,
        mha_ff_dim_factor: Optional[int] = 4,
    ) -> None:
        super(Edge_adapter, self).__init__()
        self.obs_encoding_size = obs_encoding_size

        # For small head model
        self.cat_encoder = EfficientNet.from_name("efficientnet-b0", in_channels=6) # context
        self.num_cat_features = self.cat_encoder._fc.in_features
        self.obs_encoder = EfficientNet.from_name("efficientnet-b0", in_channels=3) # context
        self.num_obs_features = self.obs_encoder._fc.in_features

        if self.num_obs_features != self.obs_encoding_size:
            self.compress_obs_enc = nn.Linear(self.num_obs_features, self.obs_encoding_size)
        else:
            self.compress_obs_enc = nn.Identity()
        if self.num_cat_features != self.obs_encoding_size:
            self.compress_cat_enc = nn.Linear(self.num_cat_features, self.obs_encoding_size)
        else:
            self.compress_cat_enc = nn.Identity()

        self.decoder = MultiLayerDecoder_trans(
            embed_dim=self.obs_encoding_size,
            seq_len=8+1+1,
            output_layers=[256, 128, 64, 32],
            nhead=mha_num_attention_heads,
            num_layers=mha_num_attention_layers,
            ff_dim_factor=mha_ff_dim_factor,
        )

        self.action_predictor = nn.Sequential(
            nn.Linear(self.obs_encoding_size, 256),   
            nn.ReLU(),            
            nn.Linear(256, 128),  
            nn.ReLU(),            
            nn.Linear(128, 64),  
            nn.ReLU(),            
            nn.Linear(64, 8 * 4),         
        )
    def forward(self, obs_img: torch.tensor, past_img: torch.tensor, vla_feature: torch.tensor) -> torch.Tensor:

        # For small head model
        batch_size = obs_img.shape[0]
        cat_img = torch.cat((obs_img, past_img), dim=1)
        cat_encoding = self.cat_encoder.extract_features(cat_img)
        cat_encoding = self.cat_encoder._avg_pooling(cat_encoding)
        if self.cat_encoder._global_params.include_top:
            cat_encoding = cat_encoding.flatten(start_dim=1)
            cat_encoding = self.cat_encoder._dropout(cat_encoding)
        cat_encoding = self.compress_cat_enc(cat_encoding)            

        obs_encoding = self.obs_encoder.extract_features(obs_img)
        obs_encoding = self.obs_encoder._avg_pooling(obs_encoding)
        if self.obs_encoder._global_params.include_top:
            obs_encoding = obs_encoding.flatten(start_dim=1)
            obs_encoding = self.obs_encoder._dropout(obs_encoding)
        obs_encoding = self.compress_obs_enc(obs_encoding)            

        tokens = torch.cat((vla_feature, obs_encoding.unsqueeze(1), cat_encoding.unsqueeze(1)), dim=1)
        tokens = self.decoder(tokens)[:,-2:-1,:]

        x = tokens.reshape(tokens.shape[0], -1)
        action_pred = self.action_predictor(x).reshape(batch_size, NUM_ACTIONS_CHUNK, -1)

        return action_pred
