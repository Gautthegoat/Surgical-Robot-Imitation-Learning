import numpy as np
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights
from torchvision.models.resnet import ResNet, BasicBlock
from torchvision.models import vit_b_16, ViT_B_16_Weights

from source.models.components import get_sinusoid_encoding_table, reparameterize, ResNetFactory, Permute

class ACTModel(nn.Module):
    def __init__(self, config):
        super(ACTModel, self).__init__()
        # Configuration for the projections
        num_queries = config['data']['length_actions'] # Number of actions to predict
        hidden_dim = config['model']['embed_dim'] # Dimension of the embeddings

        # Configuration for the model architecture
        self.style_dim = config['model']['style_dim'] # Dimension of the style variable
        self.take_current_pose = config['data']['observations']['current_pos'] # Take current pose as input
        self.take_past_pose = config['data']['observations']['past_obs'] # Take past pose as input # TODO
        self.type_predictions = config['data']['type'] # Type of predictions
        self.use_style = config['model']['use_style'] # Use style variable
        self.kl_weight = config['training']['kl_weight']
        
        # Image encoder
        if config['data']['prediction'] == "joint_angles":
            joints_dim = 12
        elif config['data']['prediction'] == "end_effector":
            joints_dim = 8 # 6 for end deffector position and 2 for tool activation
        else:
            raise ValueError(f"Prediction type {config['data']['prediction']} not found.")
        try:
            resnet_factory = ResNetFactory(
                model_name=config['model']['image_encoder']['type'], 
                pretrained=config['model']['image_encoder']['pretrained'], 
                keep_last_layer=False, 
                freeze_bn=config['model']['image_encoder']['freeze_batchnorm']
            )
        except:
            raise ValueError(f"Error while creating the ResNet model for the image encoder.")
        resnet = resnet_factory.model
        resnet_out_dim = resnet_factory.size_output
        self.image_encoder = nn.Sequential(
            resnet,
            nn.Conv2d(resnet_out_dim, hidden_dim, kernel_size=1),
            nn.Flatten(2),
            Permute(0, 2, 1)
        )

        # Transformer encoder for style variable
        self.transformer_encoder_style = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=config['model']['style_encoder']['num_heads'],
                dim_feedforward=config['model']['style_encoder']['feedforward_dim'],
                dropout=config['model']['style_encoder']['dropout'],
                batch_first=True
            ),
            num_layers=config['model']['style_encoder']['num_layers']
        )

        # Transformer for action prediction
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=config['model']['transformer']['num_heads'],
            num_encoder_layers=config['model']['transformer']['num_encoder_layers'],
            num_decoder_layers=config['model']['transformer']['num_decoder_layers'],
            dim_feedforward=config['model']['transformer']['feedforward_dim'],
            dropout=config['model']['transformer']['dropout'],
            batch_first=True
        )
        
        # Projections and embeddings
        self.encoder_action_proj = nn.Linear(joints_dim, hidden_dim)
        self.encoder_joints_proj = nn.Linear(joints_dim, hidden_dim)
        self.cls_embed = nn.Embedding(1, hidden_dim)
        self.pos_table = nn.Parameter(get_sinusoid_encoding_table(1+1+num_queries, hidden_dim), requires_grad=False)
        self.latent_proj = nn.Linear(hidden_dim, self.style_dim*2)
        self.latent_out_proj = nn.Linear(self.style_dim, hidden_dim)
        self.input_joint_proj = nn.Linear(joints_dim, hidden_dim)
        self.tgt = nn.Embedding(num_queries, hidden_dim)
        self.pos_transformer_input = nn.Embedding(1+1+1, hidden_dim) # Positional encoding for transformer input, 1 for CLS token, 1 for qpos, 1 for features (in the paper's code they actually don't use positional encoding for features...)
        self.action_out_proj = nn.Linear(hidden_dim, joints_dim)
        
        # For the hybrid prediction
        self.mask_relative = torch.ones(12)
        self.mask_relative[5] = self.mask_relative[11] = 0
        self.mask_absolute = torch.zeros(12)
        self.mask_absolute[5] = self.mask_absolute[11] = 1

        # Reset the parameters of only the transformer layers USED IN ACT
        # self._reset_parameters()

        # Loss function type
        loss_type = config['training']['loss']  # Assume you have a loss type in the config
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss()
        elif loss_type == 'mse':
            self.loss_fn = nn.MSELoss()
        else:
            raise ValueError(f"Loss {loss_type} not found.")

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
    def forward(self, inputs):
        """
        inputs: (current frame, current qpos, future actions)
        current frame: (batch_size, channels, height, width)
        current qpos: (batch_size, joints_dim)
        futur actions: (batch_size, chunk_size, joints_dim)
        """
        frame, qpos, actions = inputs
        qpos = qpos.unsqueeze(1)  # Add sequence dimension to qpos

        training = actions is not None
        bs = frame.shape[0]  # Get batch size

        # Style variable        
        if training and self.use_style:
            action_embed = self.encoder_action_proj(actions)  # Project actions to embedding space
            qpos_embed = self.encoder_joints_proj(qpos)  # Project qpos to embedding space
            cls_embed = self.cls_embed.weight  # Get CLS token embedding
            cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(bs, 1, 1)  # Repeat CLS token for batch size
            encoder_input = torch.cat([cls_embed, qpos_embed, action_embed], dim=1)  # Concatenate CLS, qpos, and actions
            pos_embed = self.pos_table.clone().detach().to(frame.device).repeat(bs, 1, 1)  # Get positional embeddings
            encoder_input += pos_embed  # Add positional embeddings to input
            
            encoder_output = self.transformer_encoder_style(encoder_input)  # Pass through transformer encoder
            encoder_output = encoder_output[:, 0, :]  # Extract CLS token output
            latent_info = self.latent_proj(encoder_output)  # Project to latent space
            mu, logvar = latent_info[:, :self.style_dim], latent_info[:, self.style_dim:]  # Split into mean and log variance
            latent_sample = reparameterize(mu, logvar)  # Sample from latent distribution
            latent_input = self.latent_out_proj(latent_sample)  # Project latent sample to input space
            latent_input = torch.unsqueeze(latent_input, axis=1)  # Add sequence dimension
        else:
            mu, logvar = torch.zeros([bs, self.style_dim], dtype=torch.float32).to(frame.device), torch.zeros([bs, self.style_dim], dtype=torch.float32).to(frame.device)  # Zero latent distribution parameters
            latent_sample = torch.zeros([bs, self.style_dim], dtype=torch.float32).to(frame.device)  # Zero latent sample
            latent_input = self.latent_out_proj(latent_sample)  # Project zero latent to input space
            latent_input = torch.unsqueeze(latent_input, axis=1)  # Add sequence dimension
        
        # Observations encoder
        features = self.image_encoder(frame)  # Extract features from image

        pos_embed = self.pos_transformer_input.weight  # Get positional embeddings for transformer input
        pos_embed = torch.unsqueeze(pos_embed, axis=0).repeat(bs, 1, 1)  # Repeat for batch size
        
        latent_input += pos_embed[:, 0, :].unsqueeze(1)  # Add positional embedding to latent input
        features += pos_embed[:, 2, :].unsqueeze(1)  # Add positional embedding to features

        if self.take_current_pose:
            qpos_input = self.input_joint_proj(qpos)  # Project current qpos to embedding space
            qpos_input += pos_embed[:, 1, :].unsqueeze(1)  # Add positional embedding to qpos input
            transformer_input = torch.cat([latent_input, qpos_input, features], dim=1)  # Concatenate all inputs for transformer
        else:
            transformer_input = torch.cat([latent_input, features], dim=1)

        transformer_output = self.transformer(src=transformer_input, tgt=self.tgt.weight.unsqueeze(0).repeat(bs, 1, 1))  # Pass through transformer
        action_output = self.action_out_proj(transformer_output)  # Project transformer output to action space

        if self.type_predictions == "Relative":
            action_output += qpos
        elif self.type_predictions == "Hybrid":
            action_relative = (action_output+qpos) * self.mask_relative.to(frame.device)
            action_absolute = action_output * self.mask_absolute.to(frame.device)
            action_output = action_relative + action_absolute

        return (action_output, mu, logvar)  # Return predicted actions and latent distribution parameters
    
    def criterion_training(self, predictions, targets):
        pred_actions, mu, logvar = predictions
        true_actions = targets
        # Reconstruction loss
        reconstruction_loss = self.loss_fn(pred_actions, true_actions)
        # KL Divergence for the VAE-style latent variable
        kl_divergence = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        # Total loss is as a weighted sum of the two losses
        total_loss = reconstruction_loss + self.kl_weight * kl_divergence
        return total_loss
    
    def criterion_evaluation(self, predictions, targets):
        pred_actions, _, _ = predictions
        true_actions = targets
        # Always use L1 loss for evaluation, as it is more interpretable and so that we can compare with other configurations and models
        return nn.L1Loss()(pred_actions, true_actions)