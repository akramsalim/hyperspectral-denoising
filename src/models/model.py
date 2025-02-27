import torch
import torch.nn as nn
import timm
from pathlib import Path

# Modified Vision Transformer from new code
class ModifiedViT(nn.Module):
    def __init__(self, model_size='base', num_input_channels=202, img_size=128):
        super(ModifiedViT, self).__init__()
        
        if model_size == 'base':
            self.vit = timm.create_model('vit_small_patch16_224', pretrained=False,patch_size=4, img_size=img_size)
            expected_embed_dim = 384
        elif model_size == 'large':
            self.vit = timm.create_model('vit_large_patch16_224', pretrained=False)
            expected_embed_dim = 1024
        else:
            raise ValueError(f"Unsupported model size: {model_size}")
        
        # patch embedding 202 channels
        self.vit.patch_embed.proj = nn.Conv2d(
            in_channels=num_input_channels,
            out_channels=expected_embed_dim,
            kernel_size=self.vit.patch_embed.proj.kernel_size,
            stride=self.vit.patch_embed.proj.stride,
            padding=self.vit.patch_embed.proj.padding,
            bias=self.vit.patch_embed.proj.bias is not None
        )
        
        # initialize layers
        nn.init.kaiming_normal_(self.vit.patch_embed.proj.weight, mode='fan_out', nonlinearity='relu')
        if self.vit.patch_embed.proj.bias is not None:
            nn.init.constant_(self.vit.patch_embed.proj.bias, 0)
        nn.init.normal_(self.vit.pos_embed, std=0.02)
        
    def forward(self, x):
        return self.vit(x)

# mae Encoder from new code
class MAEEncoder(nn.Module):
    def __init__(self, model_size='base', num_input_channels=202, img_size=128):
        super(MAEEncoder, self).__init__()
        self.vit = ModifiedViT(model_size=model_size, num_input_channels=num_input_channels, img_size=img_size).vit
        self.vit.head = nn.Identity()
    
    def forward(self, x):
        return self.vit.forward_features(x)

# Weight loading function from new code
def load_vit_weights(model, state_dict_path, model_size='base'):
    state_dict = torch.load(state_dict_path, map_location='cpu', weights_only=True)
    
    if 'vit_core' in state_dict:
        vit_core_state = state_dict['vit_core']
    else:
        vit_core_state = state_dict
    
    new_state_dict = {}
    for k, v in vit_core_state.items():
        if k.startswith('vit_core.'):
            new_key = k[len('vit_core.'):]
        else:
            new_key = k
        new_state_dict[new_key] = v
    
    exclude_keys = ['patch_embed.proj', 'pos_embed', 'spectral_adapter']
    vit_core_state_filtered = {
        k: v for k, v in new_state_dict.items()
        if not any(excl in k for excl in exclude_keys)
    }
    
    missing_keys, unexpected_keys = model.encoder.vit.load_state_dict(vit_core_state_filtered, strict=False)
    
    if hasattr(model.encoder.vit, 'pos_embed') and model.encoder.vit.pos_embed is not None:
        nn.init.normal_(model.encoder.vit.pos_embed, std=0.02)
    
    return model

# differentee head implementations
class FCHead(nn.Module):
    def __init__(self, in_chans=202, patch_size=4, embed_dim=384):
        super().__init__()
        self.in_chans = in_chans
        self.patch_size = patch_size
        self.fc = nn.Linear(embed_dim, in_chans * patch_size * patch_size)

    def forward(self, features):
        B, N, D = features.shape
        patches = []
        for i in range(N):
            patch_embed = features[:, i, :]
            patch_pixels = self.fc(patch_embed)
            patch_pixels = patch_pixels.reshape(B, self.in_chans, self.patch_size, self.patch_size)
            patches.append(patch_pixels.unsqueeze(1))
        patches = torch.cat(patches, dim=1)
        return patches

class ConvHead(nn.Module):
    def __init__(self, in_chans=202, patch_size=4, embed_dim=384):
        super().__init__()
        self.in_chans = in_chans
        self.patch_size = patch_size
        self.conv = nn.Sequential(
            nn.Conv2d(embed_dim, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, in_chans, kernel_size=3, padding=1)
        )

    def forward(self, features_2d):
        features_2d = nn.functional.interpolate(features_2d, scale_factor=4, mode='bilinear', align_corners=False)
        out = self.conv(features_2d)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + self.net(x)

class ResidualBlockHead(nn.Module):
    def __init__(self, in_chans=202, patch_size=4, embed_dim=384):
        super().__init__()
        self.in_chans = in_chans
        self.patch_size = patch_size
        self.initial_conv = nn.Conv2d(embed_dim, in_chans, kernel_size=3, padding=1)
        self.res_blocks = nn.Sequential(
            ResidualBlock(in_chans),
            ResidualBlock(in_chans),
            ResidualBlock(in_chans)
        )

    def forward(self, features_2d):
        features_2d = nn.functional.interpolate(features_2d, scale_factor=4, mode='bilinear', align_corners=False)
        x = self.initial_conv(features_2d)
        x = self.res_blocks(x)
        return x

class UNetHead(nn.Module):
    def __init__(self, in_chans=202, patch_size=4, embed_dim=384):
        super().__init__()
        self.in_chans = in_chans
        self.patch_size = patch_size

        self.initial_conv = nn.Sequential(
            nn.Conv2d(embed_dim, 256, 3, padding=1),
            nn.ReLU(True)
        )

        self.down1 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2)
        )

        self.bottom = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(True)
        )

        self.up2 = nn.ConvTranspose2d(512, 512, 2, stride=2)
        self.conv_up2 = nn.Sequential(
            nn.Conv2d(512+256, 256, 3, padding=1),
            nn.ReLU(True)
        )
        self.up1 = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.conv_up1 = nn.Sequential(
            nn.Conv2d(256+256, 256, 3, padding=1),
            nn.ReLU(True)
        )

        self.final = nn.Conv2d(256, in_chans, 3, padding=1)

    def forward(self, features_2d):
        features_2d = nn.functional.interpolate(features_2d, scale_factor=4, mode='bilinear', align_corners=False)
        x0 = self.initial_conv(features_2d)

        x1 = self.down1(x0)
        x2 = self.down2(x1)

        btm = self.bottom(x2)

        x2_up = self.up2(btm)
        x2_cat = torch.cat([x2_up, x1], dim=1)
        x2_up = self.conv_up2(x2_cat)

        x1_up = self.up1(x2_up)
        x1_cat = torch.cat([x1_up, x0], dim=1)
        x1_up = self.conv_up1(x1_cat)

        out = self.final(x1_up)
        return out

class DownstreamModel(nn.Module):
    def __init__(self, img_size=128, patch_size=4, in_chans=202, head_type="conv", model_size='base'):
        super().__init__()
        
        # Using MAEEncoder instead of raw VisionTransformer
        self.encoder = MAEEncoder(
            model_size=model_size,
            num_input_channels=in_chans,
            img_size=img_size
        )
        
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.img_size = img_size

        if head_type == "fc":
            self.head = FCHead(in_chans=in_chans, patch_size=patch_size, embed_dim=384)
        elif head_type == "conv":
            self.head = ConvHead(in_chans=in_chans, patch_size=patch_size, embed_dim=384)
        elif head_type == "residual":
            self.head = ResidualBlockHead(in_chans=in_chans, patch_size=patch_size, embed_dim=384)
        elif head_type == "unet":
            self.head = UNetHead(in_chans=in_chans, patch_size=patch_size, embed_dim=384)
        else:
            raise ValueError(f"Unknown head_type: {head_type}")

        # Set parameter requires_grad
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.head.parameters():
            param.requires_grad = True

    def forward(self, x):
        features = self.encoder(x)  # [B,1025,384]
        #print(f"Initial features shape: {features.shape}")

        features = features[:, 1:, :]  # Remove CLS token [B,1024,384]
        #print(f"After removing CLS token: {features.shape}")

        
        B, N, D = features.shape
        h = w = int((self.img_size / self.patch_size))
        #print(f"Expected h,w: {h},{w}, N: {N}")


        if isinstance(self.head, FCHead):
            patches = self.head(features)
            patches = patches.reshape(B, h, w, self.in_chans, self.patch_size, self.patch_size)
            patches = patches.permute(0,3,1,4,2,5)
            out = patches.reshape(B, self.in_chans, h*self.patch_size, w*self.patch_size)
        else:
            feat_map = features.permute(0,2,1).reshape(B, D, h, w)
            #print(f"Reshaped features: {feat_map.shape}")

            out = self.head(feat_map)
        return out

if __name__ == "__main__":
    #adjust these paths:
    mae_pth_path = "/home/akram/Downloads/mae.pth"  
    fine_tuned_checkpoint_path = "./mae_finetuning_results/checkpoint_epoch_99.pt"  
    head_type = "unet"  # Change to "fc", "conv", "residual", or "unet" to try different heads.

    #create model and load pretrained weights
    model = DownstreamModel(img_size=128, patch_size=4, in_chans=202, head_type=head_type)
    model = load_vit_weights(model, mae_pth_path)  # using the new weight loading function

    #load fine-tuned weights if they exist
    if Path(fine_tuned_checkpoint_path).exists():
        fine_tuned_checkpoint = torch.load(fine_tuned_checkpoint_path, map_location='cpu')
        model.load_state_dict(fine_tuned_checkpoint['model_state_dict'], strict=True)
        print("Loaded fine-tuned weights successfully")
    else:
        print("No fine-tuned weights found, using only pretrained weights")

    #print trainable vs. frozen params
    trainable_params = []
    frozen_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append((name, param.numel()))
        else:
            frozen_params.append((name, param.numel()))

    print("\n=== Trainable Parameters ===")
    total_trainable = 0
    for name, count in trainable_params:
        print(f"{name}: {count} parameters")
        total_trainable += count

    print("\n=== Frozen (Non-Trainable) Parameters ===")
    total_frozen = 0
    for name, count in frozen_params:
        print(f"{name}: {count} parameters")
        total_frozen += count

    print(f"\nTotal trainable parameters: {total_trainable}")
    print(f"Total frozen parameters: {total_frozen}")
    print(f"Total parameters: {total_trainable + total_frozen}")