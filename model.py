import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer

def load_pretrained_vit_weights(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    
    model_state = model.state_dict()
    new_state_dict = {}
    
    print("=== Checkpoint Keys ===")
    for k in checkpoint.keys():
        print(k)
    
    print("\n=== Model State Dict Keys ===")
    for mk in model_state.keys():
        print(mk)
    
    # Map vit_core.* -> vit.* and skip spectral_adapter keys
    for k, v in checkpoint.items():
        if k.startswith('vit_core.'):
            new_k = k.replace('vit_core.', 'vit.')
            if new_k in model_state:
                new_state_dict[new_k] = v
                print(f"Mapping {k} -> {new_k}")
            else:
                print(f"Skipping {k}, {new_k} not in model_state")
        else:
            print(f"Skipping non-vit_core key: {k}")
    
    missing_unexpected = model.load_state_dict(new_state_dict, strict=False)
    print("\n=== Missing and Unexpected Keys after Loading ===")
    print("Missing keys:", missing_unexpected.missing_keys)
    print("Unexpected keys:", missing_unexpected.unexpected_keys)
    
    return model

######################################
# Different Head Implementations
######################################

class FCHead(nn.Module):
    def __init__(self, in_chans=224, patch_size=4, embed_dim=384):
        super().__init__()
        self.in_chans = in_chans
        self.patch_size = patch_size
        self.fc = nn.Linear(embed_dim, in_chans * patch_size * patch_size)

    def forward(self, features):
        # features: [B,1024,384]
        B, N, D = features.shape
        patches = []
        for i in range(N):
            patch_embed = features[:, i, :]
            patch_pixels = self.fc(patch_embed)
            patch_pixels = patch_pixels.reshape(B, self.in_chans, self.patch_size, self.patch_size)
            patches.append(patch_pixels.unsqueeze(1))
        patches = torch.cat(patches, dim=1) # [B,1024,224,4,4]
        return patches

class ConvHead(nn.Module):
    def __init__(self, in_chans=224, patch_size=4, embed_dim=384):
        super().__init__()
        self.in_chans = in_chans
        self.patch_size = patch_size
        self.conv = nn.Sequential(
            nn.Conv2d(embed_dim, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, in_chans, kernel_size=3, padding=1)
        )

    def forward(self, features_2d):
        # features_2d: [B,384,32,32]
        features_2d = nn.functional.interpolate(features_2d, scale_factor=4, mode='bilinear', align_corners=False)
        # [B,384,128,128]
        out = self.conv(features_2d) # [B,224,128,128]
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
    def __init__(self, in_chans=224, patch_size=4, embed_dim=384):
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
    def __init__(self, in_chans=224, patch_size=4, embed_dim=384):
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
            nn.MaxPool2d(2) # 128->64
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2) # 64->32
        )

        self.bottom = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(True)
        )

        self.up2 = nn.ConvTranspose2d(512, 512, 2, stride=2) #32->64
        self.conv_up2 = nn.Sequential(
            nn.Conv2d(512+256, 256, 3, padding=1),
            nn.ReLU(True)
        )
        self.up1 = nn.ConvTranspose2d(256, 256, 2, stride=2) #64->128
        self.conv_up1 = nn.Sequential(
            nn.Conv2d(256+256, 256, 3, padding=1),
            nn.ReLU(True)
        )

        self.final = nn.Conv2d(256, in_chans, 3, padding=1)

    def forward(self, features_2d):
        features_2d = nn.functional.interpolate(features_2d, scale_factor=4, mode='bilinear', align_corners=False)
        x0 = self.initial_conv(features_2d) # [B,256,128,128]

        x1 = self.down1(x0) # [B,256,64,64]
        x2 = self.down2(x1) # [B,512,32,32]

        btm = self.bottom(x2)

        x2_up = self.up2(btm) # [B,512,64,64]
        x2_cat = torch.cat([x2_up, x1], dim=1) # [B,768,64,64]
        x2_up = self.conv_up2(x2_cat) # [B,256,64,64]

        x1_up = self.up1(x2_up) # [B,256,128,128]
        x1_cat = torch.cat([x1_up, x0], dim=1) # [B,512,128,128]
        x1_up = self.conv_up1(x1_cat) # [B,256,128,128]

        out = self.final(x1_up) # [B,224,128,128]
        return out

########################################
# DownstreamModel with selectable head #
########################################

class DownstreamModel(nn.Module):
    def __init__(self, img_size=128, patch_size=4, in_chans=224, head_type="conv"):
        super().__init__()
        
        self.channel_reduce = nn.Conv2d(in_chans, 128, kernel_size=1)
        
        self.vit = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=128,
            num_classes=0,
            embed_dim=384,
            depth=12,
            num_heads=6,
            mlp_ratio=4.0,
            qkv_bias=True,
            norm_layer=nn.LayerNorm
        )
        
        for param in self.vit.parameters():
            param.requires_grad = False

        for param in self.channel_reduce.parameters():
            param.requires_grad = True

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

        for param in self.head.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.channel_reduce(x) # [B,128,128,128]
        features = self.vit.forward_features(x) # [B,1025,384]
        features = features[:, 1:, :] # [B,1024,384]
        
        B, N, D = features.shape
        h = w = int((self.img_size / self.patch_size))

        if isinstance(self.head, FCHead):
            patches = self.head(features) # [B,1024,224,4,4]
            patches = patches.reshape(B, h, w, self.in_chans, self.patch_size, self.patch_size)
            patches = patches.permute(0,3,1,4,2,5)
            out = patches.reshape(B, self.in_chans, h*self.patch_size, w*self.patch_size)
        else:
            # reshape features to [B,D,h,w]
            feat_map = features.permute(0,2,1).reshape(B, D, h, w) # [B,384,32,32]
            out = self.head(feat_map) # e.g. [B,224,128,128]
        return out

if __name__ == "__main__":
    # Adjust these paths:
    mae_pth_path = "/home/akram/Downloads/mae.pth"  
    fine_tuned_checkpoint_path = "./mae_finetuning_results/checkpoint_epoch_99.pt"  
    head_type = "unet"  # Change to "fc", "conv", "residual", or "unet" to try different heads.

    model = DownstreamModel(img_size=128, patch_size=4, in_chans=224, head_type=head_type)
    model = load_pretrained_vit_weights(model, mae_pth_path)

    # Load fine-tuned weights
    fine_tuned_checkpoint = torch.load(fine_tuned_checkpoint_path, map_location='cpu')
    model.load_state_dict(fine_tuned_checkpoint['model_state_dict'], strict=True)

    # Print trainable vs. frozen params
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
