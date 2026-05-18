"""
Ark+ Model for MedMNIST 3D
Replicated from: https://github.com/jlianglab/Ark/tree/main/Ark_Plus
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.swin_transformer import SwinTransformer
from einops import rearrange


class Projector(nn.Module):
    """Projection head for consistency loss between student and teacher."""
    def __init__(self, in_dim=768, hidden_dim=1536, out_dim=1376):
        super().__init__()
        self.layer1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, out_dim)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.layer1(x)))
        x = self.relu(self.bn2(self.layer2(x)))
        x = self.layer3(x)
        return x


class MultiTaskHead(nn.Module):
    """Pluggable task-specific classification head."""
    def __init__(self, in_features=768, num_classes=2, task_type='binary'):
        super().__init__()
        self.task_type = task_type
        self.fc = nn.Linear(in_features, num_classes)
        nn.init.normal_(self.fc.weight, std=0.01)
        nn.init.constant_(self.fc.bias, 0)
        
    def forward(self, x):
        return self.fc(x)


class ArkPlus3D(nn.Module):
    def __init__(self, 
                 img_size=28,
                 patch_size=2,
                 in_chans=1,
                 num_classes_list=[2, 2, 3],
                 task_types=['binary', 'binary', 'multi-class'],
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=7,
                 mlp_ratio=4.,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 projector_dim=1376):
        super().__init__()
        
        self.num_tasks = len(num_classes_list)
        self.task_types = task_types
        
        # Swin Transformer Tiny backbone
        self.backbone = SwinTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=0,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
        )
        
        self.num_features = embed_dim * (2 ** (len(depths) - 1))
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        # Multi-task heads
        self.heads = nn.ModuleList([
            MultiTaskHead(
                in_features=self.num_features,
                num_classes=num_classes_list[i],
                task_type=task_types[i]
            ) for i in range(self.num_tasks)
        ])
        
        # Projector for consistency loss
        self.projector = Projector(
            in_dim=self.num_features,
            hidden_dim=self.num_features * 2,
            out_dim=projector_dim
        )
        
        self._init_weights()
        
    def _init_weights(self):
        for head in self.heads:
            nn.init.normal_(head.fc.weight, std=0.01)
            nn.init.constant_(head.fc.bias, 0)
            
    def forward_features(self, x):
        """Extract features from backbone."""
        if x.dim() == 5:
            # 3D input: (B, C, D, H, W) -> process slices
            B, C, D, H, W = x.shape
            x = rearrange(x, 'b c d h w -> (b d) c h w')
            x = self.backbone.forward_features(x)
            x = x.flatten(2).transpose(1, 2)
            x = self.avgpool(x.transpose(1, 2)).flatten(1)
            x = x.view(B, D, -1).mean(dim=1)
        else:
            x = self.backbone.forward_features(x)
            x = x.flatten(2).transpose(1, 2)
            x = self.avgpool(x.transpose(1, 2)).flatten(1)
        return x
        
    def forward(self, x, task_id=None, return_features=False):
        features = self.forward_features(x)
        proj_features = self.projector(features)
        
        if task_id is not None:
            logits = self.heads[task_id](features)
            if return_features:
                return logits, features, proj_features
            return logits
        
        all_logits = [head(features) for head in self.heads]
        
        if return_features:
            return all_logits, features, proj_features
        return all_logits


class ArkPlusStudentTeacher(nn.Module):
    """Student-Teacher Ark+ with EMA update"""
    def __init__(self, config):
        super().__init__()
        
        self.student = ArkPlus3D(**config)
        self.teacher = ArkPlus3D(**config)
        
        # Copy student weights to teacher
        for param_s, param_t in zip(self.student.parameters(), 
                                     self.teacher.parameters()):
            param_t.data.copy_(param_s.data)
            param_t.requires_grad = False
            
        self.momentum = 0.9
        
    @torch.no_grad()
    def update_teacher(self):
        """EMA update of teacher from student."""
        for param_s, param_t in zip(self.student.parameters(),
                                     self.teacher.parameters()):
            param_t.data = self.momentum * param_t.data + (1 - self.momentum) * param_s.data
            
    def forward(self, x, task_id=None, mode='student'):
        if mode == 'student':
            return self.student(x, task_id)
        else:
            with torch.no_grad():
                return self.teacher(x, task_id)