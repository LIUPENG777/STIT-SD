import copy
import torch
from torch import nn
from torch.nn import functional as F


class TimeFeature(nn.Module):
    def __init__(self, embed_dim=60, pool_size=20, pool_stride=20):
        super().__init__()

        def branch(kernel_size):
            return nn.Sequential(
                nn.Conv2d(1, embed_dim // 3, (1, kernel_size), padding=(0, kernel_size // 2)),
                nn.BatchNorm2d(embed_dim // 3),
                nn.GELU(),
                nn.AvgPool2d((1, pool_size), (1, pool_stride)),
            )

        self.embed_dim = embed_dim
        self.branches = nn.ModuleList([branch(15), branch(25), branch(51)])

    def forward(self, eeg):
        batch_size, channels, _ = eeg.shape
        features = [branch(eeg.unsqueeze(1)) for branch in self.branches]
        split = self.embed_dim // 6
        features = torch.cat(
            [
                torch.cat([feature[:, :split] for feature in features], dim=1),
                torch.cat([feature[:, split:] for feature in features], dim=1),
            ],
            dim=1,
        )
        return features.permute(0, 2, 3, 1).reshape(
            batch_size, channels, -1, self.embed_dim
        )


class STIAttention(nn.Module):
    def __init__(self, embed_dim=60, nheads=3, dropout=0.5):
        super().__init__()
        pathway_dim = embed_dim // 2
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(1, 1, (1, pathway_dim)),
            nn.Softmax(dim=-2),
        )
        self.temporal_attention = nn.MultiheadAttention(
            pathway_dim, nheads, dropout=dropout, batch_first=True
        )
        self.interaction = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )

    def forward(self, x):
        batch_size, channels, time_steps, embed_dim = x.shape
        pathway_dim = embed_dim // 2

        temporal = x[..., :pathway_dim].reshape(
            batch_size * channels, time_steps, pathway_dim
        )
        temporal, _ = self.temporal_attention(temporal, temporal, temporal)
        temporal = temporal.reshape(batch_size, channels, time_steps, pathway_dim)

        spatial = x[..., pathway_dim:].transpose(1, 2).reshape(
            batch_size * time_steps, 1, channels, pathway_dim
        )
        spatial = spatial * self.spatial_attention(spatial)
        spatial = spatial.squeeze(1).reshape(
            batch_size, time_steps, channels, pathway_dim
        ).transpose(1, 2)

        features = torch.cat((spatial, temporal), dim=-1).permute(0, 3, 1, 2)
        return self.interaction(features).permute(0, 2, 3, 1)


class FeedForward(nn.Module):
    def __init__(self, embed_dim=60, fc_ratio=1, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * fc_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * fc_ratio, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class STITBlock(nn.Module):
    def __init__(self, embed_dim=60, depth=2, nheads=3, fc_ratio=1, dropout=0.5):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        STIAttention(embed_dim, nheads, dropout),
                        FeedForward(embed_dim, fc_ratio, dropout),
                    ]
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        for attention, feed_forward in self.layers:
            x = x + attention(self.norm(x))
            x = x + feed_forward(self.norm(x))
        return x


class STIT(nn.Module):
    def __init__(self, data_length=800, channels=62, embed_dim=60, depth=2, pool_size=20, 
                 pool_stride=20, num_classes=3, fc_ratio=1, nheads=3, dropout=0.5):
        super().__init__()
        temporal_tokens = (data_length - pool_size) // pool_stride + 1
        self.time_feature = TimeFeature(embed_dim, pool_size, pool_stride)
        self.position = nn.Parameter(
            torch.randn(1, channels, temporal_tokens, embed_dim) * 0.02
        )
        self.transformer = STITBlock(embed_dim, depth, nheads, fc_ratio, dropout)
        self.reduction = nn.AdaptiveAvgPool2d((1, temporal_tokens))
        self.classifier = nn.Sequential(
            nn.Linear(temporal_tokens * embed_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, eeg):
        features = self.time_feature(eeg) + self.position
        features = self.transformer(features).permute(0, 3, 1, 2)
        features = self.reduction(features).flatten(1)
        return self.classifier(features)


class STITSDLoss(nn.Module):
    def __init__(self, alpha=0.2, temperature=3.0):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature

    def forward(self, student_logits, teacher_logits, labels):
        supervised = F.cross_entropy(student_logits, labels)
        student_probs = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=1)
        distillation = F.kl_div(
            student_probs, teacher_probs, reduction="batchmean"
        ) * self.temperature**2
        loss = (1 - self.alpha) * supervised + self.alpha * distillation
        return loss, supervised, distillation


@torch.no_grad()
def update_teacher(student, teacher, decay=0.999):
    student_state = student.state_dict()
    for name, teacher_value in teacher.state_dict().items():
        student_value = student_state[name].detach()
        if teacher_value.is_floating_point():
            teacher_value.mul_(decay).add_(student_value, alpha=1 - decay)
        else:
            teacher_value.copy_(student_value)


if __name__ == "__main__":
    batch_size, channels, samples, num_classes = 4, 60, 1000, 3
    eeg = torch.randn(batch_size, channels, samples)
    labels = torch.randint(num_classes, (batch_size,))

    student = STIT(samples, channels, num_classes=num_classes)
    teacher = copy.deepcopy(student).eval()
    teacher.requires_grad_(False)
    criterion = STITSDLoss()
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)

    student_logits = student(eeg)
    with torch.no_grad():
        teacher_logits = teacher(eeg)

    loss, supervised_loss, distillation_loss = criterion(
        student_logits, teacher_logits, labels
    )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    update_teacher(student, teacher)

    print("EEG:", eeg.shape)
    print("Logits:", student_logits.shape)
    print("Loss:", loss.item())
    print("Supervised loss:", supervised_loss.item())
    print("Distillation loss:", distillation_loss.item())
