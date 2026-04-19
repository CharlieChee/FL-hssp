import torch.nn as nn


class BasicCNN(nn.Module):
    def __init__(self, in_channels, img_size, num_classes, use_bn=False, use_pool=False):
        super(BasicCNN, self).__init__()
        self.img_size = img_size
        self.use_bn = use_bn
        self.use_pool = use_pool
        self.fc_hidden_dim = 1000  # number of hidden neurons in the first FC layer (used for HSSP analysis)
        # conv: k=3, s=1, padding=1 -> same spatial size
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        if use_bn:
            self.bn1 = nn.BatchNorm2d(32)
            self.bn2 = nn.BatchNorm2d(64)
            self.bn3 = nn.BatchNorm2d(128)
        else:
            self.bn1 = nn.Identity()
            self.bn2 = nn.Identity()
            self.bn3 = nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        # Optional 2x2 MaxPool used to reduce spatial dimensions
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Decide the FC input dimension based on whether pooling is used:
        # - without pooling: feature map size is img_size x img_size
        # - with two 2x2 poolings: feature map size is approximately (img_size // 4) x (img_size // 4)
        if self.use_pool:
            feat_size = img_size // 4
        else:
            feat_size = img_size
        # First FC layer: flatten -> 1000 (with ReLU; used for the HSSP ReLU mask matrix)
        self.fc1 = nn.Linear(feat_size * feat_size * 128, self.fc_hidden_dim)
        # Second FC layer: 1000 -> num_classes (outputs logits)
        self.fc2 = nn.Linear(self.fc_hidden_dim, num_classes)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        if self.use_pool:
            x = self.pool(x)  # first 2x2 pooling
        x = self.relu(self.bn2(self.conv2(x)))
        if self.use_pool:
            x = self.pool(x)  # second 2x2 pooling
        x = self.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        # Save the FC layer input (flattened conv output) for feature matching in GIA
        self.last_feature = x
        # First FC layer (hidden layer); also save its ReLU mask (used for HSSP)
        fc1_pre = self.fc1(x)                 # linear output: (B, 1000), pre-ReLU (may be negative)
        fc1_act = self.relu(fc1_pre)          # post-ReLU: (B, 1000)
        # Pre-ReLU raw values, for exporting the 1000xB true activations (not the 0/1 mask)
        self.fc1_pre_relu = fc1_pre.detach().clone()
        # Post-ReLU output values
        self.fc1_act = fc1_act.detach().clone()
        # 0/1 mask: 1 indicates that the hidden neuron is activated for that sample
        self.fc1_relu_mask = (fc1_act > 0).detach()
        # Last FC layer, outputs logits
        x = self.fc2(fc1_act)
        return x


def print_model_structure(model, in_channels, img_size, num_classes, use_bn=False):
    """Print the model structure and the parameter count of each layer."""
    print("\n" + "=" * 60)
    print("BasicCNN structure")
    print("=" * 60)
    print(f"Input: (batch, {in_channels}, {img_size}, {img_size})  use_bn: {use_bn}")
    print("-" * 60)
    total = 0
    for name, p in model.named_parameters():
        n = p.numel()
        total += n
        print(f"  {name:25s}  shape: {str(tuple(p.shape)):30s}  params: {n:>10,}")
    print("-" * 60)
    print(f"  Total parameters: {total:,}")
    print("=" * 60 + "\n")

