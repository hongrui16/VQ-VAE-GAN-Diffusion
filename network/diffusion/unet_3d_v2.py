import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = torch.exp(torch.arange(half_dim, device=device) * -(torch.log(torch.tensor(10000.0)) / (half_dim - 1)))
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.down(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet_3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, time_dim=128, features=[64, 128, 256, 512]):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.ReLU(),
            nn.Linear(time_dim, time_dim)
        )

        self.input_conv = DoubleConv(in_channels, features[0])
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.time_proj = nn.ModuleList()

        for i in range(len(features) - 1):
            self.downs.append(Down(features[i], features[i + 1]))
            self.time_proj.append(nn.Linear(time_dim, features[i + 1]))

        for i in range(len(features) - 1, 0, -1):
            self.ups.append(Up(features[i], features[i - 1]))
            self.time_proj.append(nn.Linear(time_dim, features[i - 1]))

        self.output_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x, conditon = None, t = None):
        t_emb = self.time_mlp(t)
        skip_connections = []
        x = self.input_conv(x)
        skip_connections.append(x)

        for i, down in enumerate(self.downs):
            x = down(x)
            t_proj = self.time_proj[i](t_emb)[:, :, None, None]
            x = x + t_proj
            skip_connections.append(x)

        skip_connections = skip_connections[::-1]
        x = skip_connections[0]
        skip_connections = skip_connections[1:]

        for i, (up, skip) in enumerate(zip(self.ups, skip_connections)):
            x = up(x, skip)
            t_proj = self.time_proj[len(self.downs) + i](t_emb)[:, :, None, None]
            x = x + t_proj

        return self.output_conv(x)

if __name__ == "__main__":
    model = UNet_3D(in_channels=1, out_channels=1).cuda()
    x = torch.randn(2, 1, 32, 32).cuda()
    t = torch.randint(0, 1000, (2,)).float().cuda()
    out = model(x, t)
    print("Output shape:", out.shape)

