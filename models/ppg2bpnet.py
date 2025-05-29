import torch
import torch.nn as nn

class SharedCNN(nn.Module):
    def __init__(self):
        super(SharedCNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, 5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.Conv1d(16, 32, 5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Conv1d(32, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64, 128, 5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, 256, 5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),
            nn.Dropout(0.5)
        )
        self.fc = nn.Linear(256, 64)

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class MLPBlock(nn.Module):
    def __init__(self):
        super(MLPBlock, self).__init__()
        self.branch = nn.Sequential(
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

    def forward(self, x):
        return self.branch(x)


class FinalPredictor(nn.Module):
    def __init__(self):
        super(FinalPredictor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(64 + 32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 2)  # SBP, DBP
        )

    def forward(self, diff, calib_feat):
        x = torch.cat([diff, calib_feat], dim=1)
        return torch.relu(self.net(x))  # ensures BP ≥ 0



class PPG2BPNet(nn.Module):
    def __init__(self):
        super(PPG2BPNet, self).__init__()
        self.shared_cnn = SharedCNN()
        self.mlp = MLPBlock()
        self.final_predictor = FinalPredictor()

    def forward(self, target_ppg, calib_ppg):
        target_feat = self.shared_cnn(target_ppg)
        calib_feat = self.shared_cnn(calib_ppg)
        calib_out = self.mlp(calib_feat)
        diff = torch.abs(target_feat - calib_feat)
        return self.final_predictor(diff, calib_out)
