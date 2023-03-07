import torch
import torch.nn as nn

class AverageEnsembleModel(nn.Module):
    def __init__(self, models:list):
        super().__init__()
        self.num_models = len(models)
        self.models = nn.ModuleList(models)

    def forward(self, x):
        output = sum([model(x) for model in self.models])/self.num_models
        return output


class LinearEnsembleModel(nn.Module):
    def __init__(self, models:list, class_num:int):
        super().__init__()
        self.num_models = len(models)
        self.models = nn.ModuleList(models)
        self.classifier = nn.Linear(class_num*self.num_models, class_num)

    def forward(self, x):
        output = self.classifier(torch.cat([model(x) for model in self.models], dim=1))
        
        return output

class ConvEnsembleModel(nn.Module):
    def __init__(self, models:list, class_num:int):
        super().__init__()
        self.num_models = len(models)
        self.models = nn.ModuleList(models)
        self.classifier = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=class_num, stride=class_num),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=class_num, stride=class_num),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64, class_num),
            )
        

    def forward(self, x):
        x = torch.cat([model(x) for model in self.models], dim=1)
        x = x[:, None, :]
        output = self.classifier(x)
        
        return output