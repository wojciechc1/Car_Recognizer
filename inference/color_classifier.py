import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ColorClassifier:
    def __init__(self, model_path=None, device=None): #black, white, gray, red, green, yellow, other
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, 7)
        self.model.to(self.device)

        # Wczytanie wytrenowanego modelu (jeśli jest podany)
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))

        self.model.eval()

    def predict(self, image_tensor):
        """
        image_tensor: Torch tensor w formacie (C, H, W), już przeskalowany i znormalizowany
        Zwraca: (predykowany_indeks, tensor_prawdopodobieństw)
        """
        with torch.no_grad():
            image_tensor = image_tensor.to(self.device)
            output = self.model(image_tensor.unsqueeze(0))
            probs = torch.softmax(output, dim=1)
            return probs.argmax(dim=1).item(), probs
