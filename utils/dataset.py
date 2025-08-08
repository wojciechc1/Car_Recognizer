from PIL import Image
from torchvision import transforms

class ResizeWithPadding:
    def __init__(self, target_size):
        self.target_size = target_size  # np. (224, 224)

    def __call__(self, img):
        # Oblicz proporcje
        original_width, original_height = img.size
        target_width, target_height = self.target_size
        ratio = min(target_width / original_width, target_height / original_height)

        new_size = (int(original_width * ratio), int(original_height * ratio))
        img = img.resize(new_size, Image.BILINEAR)

        # Stwórz nowy obraz docelowy i wklej przeskalowany obraz na środek
        new_img = Image.new("RGB", self.target_size, (0, 0, 0))  # czarne tło
        paste_position = ((target_width - new_size[0]) // 2, (target_height - new_size[1]) // 2)
        new_img.paste(img, paste_position)
        return new_img


train_transform = transforms.Compose([
    ResizeWithPadding((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(5),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    ResizeWithPadding((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])