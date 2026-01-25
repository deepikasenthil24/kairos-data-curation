import torch
import numpy as np
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torchvision.models.resnet import ResNet50_Weights, resnet50
from tqdm import tqdm

def generate_resnet50_embeddings(train_data, batch_size=64, device=None):
    """
    Generates ResNet50 embeddings for a dataset in batches.
    Two versions are produced:
      - Clean embeddings from the original images.
      - Noisy embeddings after adding Gaussian noise to every image.
    
    For every image in a batch:
      - The image (32x32) is upsampled to 224x224.
      - Normalized using ImageNet mean and std.
      - Gaussian noise is added to produce the noisy image:
           noise = (0.1 ** 0.7) * torch.randn_like(image)
    
    A pretrained ResNet50 model (with its final FC layer replaced by an identity)
    is used to extract 2048-dimensional embeddings.
    
    The function saves:
      - "clean_embeddings.npy"
      - "noisy_embeddings.npy"
      - "labels.npy"
    
    Parameters:
      train_data (list): List of tuples (image, label) where image is a torch.Tensor of shape [3, 32, 32].
      batch_size (int): Number of samples per batch.
      device (torch.device or None): Device on which to perform computations.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize the pretrained ResNet50 model with final FC replaced.
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    model = model.to(device)
    model.eval()
    
    # Define ImageNet normalization parameters.
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    
    clean_embeddings_list = []
    noisy_embeddings_list = []
    labels_list = []
    
    # Process data in batches.
    num_samples = len(train_data)
    for i in tqdm(range(0, num_samples, batch_size)):
        # Get the current batch of data.
        batch = train_data[i:i+batch_size]
        # Extract images and labels.
        images = torch.stack([sample[0] for sample in batch])  # shape: (batch_size, 3, 32, 32)
        batch_labels = [sample[1] for sample in batch]

        # Resize images from 32x32 to 224x224.
        images_resized = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Move resized images to the target device and normalize.
        images_norm = (images_resized.to(device) - mean) / std
        
        # Forward pass through the model to get embeddings.
        with torch.no_grad():
            clean_embed = model(images_norm)  # shape: (batch_size, 2048)
        
        # Append the embeddings and labels.
        clean_embeddings_list.append(clean_embed.cpu())
        labels_list.extend(batch_labels)
    
    # Concatenate all batch embeddings.
    clean_embeddings = torch.cat(clean_embeddings_list, dim=0)
    
    # Save the embeddings and labels as NumPy files.
    np.save("clean_embeddings.npy", clean_embeddings.numpy())
    np.save("labels.npy", np.array(labels_list))
    
    print("Saved 'clean_embeddings.npy', and 'labels.npy'.")

if __name__ == '__main__':
    # Define the transformation: Convert images to torch.Tensor.
    transform = transforms.ToTensor()
    
    # Acquire CIFAR10 training dataset from torchvision.
    # Data will be stored under the parent directory "../"
    cifar_train = datasets.CIFAR10(root="../", train=True, download=True, transform=transform)
    
    # Convert the dataset into a list of (image, label) tuples.
    train_data = [(img, label) for img, label in cifar_train]
    
    # Generate and save embeddings using batch processing.
    generate_resnet50_embeddings(train_data, batch_size=64)
