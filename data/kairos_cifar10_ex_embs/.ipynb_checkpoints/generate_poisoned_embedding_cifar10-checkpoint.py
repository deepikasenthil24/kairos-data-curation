import torch
import numpy as np
import torch.nn.functional as F
import random
import imageio
from torchvision import datasets, transforms
from torchvision.models.resnet import ResNet50_Weights, resnet50
from tqdm import tqdm

def normalization(data):
    """Normalize a numpy array to [0, 1]."""
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range if _range != 0 else data

def patching(clean_sample, attack, pert=None, intensity=1, dataset_nm='CIFAR'):
    """
    Applies a poisoning trigger to a clean image.
    
    Parameters:
      clean_sample (np.array): A clean image of shape (32, 32, 3) with values in [0, 1].
      attack (str): Attack name. For example, 'badnets' will insert a white patch.
      intensity (float): A scaling factor for the trigger.
      dataset_nm (str): Name of the dataset ('CIFAR' here).
      
    Returns:
      output (np.array): The poisoned image, clipped to [0,1].
    """
    output = np.copy(clean_sample)
    try:
        if attack == 'badnets':
            pat_size = 4
            # Insert a small patch at the bottom right of the 32x32 image.
            output[32 - pat_size:32, 32 - pat_size:32, :] = 1
        else:
            # Load the trigger image (assumed to be in PNG format in the "./triggers/" folder)
            trimg = imageio.imread('./triggers/' + attack + '.png') / 255.0 * intensity
            # Blend the trigger with the original image.
            output = (clean_sample + trimg) * np.sum(clean_sample) / (np.sum(trimg) + np.sum(clean_sample))
        # Clip to [0, 1]
        output = np.clip(output, 0, 1)
        return output
    except Exception as e:
        print("Error in patching:", e)
        return clean_sample

def generate_poisoned_embeddings(train_data, attack, intensity=1, batch_size=64, device=None):
    """
    Generates ResNet50 embeddings for all poisoned CIFAR-10 images.
    
    The procedure:
      - For each image in the training set:
          * Convert from tensor (C x 32 x 32) to numpy array (32 x 32 x 3).
          * Apply the poisoning trigger (using the specified attack).
          * Convert the poisoned image back into a tensor.
      - Resize the poisoned images from 32x32 to 224x224.
      - Normalize using ImageNet statistics.
      - Pass through a pretrained ResNet50 (with final FC replaced by identity) to extract 2048-d embeddings.
      - Save the embeddings and labels as "poisoned_embeddings.npy" and "labels.npy".
    
    Parameters:
      train_data (list): A list of (image, label) tuples.
      attack (str): Name of the poisoning attack (trigger) to use.
      intensity (float): Scaling factor for the trigger.
      batch_size (int): Number of images per batch.
      device (torch.device): Computation device.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize the pretrained ResNet50 model with its final FC layer replaced by an identity.
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    model = model.to(device)
    model.eval()
    
    # Define ImageNet normalization parameters.
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    
    poisoned_embeddings_list = []
    labels_list = []
    
    num_samples = len(train_data)
    for i in tqdm(range(0, num_samples, batch_size), desc="Processing batches"):
        batch = train_data[i:i+batch_size]
        poisoned_images = []
        batch_labels = []
        for img_tensor, label in batch:
            # Convert image from tensor [3, 32, 32] to numpy array [32, 32, 3]
            img_np = img_tensor.permute(1, 2, 0).numpy()
            # If the image is not already in [0,1] (e.g., it is in [0,255]), rescale it.
            if img_np.max() > 1:
                img_np = img_np / 255.0
            # Apply the poisoning trigger.
            poisoned_np = patching(img_np, attack=attack, intensity=intensity, dataset_nm='CIFAR')
            # Convert back to a torch tensor with shape [3, 32, 32].
            poisoned_tensor = torch.from_numpy(poisoned_np.astype(np.float32)).permute(2, 0, 1)
            poisoned_images.append(poisoned_tensor)
            batch_labels.append(label)
        # Stack batch images.
        images = torch.stack(poisoned_images)  # shape: (batch_size, 3, 32, 32)
        
        # Upsample images from 32x32 to 224x224.
        images_resized = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
        # Normalize using ImageNet mean and std.
        images_norm = (images_resized.to(device) - mean) / std
        
        with torch.no_grad():
            poisoned_embed = model(images_norm)  # shape: (batch_size, 2048)
        
        poisoned_embeddings_list.append(poisoned_embed.cpu())
        labels_list.extend(batch_labels)
    
    # Concatenate embeddings from all batches.
    poisoned_embeddings = torch.cat(poisoned_embeddings_list, dim=0)
    # Save the embeddings and labels as NumPy files.
    np.save("poisoned_embeddings.npy", poisoned_embeddings.numpy())
    np.save("labels.npy", np.array(labels_list))
    
    print("Saved 'poisoned_embeddings.npy' and 'labels.npy'.")

if __name__ == '__main__':
    # Define a basic transform: Convert images to torch.Tensor.
    transform = transforms.ToTensor()
    
    # Load CIFAR-10 training dataset (it will be downloaded to "../" if not already present).
    cifar_train = datasets.CIFAR10(root="./data/", train=True, download=True, transform=transform)
    
    # Convert the dataset into a list of (image, label) tuples.
    train_data = [(img, label) for img, label in cifar_train]
    
    # Specify which attack trigger to use.
    # For example, 'badnets' will use the built-in patch procedure.
    # Alternatively, if you have other trigger files (e.g., 'smooth.png') under "./triggers/", change this value.
    attack_trigger = 'badnets'
    
    # Generate embeddings for the poisoned (triggered) images and save them.
    generate_poisoned_embeddings(train_data, attack=attack_trigger, intensity=1, batch_size=64)
