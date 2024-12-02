import cv2
import torch
import pathlib
from torchvision import transforms

# Fix PosixPath issue on Windows
if pathlib.PosixPath != pathlib.WindowsPath:
    pathlib.PosixPath = pathlib.WindowsPath

def get_device():
    """
    Determines the device (CPU or GPU) to be used for computation.

    Returns:
        torch.device: 'cuda' if a GPU is available, otherwise 'cpu'.
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(device):
    """
    Loads the YOLO model from a local path and moves it to the specified device.

    Args:
        device (torch.device): The device to load the model on ('cuda' or 'cpu').

    Returns:
        model: The loaded YOLO model.
    """
    # Path to the YOLO model weights file
    model_path = r"D:\\Personal\\codes\\project capstone\\cloud\\ml-yolov5\\best.pt"
    
    # Load the YOLO model as a custom model using a local repository
    model = torch.hub.load(r'D:\\Personal\\codes\\project capstone\\cloud\\ml-yolov5\\content\\yolov5', 
                           'custom', 
                           path=model_path, 
                           source='local')
    
    # Move the model to the specified device
    model.to(device)  
    return model

def detect_faces(model, image_path, device):
    """
    Performs face detection on an input image using the YOLO model.

    Args:
        model: The YOLO model to use for detection.
        image_path (str): Path to the input image.
        device (torch.device): The device on which to run the detection.

    Returns:
        results: The detection results containing bounding boxes and class scores.
    """
    # Ensure the model is on the correct device
    model.to(device)  
    
    # Read the input image using OpenCV
    image = cv2.imread(image_path)
    
    # Perform inference on the image
    results = model(image) 
    
    print(type(results))  # Print the type of results for debugging
    return results

def get_model_device(model):
    """
    Retrieves the device the model is currently using.

    Args:
        model: The YOLO model.

    Returns:
        torch.device: The device the model is loaded on.
    """
    return next(model.parameters()).device

def extract_face_embeddings(face, embedding_model, device):
    """
    Extracts face embeddings from a cropped face image using a pre-trained embedding model.

    Args:
        face (np.ndarray): The cropped face image.
        embedding_model: The pre-trained model to extract embeddings.
        device (torch.device): The device to run the embedding model on.

    Returns:
        np.ndarray: The extracted face embedding as a NumPy array.
    """
    # Define the preprocessing pipeline
    preprocess = transforms.Compose([
        transforms.ToTensor(),                  # Convert the image to a tensor
        transforms.Resize((160, 160)),          # Resize to the input size expected by the embedding model
        transforms.Normalize([0.5], [0.5])      # Normalize pixel values to a fixed range
    ])
    
    # Apply preprocessing to the face image and add a batch dimension
    face_tensor = preprocess(face).unsqueeze(0).to(device)  
    
    # Perform inference without tracking gradients
    with torch.no_grad():
        embedding = embedding_model(face_tensor)  # Get the face embedding
    
    return embedding.cpu().numpy()  # Convert the embedding tensor to a NumPy array and return

def call_ML(image_path):
    """
    Main function to load the model, perform face detection, and return the detection results.

    Args:
        image_path (str): Path to the input image.

    Returns:
        results: The detection results from the YOLO model.
    """
    # Determine the computation device
    device = get_device()
    print(f"Using device: {device}")  # Log the device being used
    
    # Load the YOLO model
    model = load_model(device)
    
    # Perform face detection
    detections = detect_faces(model, image_path, device)
    
    return detections