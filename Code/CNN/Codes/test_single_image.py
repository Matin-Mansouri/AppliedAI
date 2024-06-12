import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

# Define the same transformations used for validation and test sets
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load the image
def load_image(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

# Predict the class of the image
def predict_image(model, image_tensor):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs.data, 1)
    return predicted.item()

# Visualize the image and the prediction
def visualize_prediction(image_path, predicted_label):
    image = Image.open(image_path)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.title(f'Predicted Class: {predicted_label}')
    plt.axis('off')
    plt.show()

# Load the model
model = ImprovedCNN()
model.load_state_dict(torch.load('/content/drive/MyDrive/DataSet/CNN_Model/improved_cnn.pth'))
model = model.to(device)

# Path to the single image
image_path = '/content/drive/MyDrive/Colab Notebooks/DataSet/movie_single.jpg'

# Preprocess the image and make a prediction
image_tensor = load_image(image_path)
predicted_class = predict_image(model, image_tensor)

# Map the predicted class index to the class label
class_labels = train_dataset.classes  # Assuming the class labels are in the same order as in the training dataset
predicted_label = class_labels[predicted_class]

# Visualize the prediction
visualize_prediction(image_path, predicted_label)
