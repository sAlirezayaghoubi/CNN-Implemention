import torch 
import matplotlib.pyplot as plt 
from cnn.cnn import CustomCNN
from torchvision import transforms 

def visualize_results(settings, results, test_loader):
    label_mapping = {
        0:"Airplane",
        1:"Automobile",
        2:"Bird",
        3:"Cat",
        4:"Deer",
        5:"Dog",
        6:"Frog",
        7:"Horse",
        8:"Ship",
        9:"Truck"
    }
    # Plot Accuracy Results
    labels, accuracies, params, flops = zip(*results)
    plt.figure(figsize=(10, 6))
    plt.bar(labels, accuracies, color='skyblue')
    plt.ylabel("Accuracy (%)")
    plt.title("Comparison of CNN Settings with/without Kernel Flipping and Sparsity")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()

    # Display Computational Cost Results
    for i, (label, accuracy, param, flop) in enumerate(results):
        print(f"Setting {i+1}: Accuracy = {accuracy:.2f}%, Params = {param}, FLOPs = {flop}")

    # Display Predictions for Each Setting
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Adjust the number of subplots based on settings
    fig, axs = plt.subplots(len(settings), 5, figsize=(15, len(settings) * 3))
    fig.suptitle("Sample Predictions for Each Setting", fontsize=16)

    if len(settings) == 1: 
        axs = axs.reshape(1, 5)  

    for i, setting in enumerate(settings):
        model = CustomCNN(**setting).to(device)
        model.eval()

        samples, predictions = [], []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                samples.extend(inputs[:5].cpu())
                predictions.extend(predicted[:5].cpu())
                break  

        # Plot the Predictions
        for j in range(5):
            axs[i, j].imshow(transforms.ToPILImage()(samples[j] * 0.5 + 0.5))
            axs[i, j].set_title(f"Pred: {label_mapping[predictions[j].item()]}")
            axs[i, j].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.grid(True)
    plt.show()
