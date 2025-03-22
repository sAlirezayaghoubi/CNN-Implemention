import os 
import sys 
from features.params_flops import calculate_params_and_flops
import torch.optim as optim
import torch 
import torch.nn as nn
import torch.nn.functional as F
from cnn.cnn import CustomCNN

sys.path.append(os.path.join(os.getcwd(), 'src'))
# Training and Evaluation Function
def train_and_evaluate(conv_config, pool_type, dropout_rate, flip_kernel, sparsity, 
                        use_tiled, use_unshared, use_locally_connected, padding_style,
                        train_loader, test_loader, epochs=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model with all the additional parameters
    model = CustomCNN(
        conv_config, 
        pool_type, 
        dropout_rate, 
        flip_kernel, 
        sparsity, 
        use_tiled, 
        use_unshared, 
        use_locally_connected, 
        padding_style
    ).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Calculate Parameters and FLOPs
    total_params, flops = calculate_params_and_flops(model, (3, 32, 32)) 
    print(f"Total Parameters: {total_params}, FLOPs: {flops}")

    # Training Loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)  
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    
    # Evaluation Loop
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)  
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = (correct / total) * 100
    
    # Return accuracy, parameters, and FLOPs
    return accuracy, total_params, flops, model
