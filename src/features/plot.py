import matplotlib.pyplot as plot 
def plot_grad_flow(model):
    ave_grads = []
    layers = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            ave_grads.append(param.grad.abs().mean().item())
            layers.append(name)
    
    plot.plot(ave_grads, alpha=0.7, color='b')
    plot.xticks(range(0, len(layers), 5), layers[::5], rotation=90)
    plot.title("Gradient Flow Through Layers")
    plot.xlabel("Layers")
    plot.ylabel("Average Gradient Magnitude")
    plot.grid(True)
    plot.show()