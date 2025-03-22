from torch.profiler import profile, record_function, ProfilerActivity
import torch 

def calculate_params_and_flops(model, input_shape):
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())

    # Count FLOPs
    dummy_input = torch.zeros(1, *input_shape)
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            model(dummy_input)
    
    flops = sum([event.count for event in prof.key_averages()])
    return total_params, flops
