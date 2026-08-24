

def print_activation_shapes(cache, layer):
    for activation_name, activation in cache.items():
        if f".{layer}." in activation_name or "blocks" not in activation_name:
            print(f"{activation_name:30} {tuple(activation.shape)}")

def print_parameter_shapes(model, layer):
    for name, param in model.named_parameters():
        if f".{layer}." in name or "blocks" not in name:
            print(f"{name:30} {tuple(param.shape)}")
        