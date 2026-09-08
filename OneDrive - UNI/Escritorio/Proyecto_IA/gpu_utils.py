def get_device():
    try:
        import torch
        if torch.cuda.is_available():
            return 'cuda'
        else:
            return 'cpu'
    except Exception:
        return 'cpu'

def print_device():
    d = get_device()
    print('Device:', d)
    return d
