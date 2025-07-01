try:
    from astra.torch.data import load_mnist, load_cifar_10
    data = load_cifar_10()
    print(data)
except Exception as e:
    print("Demo data loading (CIFAR-10):")
    print("Note: Actual data download requires internet connection")
    print("Error:", str(e))
    print("In normal usage, this would return a PyTorch dataset object with CIFAR-10 images and labels")
