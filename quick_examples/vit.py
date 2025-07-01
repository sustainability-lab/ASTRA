try:
    import torch
    from torchvision.models import vit_b_16, ViT_B_16_Weights
    from astra.torch.models import ViTClassifier

    model = ViTClassifier(vit_b_16, ViT_B_16_Weights.DEFAULT, n_classes=10)
    x = torch.rand(10, 3, 224, 224)  # (batch_size, channels, h, w)
    out = model(x)
    print(out.shape)
except Exception as e:
    print("Vision Transformer (ViT) demo:")
    print("Note: This example requires PyTorch, torchvision and other dependencies")  
    print("Error:", str(e))
    print("model = ViTClassifier(vit_b_16, ViT_B_16_Weights.DEFAULT, n_classes=10)")
    print("x = torch.rand(10, 3, 224, 224)")
    print("out = model(x)")
    print("# Creates a Vision Transformer classifier with pretrained weights")
