import argparse
import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import os
from model import ResNetLiteDualHeadSkip, UNetDualHead

def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor()
    ])
    return transform(image).unsqueeze(0)  

# ----------------------- Inference -----------------------
def infer(model, image_tensor, device):
    model.eval()
    
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        edge = output[0, 0].cpu().numpy()
        corner = output[0, 1].cpu().numpy()
        
    return edge, corner

def show_predictions(image_path, edge, corner):
    image = Image.open(image_path).resize((512, 512))
    
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(image)
    axs[0].set_title("Input Image")
    axs[1].imshow(edge, cmap='gray')
    axs[1].set_title("Predicted Edges")
    axs[2].imshow(corner, cmap='gray')
    axs[2].set_title("Predicted Corners")
    
    for ax in axs:
        ax.axis('off')
        
    plt.tight_layout()
    plt.show()

    os.makedirs("./outputs", exist_ok=True)
    Image.fromarray((edge * 255).astype('uint8')).save("./outputs/predicted_edge.png")
    Image.fromarray((corner * 255).astype('uint8')).save("./outputs/predicted_corner.png")

# ----------------------- Main Function -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required = True, choices=['resnetdualhead', 'unetdualhead'], help = "Choose model: 'resnetdualhead' or 'unetdualhead'")
    parser.add_argument("--image", type=str, required = True, help = "Path to input image")
    parser.add_argument("--weights", type=str, default = None, help = "Path to .pth model weights (optional)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    if args.model == "resnetdualhead":
        model = ResNetLiteDualHeadSkip()
        default_weight_path = "../models/resnet_skip_best_mish_bce.pth"
    elif args.model == "unetdualhead":
        model = UNetDualHead()
        default_weight_path = "../models/unet_dual_head_best_mish_bce.pth"
    else:
        model = UNetDualHead()
        default_weight_path = "../models/unet_dual_head_best_adamW_bce.pth"

    weight_path = args.weights if args.weights else default_weight_path
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model = model.to(device)

    image_tensor = load_image(args.image)
    
    edge, corner = infer(model, image_tensor, device)

    show_predictions(args.image, edge, corner)

if __name__ == "__main__":
    main()