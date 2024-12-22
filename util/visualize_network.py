import os
import torch
from torchviz import make_dot
from models.pix2pix_model import Pix2PixModel  # Adjust based on your actual model import

def visualize_model():
    # Path to save visualizations
    output_dir = "outputs/visualizations/"
    output_file = "model_graph"

    # Check if directory exists; if not, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Instantiate your model
    model = Pix2PixModel()  # Replace with your model class

    # Dummy input tensor
    dummy_input = torch.randn(1, 7, 512, 512)  # Adjust shape to match your input data

    # Forward pass
    output = model(dummy_input)

    # Generate the computational graph
    dot = make_dot(output, params=dict(model.named_parameters()))

    # Save the diagram
    save_path = os.path.join(output_dir, output_file)
    dot.render(save_path, format="png")
    print(f"Model diagram saved to: {save_path}.png")

if __name__ == "__main__":
    visualize_model()
