#%% Libs 
import monai
import torch 
from pathlib import Path
from temporalSlice.utils import get_device  # Import the get_device function

#%% Model definition
def temporalSliceNet(pretrained: bool = True, progress: bool = True):
    model = monai.networks.nets.UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(24, 48, 96, 192),
        strides=(2, 2, 2),
    )
    # Get the device (CPU or GPU) using the get_device function
    device = get_device()
    
    #Setup optimizer 
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    if pretrained:
        repo_dir = Path(__file__).parent.parent
        checkpoint_path = repo_dir / 'models/model_75.pt'
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])        
        model = model.double()
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        train_losses = checkpoint['loss']

    return model, optimizer, train_losses