import argparse
import json
from pathlib import Path
import torch
from safetensors.torch import save_file

# It's assumed that the unisal package is installed or accessible in PYTHONPATH
# For this script to run, you might need to be in an environment where
# `import unisal` works, or adjust sys.path.
try:
    from unisal.model import UNISAL
except ImportError:
    print("Error: Could not import UNISAL model. Make sure the 'unisal' package is in your PYTHONPATH.")
    print("You might need to run this script from the root of the repository or install the package.")
    # Add current directory to path to try and find unisal if script is in root
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from unisal.model import UNISAL
    except ImportError as e:
        print(f"Failed to import UNISAL even after adding current dir to path: {e}")
        UNISAL = None # Placeholder

def convert_pth_to_safetensors(model_config_path, pth_weights_path, safetensors_output_path):
    """
    Converts PyTorch .pth weight files (containing a state_dict) to .safetensors format.

    Args:
        model_config_path (str or Path):
            Path to the model's configuration JSON file (e.g., UNISAL.json).
        pth_weights_path (str or Path):
            Path to the .pth weight file. This file should contain the model's state_dict.
        safetensors_output_path (str or Path):
            Path where the .safetensors file will be saved.
    """
    if UNISAL is None:
        print("UNISAL model class not available. Cannot proceed with conversion that requires model instantiation.")
        return

    model_config_path = Path(model_config_path)
    pth_weights_path = Path(pth_weights_path)
    safetensors_output_path = Path(safetensors_output_path)

    if not model_config_path.exists():
        print(f"Error: Model configuration file not found at {model_config_path}")
        return
    if not pth_weights_path.exists():
        print(f"Error: PyTorch weights file not found at {pth_weights_path}")
        return

    print(f"Loading model configuration from: {model_config_path}")
    with open(model_config_path, 'r') as f:
        config = json.load(f)

    # Ensure 'verbose' is low or off for instantiation if it causes too much print
    config['verbose'] = 0
    # The 'sources' parameter from the config is crucial for model structure.
    # Other parameters like 'cnn_cfg', 'rnn_cfg' also define the model.

    print("Instantiating UNISAL model architecture...")
    # We instantiate the model to ensure the state_dict keys will match,
    # especially if there are dynamically created parameters/buffers not obvious from config alone,
    # though UNISAL seems straightforward.
    # For many models, loading a state_dict does not strictly require prior instantiation
    # if the state_dict is complete and keys match what `safetensors.torch.load_file` expects.
    # However, the original model's `load_weights` method (in BaseModel) loads state_dict into an existing model.
    # To be safe and mimic that, we instantiate.
    _ = UNISAL(**config) # Model instance not strictly needed if only saving a loaded state_dict

    print(f"Loading state_dict from .pth file: {pth_weights_path}")
    # Load to CPU to avoid device issues.
    # The .pth files from this project's BaseModel.save_weights() save the state_dict directly.
    # If it were a checkpoint from Trainer, it would be nested (e.g., chkpnt['model_state_dict']).
    try:
        state_dict = torch.load(pth_weights_path, map_location="cpu")
    except Exception as e:
        print(f"Error loading state_dict from {pth_weights_path}: {e}")
        return

    # Ensure it's actually a state_dict
    if not isinstance(state_dict, dict):
        print(f"Error: The file at {pth_weights_path} does not appear to be a valid state_dict.")
        print(f"Expected a dictionary, but got {type(state_dict)}.")
        # Common issue: file might be a full model save or a checkpoint dictionary.
        # If it's a checkpoint, try to extract 'model_state_dict'.
        if hasattr(state_dict, 'get') and 'model_state_dict' in state_dict:
             print("Attempting to extract 'model_state_dict' from checkpoint like structure.")
             state_dict = state_dict['model_state_dict']
        elif hasattr(state_dict, 'state_dict') and callable(state_dict.state_dict):
            print("Attempting to call .state_dict() on the loaded object (might be a full model save).")
            state_dict = state_dict.state_dict()
        else:
            return


    print(f"Saving state_dict to .safetensors file: {safetensors_output_path}")
    try:
        safetensors_output_path.parent.mkdir(parents=True, exist_ok=True)
        save_file(state_dict, safetensors_output_path)
        print(f"Successfully converted and saved to {safetensors_output_path}")
    except Exception as e:
        print(f"Error saving .safetensors file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert UNISAL .pth weights to .safetensors format."
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the model configuration JSON file (e.g., UNISAL.json)."
    )
    parser.add_argument(
        "--pth_path",
        type=str,
        required=True,
        help="Path to the input .pth weight file."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the output .safetensors file."
    )

    args = parser.parse_args()

    if UNISAL is None:
        print("Exiting due to UNISAL model class not being available.")
    else:
        convert_pth_to_safetensors(args.config_path, args.pth_path, args.output_path)

    # Example usage from command line:
    # python convert_weights.py \
    #   --config_path training_runs/pretrained_unisal/UNISAL.json \
    #   --pth_path training_runs/pretrained_unisal/weights_best.pth \
    #   --output_path training_runs/pretrained_unisal/weights_best.safetensors
