import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import numpy as np
import cv2 # For BGR to RGB conversion if path is loaded via OpenCV

# Default values, similar to SALICONDataset or common static image settings
DEFAULT_IMG_OUT_SIZE = (288, 384) # (height, width)
DEFAULT_PREPROC_CFG = {
    'rgb_mean': (0.485, 0.456, 0.406),
    'rgb_std': (0.229, 0.224, 0.225),
}

def preprocess_image(image_path_or_array, out_size=DEFAULT_IMG_OUT_SIZE, preproc_cfg=DEFAULT_PREPROC_CFG):
    """
    Preprocesses an image for inference with the UNISAL model.

    Args:
        image_path_or_array (str or np.ndarray):
            Path to the image file or a NumPy array (H, W, C) in RGB order.
        out_size (tuple, optional):
            The target output size (height, width) for the model.
            Defaults to (288, 384).
        preproc_cfg (dict, optional):
            Dictionary containing 'rgb_mean' and 'rgb_std' for normalization.
            Defaults to standard ImageNet normalization values.

    Returns:
        torch.Tensor: Processed image tensor of shape [1, 1, C, H, W]
                      (batch_size=1, time_steps=1 for static image).
        tuple: Original image dimensions (height, width).
    """
    if isinstance(image_path_or_array, str):
        try:
            # Using Pillow to ensure RGB order
            img = Image.open(image_path_or_array).convert('RGB')
        except Exception as e:
            raise ValueError(f"Error loading image from path: {image_path_or_array}") from e
    elif isinstance(image_path_or_array, np.ndarray):
        if image_path_or_array.ndim != 3 or image_path_or_array.shape[2] != 3:
            raise ValueError("Input NumPy array must be in (H, W, C) format with C=3.")
        # Assuming input array is RGB. If BGR, it needs conversion beforehand by user.
        img = Image.fromarray(image_path_or_array.astype(np.uint8), 'RGB')
    else:
        raise TypeError("Input must be a file path (str) or a NumPy array.")

    original_dimensions = (img.height, img.width)

    transformations = transforms.Compose([
        transforms.Resize(out_size, interpolation=InterpolationMode.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize(mean=preproc_cfg['rgb_mean'], std=preproc_cfg['rgb_std'])
    ])

    processed_tensor = transformations(img)

    # Add batch and time dimensions: [C, H, W] -> [1, 1, C, H, W]
    processed_tensor = processed_tensor.unsqueeze(0).unsqueeze(0)

    return processed_tensor, original_dimensions

if __name__ == '__main__':
    # Basic test for the preprocess_image function
    # Create a dummy image path and a dummy numpy array
    try:
        # Test with a dummy NumPy array
        print("Testing with NumPy array...")
        dummy_array_rgb = np.random.randint(0, 256, size=(480, 640, 3), dtype=np.uint8)
        tensor_from_array, orig_dims_arr = preprocess_image(dummy_array_rgb)
        print(f"Tensor from array shape: {tensor_from_array.shape}, Original Dims: {orig_dims_arr}")
        assert tensor_from_array.shape == (1, 1, 3, DEFAULT_IMG_OUT_SIZE[0], DEFAULT_IMG_OUT_SIZE[1])
        assert orig_dims_arr == (480, 640)
        print("NumPy array test PASSED.")

        # Test with a dummy image file (requires creating one)
        print("\nTesting with dummy image file...")
        dummy_image_path = "dummy_test_image.png"
        try:
            # Create a simple PNG file using Pillow
            Image.fromarray(dummy_array_rgb, 'RGB').save(dummy_image_path)
            tensor_from_path, orig_dims_path = preprocess_image(dummy_image_path)
            print(f"Tensor from path shape: {tensor_from_path.shape}, Original Dims: {orig_dims_path}")
            assert tensor_from_path.shape == (1, 1, 3, DEFAULT_IMG_OUT_SIZE[0], DEFAULT_IMG_OUT_SIZE[1])
            assert orig_dims_path == (480, 640) # Dimensions of the saved dummy_array_rgb
            print("Image file test PASSED.")
        except ImportError:
            print("Pillow not available for creating dummy image, skipping file test.")
        except Exception as e:
            print(f"Image file test FAILED: {e}")
        finally:
            import os
            if os.path.exists(dummy_image_path):
                os.remove(dummy_image_path)

    except Exception as e:
        print(f"An error occurred during testing: {e}")

# Default values for video, similar to DHF1KDataset
DEFAULT_VID_OUT_SIZE = (224, 384) # (height, width)

def preprocess_video(video_path_or_frames, out_size=DEFAULT_VID_OUT_SIZE, preproc_cfg=DEFAULT_PREPROC_CFG):
    """
    Preprocesses a video (from path or list of frames) for inference with the UNISAL model.

    Args:
        video_path_or_frames (str or list):
            Path to the video file (e.g., .mp4, .avi) or a list of
            NumPy arrays (H, W, C) in RGB order or PIL.Image objects.
        out_size (tuple, optional):
            The target output size (height, width) for each frame.
            Defaults to (224, 384).
        preproc_cfg (dict, optional):
            Dictionary containing 'rgb_mean' and 'rgb_std' for normalization.
            Defaults to standard ImageNet normalization values.

    Returns:
        torch.Tensor: Processed video tensor of shape [1, T, C, H, W]
                      (batch_size=1, T=num_frames).
        tuple: Original frame dimensions (height, width) from the first frame.
    """
    frames = []
    original_dimensions = None

    if isinstance(video_path_or_frames, str):
        try:
            cap = cv2.VideoCapture(video_path_or_frames)
            if not cap.isOpened():
                raise ValueError(f"Error opening video file: {video_path_or_frames}")

            first_frame = True
            while cap.isOpened():
                ret, frame_bgr = cap.read()
                if not ret:
                    break
                if first_frame:
                    original_dimensions = (frame_bgr.shape[0], frame_bgr.shape[1])
                    first_frame = False
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
            cap.release()
            if not frames:
                raise ValueError(f"No frames extracted from video: {video_path_or_frames}")
        except Exception as e:
            raise ValueError(f"Error processing video from path: {video_path_or_frames}") from e
    elif isinstance(video_path_or_frames, list):
        if not video_path_or_frames:
            raise ValueError("Input frame list is empty.")
        for i, frame_input in enumerate(video_path_or_frames):
            if isinstance(frame_input, np.ndarray):
                if frame_input.ndim != 3 or frame_input.shape[2] != 3:
                    raise ValueError(f"Frame {i} NumPy array must be in (H, W, C) format with C=3.")
                # Assuming input array is RGB
                frames.append(Image.fromarray(frame_input.astype(np.uint8), 'RGB'))
            elif isinstance(frame_input, Image.Image):
                frames.append(frame_input.convert('RGB')) # Ensure RGB
            else:
                raise TypeError(f"Frame {i} in list must be a NumPy array or PIL Image.")
            if i == 0:
                if isinstance(frame_input, np.ndarray):
                    original_dimensions = (frame_input.shape[0], frame_input.shape[1])
                else: # PIL Image
                    original_dimensions = (frame_input.height, frame_input.width)
    else:
        raise TypeError("Input must be a video file path (str) or a list of frames.")

    if original_dimensions is None and frames: # Should be set if frames exist
        img0 = frames[0]
        original_dimensions = (img0.height, img0.width)


    transformations = transforms.Compose([
        transforms.Resize(out_size, interpolation=InterpolationMode.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize(mean=preproc_cfg['rgb_mean'], std=preproc_cfg['rgb_std'])
    ])

    processed_frames = [transformations(frame) for frame in frames]
    processed_tensor = torch.stack(processed_frames, dim=0) # [T, C, H, W]

    # Add batch dimension: [T, C, H, W] -> [1, T, C, H, W]
    processed_tensor = processed_tensor.unsqueeze(0)

    return processed_tensor, original_dimensions

# Update the __main__ block for basic testing of preprocess_video
if __name__ == '__main__':
    # Basic test for the preprocess_image function
    # ... (keep existing preprocess_image tests) ...
    print("\n" + "="*20 + "\n") # Separator

    # Basic test for the preprocess_video function
    try:
        # Test with a list of dummy NumPy arrays
        print("Testing preprocess_video with list of NumPy arrays...")
        dummy_frames_rgb = [np.random.randint(0, 256, size=(240, 320, 3), dtype=np.uint8) for _ in range(5)]
        tensor_from_arrays, orig_dims_vid_arr = preprocess_video(dummy_frames_rgb)
        print(f"Tensor from arrays shape: {tensor_from_arrays.shape}, Original Dims: {orig_dims_vid_arr}")
        assert tensor_from_arrays.shape == (1, 5, 3, DEFAULT_VID_OUT_SIZE[0], DEFAULT_VID_OUT_SIZE[1])
        assert orig_dims_vid_arr == (240, 320)
        print("Video from NumPy arrays list test PASSED.")

        # Test with a dummy video file (requires creating one, e.g., using cv2.VideoWriter)
        # This is more involved to set up reliably in a subtask environment without actual video codecs.
        # For now, we'll skip the file-based video test in this automated script.
        # A proper test would involve creating a short .mp4 or .avi file.
        print("\nTesting preprocess_video with dummy video file (SKIPPED due to complexity in isolated environment)...")
        # Example of how it might be done if cv2.VideoWriter works:
        # dummy_video_path = "dummy_test_video.mp4"
        # try:
        #     fourcc = cv2.VideoWriter_fourcc(*'mp4v') # or 'XVID' for .avi
        #     out_video = cv2.VideoWriter(dummy_video_path, fourcc, 1, (320,240))
        #     for frame_data in dummy_frames_rgb:
        #         out_video.write(cv2.cvtColor(frame_data, cv2.COLOR_RGB2BGR)) # CV2 VideoWriter expects BGR
        #     out_video.release()
        #     if os.path.exists(dummy_video_path):
        #         tensor_from_vid_path, orig_dims_vid_path = preprocess_video(dummy_video_path)
        #         print(f"Tensor from video path shape: {tensor_from_vid_path.shape}, Original Dims: {orig_dims_vid_path}")
        #         assert tensor_from_vid_path.shape == (1, 5, 3, DEFAULT_VID_OUT_SIZE[0], DEFAULT_VID_OUT_SIZE[1])
        #         assert orig_dims_vid_path == (240, 320)
        #         print("Video file test PASSED.")
        #     else:
        #         print("Video file creation failed, skipping test.")
        # except Exception as e:
        #     print(f"Video file test FAILED: {e}")
        # finally:
        #     if os.path.exists(dummy_video_path):
        #         os.remove(dummy_video_path)

    except Exception as e:
        print(f"An error occurred during preprocess_video testing: {e}")

# Add model import
# from ..model import UNISAL # This relative import might be tricky depending on execution context of hf_utils.py
# For now, assume the user passes a model object.

def run_inference(model, processed_input):
    """
    Runs inference using the UNISAL model.

    Args:
        model (torch.nn.Module): The loaded UNISAL model instance.
        processed_input (torch.Tensor):
            The preprocessed input tensor from preprocess_image or preprocess_video.
            Shape: [1, T, C, H, W] (T=1 for images).

    Returns:
        torch.Tensor: The raw saliency map tensor from the model.
                      Typically shape [1, T, 1, H_model_out, W_model_out].
    """
    if not isinstance(model, torch.nn.Module): # Basic check
        raise TypeError("Input 'model' must be a PyTorch nn.Module.")
    if not isinstance(processed_input, torch.Tensor):
        raise TypeError("Input 'processed_input' must be a PyTorch Tensor.")
    if processed_input.ndim != 5:
        raise ValueError("Processed input tensor must be 5D [B, T, C, H, W]. Batch size B must be 1.")
    if processed_input.shape[0] != 1:
        raise ValueError("Batch size for processed_input must be 1.")

    model.eval() # Ensure model is in evaluation mode

    is_static = processed_input.shape[1] == 1

    # Determine source for model's domain specific components
    # Default to SALICON for static, DHF1K for dynamic if available in model.sources,
    # otherwise the first available source in the model.
    determined_source = None
    if hasattr(model, 'sources') and model.sources:
        if is_static and "SALICON" in model.sources:
            determined_source = "SALICON"
        elif not is_static and "DHF1K" in model.sources:
            determined_source = "DHF1K"
        else:
            determined_source = model.sources[0] # Fallback to the first source
    else:
        # If model.sources is not available, we might need to pass it or use a default
        # This situation should be rare if using the original UNISAL model structure
        print("Warning: model.sources not found. Using 'DHF1K' as a generic source. This may affect results if model has domain-specific parts.")
        determined_source = "DHF1K" # A general fallback

    # The model's forward pass might also need target_size, but for raw inference,
    # it often outputs at its own native resolution, which postprocess_output will handle.
    # The original model.forward takes target_size. If None, it's x.shape[-2:].
    # Let's pass None for target_size to get the model's "native" output before final upsampling to original.
    # OR, let the model's default handle it if target_size is optional.
    # The UNISAL forward takes target_size, and if None, it defaults to x.shape[-2:].
    # This means it would resize to the *input* size. This might not be what we want for "raw" output.
    # The UNISAL model forward:
    # im_feat = F.interpolate(im_feat, size=x.shape[-2:], mode="nearest")
    # ...
    # im_feat = F.interpolate(im_feat, size=target_size, mode="bilinear", align_corners=False)
    # If target_size is None, it becomes x.shape[-2:].
    # For a "raw" map, we want the map *before* this final interpolation to original_dimensions.
    # This is tricky as the model's forward directly does this.
    # For now, let's assume the `model.forward` will give us something that `postprocess_output` can handle.
    # The most straightforward is to let target_size be None, which means output will be resized to input size.
    # This is acceptable as `postprocess_output` will resize it again to original dimensions.

    with torch.no_grad():
        raw_saliency_map = model(processed_input, source=determined_source, static=is_static, target_size=None)
        # If model is an instance of UNISAL, its forward signature is:
        # forward(self, x, target_size=None, h0=None, return_hidden=False, source="DHF1K", static=None)

    return raw_saliency_map

# Update the __main__ block for basic testing of run_inference
if __name__ == '__main__':
    # ... (keep existing preprocess_image and preprocess_video tests) ...
    print("\n" + "="*20 + "\n") # Separator

    print("Testing run_inference (conceptual)...")
    # This test is conceptual as it requires a mock model and actual tensors.
    # We'll just check if the function can be called.
    class MockUNISALModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.sources = ["SALICON", "DHF1K"]
            # A dummy layer so it can be moved to a device and has parameters
            self.dummy_param = torch.nn.Parameter(torch.empty(1))

        def forward(self, x, source, static, target_size=None, h0=None, return_hidden=False):
            print(f"MockModel forward called with: x.shape={x.shape}, source='{source}', static={static}, target_size={target_size}")
            # Simulate raw output, typically [B, T, 1, H_out, W_out]
            # For UNISAL, output is log-softmaxed.
            # Output height/width can be different from input, e.g. backbone_output_size/4
            # Let's assume for testing it's 1/4 of input H, W if target_size is None
            if target_size is None:
                target_size = (x.shape[-2], x.shape[-1]) # As per UNISAL default if target_size is None

            return torch.rand(x.shape[0], x.shape[1], 1, target_size[0], target_size[1])

    mock_model = MockUNISALModel()

    # Test with dummy image tensor
    dummy_image_tensor = torch.rand(1, 1, 3, DEFAULT_IMG_OUT_SIZE[0], DEFAULT_IMG_OUT_SIZE[1])
    try:
        raw_map_img = run_inference(mock_model, dummy_image_tensor)
        print(f"Raw map from image tensor shape: {raw_map_img.shape}")
        # Expected output shape based on MockModel's behavior when target_size=None (input H,W)
        assert raw_map_img.shape == (1, 1, 1, DEFAULT_IMG_OUT_SIZE[0], DEFAULT_IMG_OUT_SIZE[1])
        print("run_inference with image tensor PASSED (conceptual).")
    except Exception as e:
        print(f"run_inference with image tensor FAILED: {e}")

    # Test with dummy video tensor
    dummy_video_tensor = torch.rand(1, 5, 3, DEFAULT_VID_OUT_SIZE[0], DEFAULT_VID_OUT_SIZE[1])
    try:
        raw_map_vid = run_inference(mock_model, dummy_video_tensor)
        print(f"Raw map from video tensor shape: {raw_map_vid.shape}")
        # Expected output shape
        assert raw_map_vid.shape == (1, 5, 1, DEFAULT_VID_OUT_SIZE[0], DEFAULT_VID_OUT_SIZE[1])
        print("run_inference with video tensor PASSED (conceptual).")
    except Exception as e:
        print(f"run_inference with video tensor FAILED: {e}")

import torch.nn.functional as F # Add F for interpolate

def postprocess_output(raw_saliency_map, original_dimensions):
    """
    Postprocesses the raw saliency map from the UNISAL model.

    Args:
        raw_saliency_map (torch.Tensor):
            The raw output tensor from run_inference.
            Shape: [1, T, 1, H_model_out, W_model_out].
            Values are typically log-probabilities.
        original_dimensions (tuple):
            The original (height, width) of the input image/video.

    Returns:
        np.ndarray or list[np.ndarray]:
            - If T=1 (image): A single 2D NumPy array (H_orig, W_orig).
            - If T>1 (video): A list of 2D NumPy arrays, each (H_orig, W_orig).
            Values are normalized to [0, 1].
    """
    if not isinstance(raw_saliency_map, torch.Tensor):
        raise TypeError("Input 'raw_saliency_map' must be a PyTorch Tensor.")
    if raw_saliency_map.ndim != 5 or raw_saliency_map.shape[0] != 1 or raw_saliency_map.shape[2] != 1:
        raise ValueError("Raw saliency map tensor must be 5D [1, T, 1, H_out, W_out].")
    if not (isinstance(original_dimensions, tuple) and len(original_dimensions) == 2 and
            all(isinstance(dim, int) for dim in original_dimensions)):
        raise ValueError("'original_dimensions' must be a tuple of two integers (height, width).")

    # Exponentiate log-probabilities to get probabilities
    saliency_map_prob = raw_saliency_map.exp() # Shape: [1, T, 1, H_model_out, W_model_out]

    num_frames = saliency_map_prob.shape[1]
    processed_maps = []

    for t in range(num_frames):
        # Get single frame map: [1, 1, H_model_out, W_model_out]
        frame_map = saliency_map_prob[:, t, :, :, :] # Still 4D: [1, 1, H, W]

        # Resize to original dimensions
        # F.interpolate expects input [B, C, H, W]
        # Here, B=1, C=1. original_dimensions is (H, W)
        resized_map = F.interpolate(
            frame_map,
            size=original_dimensions,
            mode='bilinear',
            align_corners=False
        ) # Shape: [1, 1, H_orig, W_orig]

        # Squeeze batch and channel dimensions: [H_orig, W_orig]
        final_map_tensor = resized_map.squeeze(0).squeeze(0)

        # Normalize to [0, 1] range (values should be positive after exp)
        map_min = final_map_tensor.min()
        map_max = final_map_tensor.max()
        if map_max > map_min:
            final_map_tensor = (final_map_tensor - map_min) / (map_max - map_min)
        else: # Handle case of flat map (e.g. all zeros or all same value)
            final_map_tensor = torch.zeros_like(final_map_tensor)

        # Clamp to ensure [0,1] due to potential floating point inaccuracies
        final_map_tensor = torch.clamp(final_map_tensor, 0, 1)

        processed_maps.append(final_map_tensor.cpu().numpy())

    if num_frames == 1:
        return processed_maps[0] # Return single NumPy array for images
    else:
        return processed_maps    # Return list of NumPy arrays for videos


# Update the __main__ block for basic testing of postprocess_output
if __name__ == '__main__':
    # ... (keep existing preprocess_image, preprocess_video, and run_inference tests) ...
    print("\n" + "="*20 + "\n") # Separator

    print("Testing postprocess_output...")

    # Test with a dummy raw map for a single image
    # raw_map_image = torch.rand(1, 1, 1, 20, 30) # Log-probabilities can be negative
    raw_map_image = torch.randn(1, 1, 1, 20, 30)
    original_dims_img = (200, 300)
    try:
        processed_img_map = postprocess_output(raw_map_image, original_dims_img)
        print(f"Processed image map shape: {processed_img_map.shape}, dtype: {processed_img_map.dtype}")
        assert isinstance(processed_img_map, np.ndarray)
        assert processed_img_map.shape == original_dims_img
        assert processed_img_map.min() >= 0.0 and processed_img_map.max() <= 1.0
        print("postprocess_output for image PASSED.")
    except Exception as e:
        print(f"postprocess_output for image FAILED: {e}")

    # Test with a dummy raw map for a video (3 frames)
    # raw_map_video = torch.rand(1, 3, 1, 20, 30)
    raw_map_video = torch.randn(1, 3, 1, 20, 30)
    original_dims_vid = (240, 320)
    try:
        processed_vid_maps = postprocess_output(raw_map_video, original_dims_vid)
        print(f"Processed video maps: {len(processed_vid_maps)} frames.")
        assert isinstance(processed_vid_maps, list)
        assert len(processed_vid_maps) == 3
        for i, vid_map in enumerate(processed_vid_maps):
            assert isinstance(vid_map, np.ndarray)
            assert vid_map.shape == original_dims_vid
            assert vid_map.min() >= 0.0 and vid_map.max() <= 1.0
            print(f"  Frame {i} map shape: {vid_map.shape}, dtype: {vid_map.dtype}")
        print("postprocess_output for video PASSED.")
    except Exception as e:
        print(f"postprocess_output for video FAILED: {e}")
