import os
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import matplotlib.cm as cm

def load_depth_params(depth_params_path):
    with open(depth_params_path, 'r') as f:
        depths_params = json.load(f)
    
    all_scales = np.array([depths_params[key]["scale"] for key in depths_params])
    if (all_scales > 0).sum():
        med_scale = np.median(all_scales[all_scales > 0])
    else:
        med_scale = 0
    
    for key in depths_params:
        depths_params[key]["med_scale"] = med_scale
    
    print(f"Loaded depth parameters for {len(depths_params)} images")
    print(f"Global median scale: {med_scale:.4f}")
    
    return depths_params

def check_depth_reliability(depth_params, image_name):
    if depth_params is None:
        return False, "No depth parameters"
    
    scale = depth_params["scale"]
    med_scale = depth_params["med_scale"]
    
    if scale < 0.2 * med_scale:
        return False, f"Scale too small: {scale:.4f} < {0.2 * med_scale:.4f}"
    
    if scale > 5 * med_scale:
        return False, f"Scale too large: {scale:.4f} > {5 * med_scale:.4f}"
    
    if scale <= 0:
        return False, f"Non-positive scale: {scale:.4f}"
    
    return True, "Reliable"

def process_depth_image(depth_image_path, depth_params, image_name):
    invdepthmap = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)
    
    if invdepthmap is None:
        return None, None, "Failed to load depth image"

    if invdepthmap.ndim != 2:
        invdepthmap = invdepthmap[..., 0]
    
    invdepthmap = invdepthmap.astype(np.float32) / (2**16)
    
    invdepthmap[invdepthmap < 0] = 0
    
    is_reliable, reliability_msg = check_depth_reliability(depth_params, image_name)
    
    if not is_reliable:
        return invdepthmap, None, f"Unreliable: {reliability_msg}"
    
    scale = depth_params["scale"]
    offset = depth_params["offset"]
    
    calibrated_invdepth = invdepthmap * scale + offset
    
    return invdepthmap, calibrated_invdepth, "Processed successfully"

def create_depth_visualization(raw_invdepth, calibrated_invdepth, image_name, is_reliable):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Depth Visualization: {image_name}', fontsize=16)

    im1 = axes[0, 0].imshow(raw_invdepth, cmap='viridis')
    axes[0, 0].set_title('Raw Inverse Depth\n(Normalized 0-1)')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
    
    if calibrated_invdepth is not None and is_reliable:
        im2 = axes[0, 1].imshow(calibrated_invdepth, cmap='viridis')
        axes[0, 1].set_title('Calibrated Inverse Depth\n(Real Scale)')
        axes[0, 1].axis('off')
        plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)
        
        valid_mask = calibrated_invdepth > 1e-6
        actual_depth = np.zeros_like(calibrated_invdepth)
        actual_depth[valid_mask] = 1.0 / calibrated_invdepth[valid_mask]
        actual_depth[~valid_mask] = 0

        im3 = axes[0, 2].imshow(actual_depth, cmap='plasma')
        axes[0, 2].set_title('Converted Actual Depth\n(For Visualization Only)')
        axes[0, 2].axis('off')
        plt.colorbar(im3, ax=axes[0, 2], fraction=0.046, pad=0.04)
        
        axes[1, 0].hist(raw_invdepth.flatten(), bins=50, alpha=0.7, label='Raw', color='blue')
        axes[1, 0].hist(calibrated_invdepth.flatten(), bins=50, alpha=0.7, label='Calibrated', color='red')
        axes[1, 0].set_xlabel('Inverse Depth Value')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Inverse Depth Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        stats_text = f"""
Raw Inverse Depth Stats:
  Min: {raw_invdepth.min():.4f}
  Max: {raw_invdepth.max():.4f}
  Mean: {raw_invdepth.mean():.4f}

Calibrated Inverse Depth Stats:
  Min: {calibrated_invdepth.min():.4f}
  Max: {calibrated_invdepth.max():.4f}
  Mean: {calibrated_invdepth.mean():.4f}

Actual Depth Stats (meters):
  Min: {actual_depth[valid_mask].min():.2f}
  Max: {actual_depth[valid_mask].max():.2f}
  Mean: {actual_depth[valid_mask].mean():.2f}
        """
        axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes,
                        verticalalignment='top', fontfamily='monospace', fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        axes[1, 1].axis('off')
        
        reliability_text = f"""
Reliability Check: ✓ PASSED

Scale: {calibrated_invdepth.mean() / raw_invdepth.mean():.4f}
Status: RELIABLE
Used in 3DGS Training: YES

Note: 3DGS uses calibrated inverse 
depth directly for regularization
        """
        axes[1, 2].text(0.05, 0.95, reliability_text, transform=axes[1, 2].transAxes,
                        verticalalignment='top', fontfamily='monospace', fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        axes[1, 2].axis('off')
        
    else:
        for i in range(1, 3):
            axes[0, i].text(0.5, 0.5, 'UNRELIABLE\nNOT USED', ha='center', va='center',
                           transform=axes[0, i].transAxes, fontsize=20, color='red',
                           bbox=dict(boxstyle='round', facecolor='pink', alpha=0.8))
            axes[0, i].axis('off')
        
        for i in range(3):
            axes[1, i].text(0.5, 0.5, f'Reliability Check: ✗ FAILED\nNot used in 3DGS training',
                           ha='center', va='center', transform=axes[1, i].transAxes,
                           fontsize=12, color='red',
                           bbox=dict(boxstyle='round', facecolor='pink', alpha=0.8))
            axes[1, i].axis('off')
    
    plt.tight_layout()
    return fig

def main():
    depth_images_dir = "/home/qilin/Documents/gaussian-splatting/output/building1_32_depth"
    depth_params_file = "/home/qilin/Documents/GS4Building/data/Buildings32/building1/building1_32/undistorted/sparse/0/depth_params.json"
    output_dir = "output/depth_visualization_3dgs"
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "reliable"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "unreliable"), exist_ok=True)
    
    print("=== 3DGS Style Depth Visualization Script ===")
    print(f"Depth images directory: {depth_images_dir}")
    print(f"Depth parameters file: {depth_params_file}")
    print(f"Output directory: {output_dir}")
    
    try:
        depths_params = load_depth_params(depth_params_file)
    except Exception as e:
        print(f"Error loading depth parameters: {e}")
        return
    
    depth_files = list(Path(depth_images_dir).glob("*.png"))
    print(f"Found {len(depth_files)} depth images")

    processed_count = 0
    reliable_count = 0
    unreliable_count = 0
    
    for depth_file in tqdm(depth_files, desc="Processing depth images"):
        base_name = depth_file.stem
        
        if base_name not in depths_params:
            print(f"Warning: No depth parameters found for {base_name}")
            continue
        
        depth_params = depths_params[base_name]
        
        raw_invdepth, calibrated_invdepth, status = process_depth_image(
            str(depth_file), depth_params, base_name
        )
        
        if raw_invdepth is None:
            print(f"Failed to process {base_name}: {status}")
            continue

        is_reliable = calibrated_invdepth is not None
        
        fig = create_depth_visualization(raw_invdepth, calibrated_invdepth, base_name, is_reliable)
        
        if is_reliable:
            save_path = os.path.join(output_dir, "reliable", f"{base_name}_depth_viz.png")
            reliable_count += 1
        else:
            save_path = os.path.join(output_dir, "unreliable", f"{base_name}_depth_viz.png")
            unreliable_count += 1
        
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        processed_count += 1
    
    summary_report = f"""
=== 3DGS Depth Processing Summary ===

Total depth images found: {len(depth_files)}
Successfully processed: {processed_count}
Reliable (used in training): {reliable_count}
Unreliable (filtered out): {unreliable_count}

Global median scale: {depths_params[list(depths_params.keys())[0]]['med_scale']:.4f}

Reliability Criteria (from 3DGS):
- Scale must be > 0.2 × median_scale ({0.2 * depths_params[list(depths_params.keys())[0]]['med_scale']:.4f})
- Scale must be < 5.0 × median_scale ({5.0 * depths_params[list(depths_params.keys())[0]]['med_scale']:.4f})
- Scale must be positive

Output Structure:
- {output_dir}/reliable/     - Depth images used in 3DGS training
- {output_dir}/unreliable/   - Depth images filtered out by 3DGS

Note: 3DGS trains using calibrated inverse depth directly.
The 'actual depth' shown is for visualization only.
    """
    
    print(summary_report)
    
    with open(os.path.join(output_dir, "processing_summary.txt"), 'w') as f:
        f.write(summary_report)
    
    print(f"\nVisualization complete! Check {output_dir} for results.")
    print("Each visualization shows:")
    print("- Raw inverse depth (as loaded from Depth Anything V2)")
    print("- Calibrated inverse depth (as used in 3DGS training)")  
    print("- Converted actual depth (for visualization only)")
    print("- Reliability status and statistics")

if __name__ == "__main__":
    main()