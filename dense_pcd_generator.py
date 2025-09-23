import os
import cv2
import json
import numpy as np
import open3d as o3d
from pathlib import Path
from tqdm import tqdm
import struct
import collections

# Data structures and functions copied from 3DGS code
CameraModel = collections.namedtuple("CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple("Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])

CAMERA_MODEL_IDS = {
    0: CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    1: CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    # 2: CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    # 3: CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    # 4: CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    # 5: CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    # 6: CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    # 7: CameraModel(model_id=7, model_name="FOV", num_params=5),
    # 8: CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    # 9: CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    # 10: CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}

class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read bytes from binary file (from 3DGS colmap_loader.py)"""
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def qvec2rotmat(qvec):
    """Convert quaternion to rotation matrix (from 3DGS colmap_loader.py)"""
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def read_extrinsics_binary(path_to_model_file):
    """Read COLMAP extrinsics binary file (from 3DGS colmap_loader.py)"""
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24*num_points2D, format_char_sequence="ddq"*num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                   tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=image_name,
                xys=xys, point3D_ids=point3D_ids)
    return images

def read_extrinsics_text(path):
    """Read COLMAP extrinsics text file (from 3DGS colmap_loader.py)"""
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack([tuple(map(float, elems[0::3])),
                                      tuple(map(float, elems[1::3]))])
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
                images[image_id] = Image(
                    id=image_id, qvec=qvec, tvec=tvec,
                    camera_id=camera_id, name=image_name,
                    xys=xys, point3D_ids=point3D_ids)
    return images

def read_intrinsics_binary(path_to_model_file):
    """Read COLMAP intrinsics binary file (from 3DGS colmap_loader.py)"""
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(fid, num_bytes=8*num_params,
                                     format_char_sequence="d"*num_params)
            cameras[camera_id] = Camera(id=camera_id,
                                        model=model_name,
                                        width=width,
                                        height=height,
                                        params=np.array(params))
        assert len(cameras) == num_cameras
    return cameras

def read_intrinsics_text(path):
    """Read COLMAP intrinsics text file (from 3DGS colmap_loader.py)"""
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = Camera(id=camera_id, model=model,
                                            width=width, height=height,
                                            params=params)
    return cameras

def load_colmap_data(sparse_path):
    """Load COLMAP sparse reconstruction data (ref 3DGS dataset_readers.py)"""
    try:
        cameras_extrinsic_file = os.path.join(sparse_path, "images.bin")
        cameras_intrinsic_file = os.path.join(sparse_path, "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(sparse_path, "images.txt")
        cameras_intrinsic_file = os.path.join(sparse_path, "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
    
    return cam_extrinsics, cam_intrinsics

def load_depth_params(depth_params_path):
    """Load depth calibration parameters (ref 3DGS dataset_readers.py)"""
    with open(depth_params_path, 'r') as f:
        depths_params = json.load(f)
    
    # Compute global median scale
    all_scales = np.array([depths_params[key]["scale"] for key in depths_params])
    if (all_scales > 0).sum():
        med_scale = np.median(all_scales[all_scales > 0])
    else:
        med_scale = 0
    
    for key in depths_params:
        depths_params[key]["med_scale"] = med_scale
    
    return depths_params

def check_depth_reliability(depth_params):
    """Check depth reliability (ref 3DGS cameras.py)"""
    if depth_params is None:
        return False
    
    scale = depth_params["scale"]
    med_scale = depth_params["med_scale"]
    
    if scale < 0.2 * med_scale or scale > 5 * med_scale:
        return False
    
    if scale <= 0:
        return False
    
    return True

def load_and_calibrate_depth(depth_path, depth_params, target_resolution):
    """Load and calibrate inverse depth map (ref 3DGS cameras.py)"""
    # Read 16-bit PNG depth map
    invdepthmap = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    
    if invdepthmap is None:
        return None
    
    if invdepthmap.ndim != 2:
        invdepthmap = invdepthmap[..., 0]
    
    # Resize to target resolution (ref DGS cameras.py)
    invdepthmap = cv2.resize(invdepthmap, target_resolution)
    
    # Normalize to [0,1]
    invdepthmap = invdepthmap.astype(np.float32) / (2**16)
    
    # Filter negative values
    invdepthmap[invdepthmap < 0] = 0
    
    # Apply calibration parameters
    if depth_params is not None and depth_params["scale"] > 0:
        invdepthmap = invdepthmap * depth_params["scale"] + depth_params["offset"]
    
    return invdepthmap

def depth_to_pointcloud(invdepthmap, camera_intrinsic, extrinsic, original_image_path, downsample_factor=4):
    """Convert depth map to point cloud"""
    height, width = invdepthmap.shape
    
    if camera_intrinsic.model == "SIMPLE_PINHOLE":
        fx = fy = camera_intrinsic.params[0]
        cx = width / 2.0
        cy = height / 2.0
    elif camera_intrinsic.model == "PINHOLE":
        fx = camera_intrinsic.params[0]
        fy = camera_intrinsic.params[1]
        cx = camera_intrinsic.params[2] if len(camera_intrinsic.params) > 2 else width / 2.0
        cy = camera_intrinsic.params[3] if len(camera_intrinsic.params) > 3 else height / 2.0
    else:
        print(f"Warning: Unsupported camera model {camera_intrinsic.model}, using PINHOLE approximation")
        fx = camera_intrinsic.params[0]
        fy = camera_intrinsic.params[1] if len(camera_intrinsic.params) > 1 else fx
        cx = width / 2.0
        cy = height / 2.0
    
    # Downsample to reduce point cloud density
    if downsample_factor > 1:
        invdepthmap = invdepthmap[::downsample_factor, ::downsample_factor]
        height, width = invdepthmap.shape
        fx /= downsample_factor
        fy /= downsample_factor
        cx /= downsample_factor
        cy /= downsample_factor
    
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    u = u.flatten()
    v = v.flatten()
    inv_depth = invdepthmap.flatten()
    
    # Filter valid inverse depth values
    valid_mask = (inv_depth > 1e-6) & (inv_depth < 100.0)
    u = u[valid_mask]
    v = v[valid_mask]
    inv_depth = inv_depth[valid_mask]
    
    if len(inv_depth) == 0:
        return None, None
    
    depth = 1.0 / inv_depth
    
    # Filter depth range
    depth_mask = (depth > 0.1) & (depth < 100.0)
    u = u[depth_mask]
    v = v[depth_mask]
    depth = depth[depth_mask]
    
    if len(depth) == 0:
        return None, None
    
    x_cam = (u - cx) * depth / fx
    y_cam = (v - cy) * depth / fy
    z_cam = depth
    
    points_camera = np.column_stack([x_cam, y_cam, z_cam])
    
    # Convert to world coordinates (same as 3DGS dataset_readers.py)
    # Note: COLMAP qvec and tvec are world-to-camera transform
    # We need camera-to-world transform
    R = qvec2rotmat(extrinsic.qvec)  # world-to-camera rotation
    T = np.array(extrinsic.tvec)     # world-to-camera translation
    
    # COLMAP: camera_point = R * world_point + T
    # world_point = R^T * (camera_point - T)
    points_world = np.dot(points_camera - T, R)
    
    # Load image colors if available
    colors = None
    if original_image_path and os.path.exists(original_image_path):
        try:
            image = cv2.imread(original_image_path)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                if downsample_factor > 1:
                    image = cv2.resize(image, (width, height))
                
                colors = image[v.astype(int), u.astype(int)] / 255.0
        except:
            pass
    
    if colors is None:
        colors = np.ones((len(points_world), 3)) * 0.7  # default gray
    
    return points_world, colors

def main():
    # Path configuration
    depth_images_dir = "/home/qilin/Documents/gaussian-splatting/output/building1_32_depth"
    depth_params_file = "/home/qilin/Documents/GS4Building/data/Buildings32/building1/building1_32/undistorted/sparse/0/depth_params.json"
    sparse_model_dir = "/home/qilin/Documents/GS4Building/data/Buildings32/building1/building1_32/undistorted/sparse/0"
    images_dir = "/home/qilin/Documents/GS4Building/data/Buildings32/building1/building1_32/undistorted/images"
    output_pointcloud_path = "output/depth_visualization_3dgs/test_pcd/dense_pointcloud_3dgs.ply"
    
    # Downsampling parameter
    downsample_factor = 4 
    
    print("=== Dense point cloud generation based on 3DGS logic ===")
    print(f"Depth image directory: {depth_images_dir}")
    print(f"Depth parameters file: {depth_params_file}")
    print(f"COLMAP model directory: {sparse_model_dir}")
    print(f"Downsample factor: {downsample_factor}")
    
    # Load COLMAP data
    print("Loading COLMAP camera data...")
    cam_extrinsics, cam_intrinsics = load_colmap_data(sparse_model_dir)
    
    # Load depth parameters
    print("Loading depth calibration parameters...")
    depth_params = load_depth_params(depth_params_file)
    
    print(f"Loaded {len(cam_extrinsics)} camera poses")
    print(f"Loaded {len(depth_params)} depth parameters")
    
    # Process each image to generate point cloud
    all_points = []
    all_colors = []
    processed_count = 0
    reliable_count = 0
    
    for image_id, extrinsic in tqdm(cam_extrinsics.items(), desc="Generating point clouds"):
        image_name = extrinsic.name
        n_remove = len(image_name.split('.')[-1]) + 1
        base_name = image_name[:-n_remove]
        
        # Check depth parameters
        if base_name not in depth_params:
            print(f"Warning: Depth parameters not found for {base_name}")
            continue
        
        # Check reliability
        img_depth_params = depth_params[base_name]
        if not check_depth_reliability(img_depth_params):
            continue
        
        # Depth image path
        depth_image_path = os.path.join(depth_images_dir, f"{base_name}.png")
        if not os.path.exists(depth_image_path):
            print(f"Warning: Depth image does not exist {depth_image_path}")
            continue
        
        original_image_path = os.path.join(images_dir, image_name)
        
        camera_intrinsic = cam_intrinsics[extrinsic.camera_id]
        target_resolution = (camera_intrinsic.width, camera_intrinsic.height)
        
        invdepthmap = load_and_calibrate_depth(depth_image_path, img_depth_params, target_resolution)
        
        if invdepthmap is None:
            continue
        
        points, colors = depth_to_pointcloud(
            invdepthmap, camera_intrinsic, extrinsic, original_image_path, downsample_factor
        )
        
        if points is not None and len(points) > 0:
            all_points.append(points)
            all_colors.append(colors)
            processed_count += 1
            reliable_count += 1
    
    if len(all_points) == 0:
        print("Error: No valid point cloud generated!")
        return

    # Merge all point clouds
    print("Merging point clouds...")
    dense_points = np.vstack(all_points)
    dense_colors = np.vstack(all_colors)

    print(f"Total points: {len(dense_points)}")

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(dense_points)
    pcd.colors = o3d.utility.Vector3dVector(dense_colors)

    # Statistical outlier removal
    print("Applying statistical outlier removal...")
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    # Voxel downsampling (further reduce point cloud density)
    voxel_size = 0.05  # adjustable
    print(f"Applying voxel downsampling (voxel size: {voxel_size})...")
    pcd = pcd.voxel_down_sample(voxel_size)

    output_dir = os.path.dirname(output_pointcloud_path)
    os.makedirs(output_dir, exist_ok=True)

    # Save point cloud
    print(f"Saving dense point cloud to {output_pointcloud_path}")
    o3d.io.write_point_cloud(output_pointcloud_path, pcd)

    # Statistics
    final_point_count = len(pcd.points)
    bbox = pcd.get_axis_aligned_bounding_box()

    summary = f"""
    === Dense Point Cloud Generation Completed ===

    Processing Statistics:
    - Number of processed images: {processed_count}
    - Reliable depth maps: {reliable_count}
    - Original points: {len(dense_points)}
    - Points after filtering: {final_point_count}

    Point Cloud Information:
    - Bounding box: {bbox}
    - Downsample factor: {downsample_factor}
    - Voxel size: {voxel_size}

    Output file: {output_pointcloud_path}

    Note: This point cloud is generated based on 3DGS depth processing logic,
    using the same reliability check and calibration method.
    """

    print(summary)

    # Save report
    report_path = "output/depth_visualization_3dgs/test_pcd/pointcloud_generation_report.txt"
    report_dir = os.path.dirname(report_path)
    os.makedirs(report_dir, exist_ok=True)
    
    with open(report_path, "w") as f:
        f.write(summary)


if __name__ == "__main__":
    main()