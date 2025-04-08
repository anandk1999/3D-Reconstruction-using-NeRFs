#!/usr/bin/env python3
import os
import sys
import glob
import pickle
import cv2
import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import mcubes  # for mesh extraction via marching cubes
import time

#############################################
#  FreeNeRF Model Definition
#############################################

class NerfModel(nn.Module):
    def __init__(self, embedding_dim_pos=16, embedding_dim_direction=4, hidden_dim=128, T=40000):
        super(NerfModel, self).__init__()
        self.block1 = nn.Sequential(
            nn.Linear(embedding_dim_pos * 6 + 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.block2 = nn.Sequential(
            nn.Linear(embedding_dim_pos * 6 + hidden_dim + 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim + 1)
        )
        self.block3 = nn.Sequential(
            nn.Linear(embedding_dim_direction * 6 + hidden_dim + 3, hidden_dim // 2),
            nn.ReLU()
        )
        self.block4 = nn.Sequential(
            nn.Linear(hidden_dim // 2, 3),
            nn.Sigmoid()
        )
        self.embedding_dim_pos = embedding_dim_pos
        self.embedding_dim_direction = embedding_dim_direction
        self.T = T

    def positional_encoding(self, x, L, step, is_pos=False):
        out = [x]
        for j in range(L):
            out.append(torch.sin((2**j) * x))
            out.append(torch.cos((2**j) * x))
        out = torch.cat(out, dim=1)
        Lmax = 2 * 3 * L + 3
        if is_pos:
            # Gradually reveal higher frequencies during training.
            out[:, int(step / self.T * Lmax) + 3:] = 0.
        return out

    def forward(self, o, d, step):
        emb_x = self.positional_encoding(o, self.embedding_dim_pos, step, is_pos=True)
        emb_d = self.positional_encoding(d, self.embedding_dim_direction, step, is_pos=False)
        h = self.block1(emb_x)
        tmp = self.block2(torch.cat((h, emb_x), dim=1))
        h, sigma = tmp[:, :-1], F.softplus(tmp[:, -1])
        h = self.block3(torch.cat((h, emb_d), dim=1))
        c = self.block4(h)
        return c, sigma

#############################################
#  Rendering and Training Functions
#############################################

def compute_accumulated_transmittance(alphas):
    acc = torch.cumprod(alphas, dim=1)
    return torch.cat((torch.ones((acc.shape[0], 1), device=alphas.device), acc[:, :-1]), dim=-1)

def render_rays(model, ray_origins, ray_directions, step, hn=0, hf=0.5, nb_bins=192):
    device = ray_origins.device
    t = torch.linspace(hn, hf, nb_bins, device=device).expand(ray_origins.shape[0], nb_bins)
    mid = (t[:, :-1] + t[:, 1:]) / 2.
    lower = torch.cat((t[:, :1], mid), dim=-1)
    upper = torch.cat((mid, t[:, -1:]), dim=-1)
    u = torch.rand(t.shape, device=device)
    t = lower + (upper - lower) * u
    delta = torch.cat((t[:, 1:] - t[:, :-1], torch.tensor([1e10], device=device).expand(ray_origins.shape[0], 1)), dim=-1)
    x = ray_origins.unsqueeze(1) + t.unsqueeze(2) * ray_directions.unsqueeze(1)
    # Use same ray direction for all samples
    ray_directions_expanded = ray_directions.unsqueeze(1).expand(-1, nb_bins, -1)
    colors, sigma = model(x.reshape(-1, 3), ray_directions_expanded.reshape(-1, 3), step)
    colors = colors.reshape(x.shape)
    sigma = sigma.reshape(x.shape[:-1])
    alpha = 1 - torch.exp(-sigma * delta)
    weights = compute_accumulated_transmittance(1 - alpha).unsqueeze(2) * alpha.unsqueeze(2)
    c = (weights * colors).sum(dim=1)
    weight_sum = weights.sum(dim=-1).sum(dim=-1)
    # Return both color and sigma for regularization and debugging
    return c + 1 - weight_sum.unsqueeze(-1), sigma

def sample_batch(data, batch_size, device):
    idx = torch.randperm(data.shape[0])[:batch_size]
    if isinstance(data, torch.Tensor):
        return data[idx].to(device=device, dtype=torch.float32)
    else:
        return torch.tensor(data[idx], dtype=torch.float32, device=device)

def train(model, optimizer, training_data, nb_epochs, batch_size, device, hn=0, hf=0.5, nb_bins=192, testing_data=None, test_H=400, test_W=400):
    training_loss = []
    for step in tqdm(range(nb_epochs)):
        batch = sample_batch(training_data, batch_size, device)
        rays_o = batch[:, :3].to(device)
        rays_d = batch[:, 3:6].to(device)
        gt_colors = batch[:, 6:].to(device)
        pred_colors, sigma = render_rays(model, rays_o, rays_d, step, hn=hn, hf=hf, nb_bins=nb_bins)
        loss = ((gt_colors - pred_colors) ** 2).sum()
        # Occlusion regularization: penalize high sigma for near samples
        M = 5  # number of near samples along each ray to regularize
        occlusion_loss = sigma[:, :M].sum() * 0.01  # weight can be adjusted
        loss = loss + occlusion_loss
        training_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (step+1) % 1000 == 0 or step == nb_epochs - 1:
            sigma_np = sigma.detach().cpu().numpy()
            print(f"Step {step+1}: sigma min {sigma_np.min():.6f}, max {sigma_np.max():.6f}, mean {sigma_np.mean():.6f}")
            plt.figure()
            plt.plot(training_loss)
            plt.title("Training Loss")
            plt.xlabel("Iterations")
            plt.ylabel("MSE Loss")
            plt.savefig(f'nerf_data/loss_{step+1}.png')
            plt.close()
            if testing_data is not None:
                test_img = test(model, testing_data, step, nb_bins=nb_bins, H=test_H, W=test_W, hn=hn, hf=hf)
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, f'nerf_data/checkpoint_{step+1}.pt')
    return training_loss

@torch.no_grad()
def test(model, dataset, step, chunk_size=10, img_index=0, nb_bins=192, H=400, W=400, hn=0, hf=0.5, output_dir='novel_views'):
    os.makedirs(output_dir, exist_ok=True)
    rays_per_image = H * W
    start_idx = img_index * rays_per_image
    end_idx = start_idx + rays_per_image
    ray_origins = dataset[start_idx:end_idx, :3]
    ray_directions = dataset[start_idx:end_idx, 3:6]
    rays_o = torch.tensor(ray_origins, dtype=torch.float32, device=model.block1[0].weight.device)
    rays_d = torch.tensor(ray_directions, dtype=torch.float32, device=model.block1[0].weight.device)
    preds = []
    for i in range(int(np.ceil(rays_per_image / (W * chunk_size)))):
        cs = i * W * chunk_size
        ce = min((i+1) * W * chunk_size, rays_per_image)
        pred, _ = render_rays(model, rays_o[cs:ce], rays_d[cs:ce], step, hn=hn, hf=hf, nb_bins=nb_bins)
        preds.append(pred)
    img = torch.cat(preds, dim=0).cpu().numpy().reshape(H, W, 3)
    plt.figure()
    plt.imshow(img)
    plt.axis('off')
    plt.savefig(f'{output_dir}/img_{img_index}_step_{step}.png', bbox_inches='tight')
    plt.close()
    return img

#############################################
#  Data Conversion from nerf_data/ Folder
#############################################

def generate_rays(image, pose, intrinsic):
    """
    Given an image (H x W x 3), a camera pose (4x4), and intrinsic parameters,
    generate a ray for each pixel.
    Each ray is represented as [ray_origin (3), ray_direction (3), rgb (3)].
    Colors are normalized to [0,1].
    """
    H, W = image.shape[:2]
    fx = intrinsic.intrinsic_matrix[0, 0]
    fy = intrinsic.intrinsic_matrix[1, 1]
    cx = intrinsic.intrinsic_matrix[0, 2]
    cy = intrinsic.intrinsic_matrix[1, 2]
    cam_origin = pose[:3, 3]
    R = pose[:3, :3]
    rays = []
    for i in range(H):
        for j in range(W):
            x_cam = (j - cx) / fx
            y_cam = (i - cy) / fy
            direction_cam = np.array([x_cam, y_cam, 1.0])
            direction_cam /= np.linalg.norm(direction_cam)
            ray_direction = R @ direction_cam
            ray_direction /= np.linalg.norm(ray_direction)
            # Get pixel color (convert BGR to RGB and normalize)
            b, g, r = image[i, j]
            color = np.array([r, g, b]) / 255.0
            ray = np.concatenate([cam_origin, ray_direction, color])
            rays.append(ray)
    return np.array(rays)

def convert_data_to_nerf_format(nerf_data_dir, intrinsic, output_training_file, output_testing_file, train_split=0.9):
    """
    Converts images and poses from nerf_data_dir to a single array of shape (N_pixels,9)
    and splits it into training and testing pickle files.
    Assumes images are named 'color_*.png' and poses.npy exists in nerf_data_dir.
    """
    image_pattern = os.path.join(nerf_data_dir, "color_*.png")
    image_files = sorted(glob.glob(image_pattern))
    poses_path = os.path.join(nerf_data_dir, "poses.npy")
    if not os.path.exists(poses_path):
        print("Poses file not found!")
        return
    poses = np.load(poses_path)  # shape (N,4,4)
    if len(image_files) != poses.shape[0]:
        print("Mismatch between number of images and poses!")
        return
    all_rays = []
    for idx, img_file in enumerate(image_files):
        print(f"Processing {img_file} ...")
        image = cv2.imread(img_file)
        if image is None:
            print(f"Failed to load {img_file}")
            continue
        image = cv2.resize(image, (400, 400))
        pose = poses[idx]
        rays = generate_rays(image, pose, intrinsic)
        all_rays.append(rays)
    all_rays = np.concatenate(all_rays, axis=0)
    total_pixels = all_rays.shape[0]
    split_idx = int(train_split * total_pixels)
    training_data = all_rays[:split_idx, :]
    testing_data = all_rays[split_idx:, :]
    with open(output_training_file, 'wb') as f:
        pickle.dump(training_data, f)
    with open(output_testing_file, 'wb') as f:
        pickle.dump(testing_data, f)
    print(f"Saved training data shape {training_data.shape} to {output_training_file}")
    print(f"Saved testing data shape {testing_data.shape} to {output_testing_file}")

#############################################
#  Mesh Extraction from the Learned NeRF
#############################################

def extract_mesh(model, hn, hf, nb_bins, resolution, step, output_filename="mesh.obj"):
    """
    Extracts a 3D mesh from the model's density field using marching cubes.
    Reads scene bounds from nerf_data/bounds.npy.
    resolution: number of samples per axis.
    Uses a fixed view direction [0, 0, 1] for sigma computation.
    """
    bounds_path = os.path.join("nerf_data", "bounds1.npy")
    if os.path.exists(bounds_path):
        bounds = np.load(bounds_path)  # shape (2,3): [bounds_min, bounds_max]
        bounds_min = bounds[0]
        bounds_max = bounds[1]
        grid_bounds = ((bounds_min[0], bounds_max[0]), (bounds_min[1], bounds_max[1]), (bounds_min[2], bounds_max[2]))
        print("Using scene bounds from bounds.npy:", grid_bounds)
    else:
        grid_bounds = ((-1, 1), (-1, 1), (-1, 1))
        print("Bounds file not found, using default bounds:", grid_bounds)
    xmin, xmax = grid_bounds[0]
    ymin, ymax = grid_bounds[1]
    zmin, zmax = grid_bounds[2]
    xs = np.linspace(xmin, xmax, resolution)
    ys = np.linspace(ymin, ymax, resolution)
    zs = np.linspace(zmin, zmax, resolution)
    grid = np.stack(np.meshgrid(xs, ys, zs, indexing='ij'), -1)  # shape (res, res, res, 3)
    grid_flat = grid.reshape(-1, 3)
    fixed_dir = torch.tensor([0, 0, 1], dtype=torch.float32, device=next(model.parameters()).device)
    fixed_dir = fixed_dir.unsqueeze(0).expand(grid_flat.shape[0], 3)
    points = torch.tensor(grid_flat, dtype=torch.float32, device=next(model.parameters()).device)
    with torch.no_grad():
        _, sigma = model(points, fixed_dir, step)
    sigma_np = sigma.cpu().numpy()
    print("Sigma distribution in mesh extraction: min {:.6f}, max {:.6f}, mean {:.6f}".format(sigma_np.min(), sigma_np.max(), sigma_np.mean()))
    sigma = sigma_np.reshape(resolution, resolution, resolution)
    threshold = 0.5  # Adjust this threshold based on sigma statistics if necessary
    print("Using marching cubes threshold:", threshold)
    vertices, triangles = mcubes.marching_cubes(sigma, threshold)
    scale = np.array([(xmax - xmin) / (resolution - 1), (ymax - ymin) / (resolution - 1), (zmax - zmin) / (resolution - 1)])
    vertices = vertices * scale
    vertices[:, 0] += xmin
    vertices[:, 1] += ymin
    vertices[:, 2] += zmin
    mcubes.export_obj(vertices, triangles, output_filename)
    print(f"Mesh saved to {output_filename}")

#############################################
#  Main Function: Mode Dispatcher
#############################################

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py [convert|train|test|mesh]")
        return
    mode = sys.argv[1]
    if mode == "convert":
        # Set up default intrinsic for a 400x400 image (adjust as needed)
        fx, fy, cx, cy = 601.06349533, 601.33647289, 311.77020184, 247.5052488
        intrinsic = o3d.camera.PinholeCameraIntrinsic(400, 400, fx, fy, cx, cy)
        nerf_data_dir = "./nerf_data"
        convert_data_to_nerf_format(nerf_data_dir, intrinsic, "training_data2.pkl", "testing_data2.pkl", train_split=0.9)
    elif mode == "train":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        with open("training_data2.pkl", "rb") as f:
            training_data = pickle.load(f)
        with open("testing_data2.pkl", "rb") as f:
            testing_data = pickle.load(f)
        nb_epochs = 80000
        batch_size = 1024
        model = NerfModel(hidden_dim=256, T=20000).to(device)
        optimizer = optim.Adam(model.parameters(), lr=5e-4)
        
        # # Reshape into (num_images, 400*400, 9)
        # images = training_data.reshape(-1, 400*400, 9)
        
        # # Select the desired image indices
        # selected_indices = [26, 86, 2, 55, 75, 93, 16, 73]
        # selected_images = images[selected_indices]
        
        # # If needed, reshape back to (8*400*400, 9)
        # selected_images = selected_images.reshape(-1, 9)

        # Use realistic near/far bounds based on your capture setup:
        train(model, optimizer, training_data, nb_epochs, batch_size, device, hn=0.1, hf=0.8, nb_bins=192, testing_data=testing_data, test_H=400, test_W=400)
    elif mode == "test":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with open("testing_data2.pkl", "rb") as f:
            testing_data = pickle.load(f)
        nb_epochs = 80000  # or set to final training step value
        model = NerfModel(hidden_dim=256, T=20000).to(device)
        state_dict = torch.load("nerf_data/checkpoint_80000.pt")['model_state_dict']
        model.load_state_dict(state_dict)
        test(model, testing_data, nb_epochs, img_index=0, nb_bins=192, H=400, W=400, hn=0.1, hf=0.8, output_dir="novel_views")
    elif mode == "mesh":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        nb_epochs = 35000
        model = NerfModel(hidden_dim=256, T=20000).to(device)
        state_dict = torch.load("nerf_data/checkpoint_35000.pt")['model_state_dict']
        model.load_state_dict(state_dict)
        resolution = 256
        extract_mesh(model, hn=0.1, hf=0.8, nb_bins=192, resolution=resolution, step=nb_epochs, output_filename="mesh.obj")
    else:
        print("Unknown mode. Use convert, train, test, or mesh.")

if __name__ == "__main__":
    main()
