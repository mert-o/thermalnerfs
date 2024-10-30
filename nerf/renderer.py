import os
import cv2
import math
import json
import tqdm
import mcubes
import trimesh
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import raymarching
from torch_efficient_distloss import eff_distloss

from .utils import custom_meshgrid
from meshutils import *
from PIL import Image
import matplotlib.pyplot as plt


@torch.cuda.amp.autocast(enabled=False)
def sample_pdf(bins, weights, T, perturb=False):
    # bins: [N, T0+1]
    # weights: [N, T0]
    # return: [N, T]
    
    N, T0 = weights.shape
    weights = weights + 0.01  # prevent NaNs
    weights_sum = torch.sum(weights, -1, keepdim=True) # [N, 1]
    pdf = weights / weights_sum
    cdf = torch.cumsum(pdf, -1).clamp(max=1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1) # [N, T+1]
    
    u = torch.linspace(0.5 / T, 1 - 0.5 / T, steps=T).to(weights.device)
    u = u.expand(N, T)

    if perturb:
        u = u + (torch.rand_like(u) - 0.5) / T
        
    u = u.contiguous()

    inds = torch.searchsorted(cdf, u, right=True) # [N, t]

    below = torch.clamp(inds - 1, 0, T0)
    above = torch.clamp(inds, 0, T0)

    cdf_g0 = torch.gather(cdf, -1, below)
    cdf_g1 = torch.gather(cdf, -1, above)
    bins_g0 = torch.gather(bins, -1, below)
    bins_g1 = torch.gather(bins, -1, above)

    bins_t = torch.clamp(torch.nan_to_num((u - cdf_g0) / (cdf_g1 - cdf_g0)), 0, 1) # [N, t]
    bins = bins_g0 + bins_t * (bins_g1 - bins_g0) # [N, t]

    return bins


@torch.cuda.amp.autocast(enabled=False)
def near_far_from_aabb(rays_o, rays_d, aabb, min_near=0.05):
    # rays: [N, 3], [N, 3]
    # bound: int, radius for ball or half-edge-length for cube
    # return near [N, 1], far [N, 1]

    tmin = (aabb[:3] - rays_o) / (rays_d + 1e-15) # [N, 3]
    tmax = (aabb[3:] - rays_o) / (rays_d + 1e-15)
    near = torch.where(tmin < tmax, tmin, tmax).amax(dim=-1, keepdim=True)
    far = torch.where(tmin > tmax, tmin, tmax).amin(dim=-1, keepdim=True)
    # if far < near, means no intersection, set both near and far to inf (1e9 here)
    mask = far < near
    near[mask] = 1e9
    far[mask] = 1e9
    # restrict near to a minimal value
    near = torch.clamp(near, min=min_near)

    return near, far


class NeRFRenderer(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.opt = opt

        # bound for ray marching (world space)
        self.real_bound = opt.bound

        self.bound = opt.bound
        
        self.cascade = 1 + math.ceil(math.log2(self.bound))

        self.grid_size = opt.grid_size
        self.min_near = opt.min_near
        self.density_thresh = opt.density_thresh

        # prepare aabb with a 6D tensor (xmin, ymin, zmin, xmax, ymax, zmax)
        aabb_train = torch.FloatTensor([-self.real_bound, -self.real_bound, -self.real_bound, self.real_bound, self.real_bound, self.real_bound])
        aabb_infer = aabb_train.clone()
        self.register_buffer('aabb_train', aabb_train)
        self.register_buffer('aabb_infer', aabb_infer)

        self.cuda_ray = opt.cuda_ray
        
        # extra state for cuda raymarching
        if self.cuda_ray:
            # density grid
            density_grid = torch.zeros([self.cascade, self.grid_size ** 3]) # [CAS, H * H * H]
            self.register_buffer('density_grid', density_grid)
            density_bitfield = torch.zeros(self.cascade * self.grid_size ** 3 // 8, dtype=torch.uint8) # [CAS * H * H * H // 8]
            self.register_buffer('density_bitfield', density_bitfield)
            self.mean_density = 0
            self.iter_density = 0
        else:
            raise NotImplementedError("Only cuda ray mode is implemented for ThermalNeRF.")
    
    def forward(self, x, d, **kwargs):
        raise NotImplementedError()

    # separated density and color query (can accelerate non-cuda-ray mode.)
    def density(self, x, **kwargs):
        raise NotImplementedError()

    def update_aabb(self, aabb):
        # aabb: tensor of [6]
        if not torch.is_tensor(aabb):
            aabb = torch.from_numpy(aabb).float()
        self.aabb_train = aabb.clamp(-self.real_bound, self.real_bound).to(self.aabb_train.device)
        self.aabb_infer = self.aabb_train.clone()
        print(f'[INFO] update_aabb: {self.aabb_train.cpu().numpy().tolist()}')
    
    def render(self, rays_o, rays_d, bg_color=None, perturb=False, cam_near_far=None, shading='full', **kwargs):
        # rays_o, rays_d: [N, 3]
        # return: image: [N, 3], depth: [N]
        
        rays_o = rays_o.contiguous()
        rays_d = rays_d.contiguous()

        N = rays_o.shape[0]
        device = rays_o.device

        # pre-calculate near far
        nears, fars = raymarching.near_far_from_aabb(rays_o, rays_d, self.aabb_train if self.training else self.aabb_infer, self.min_near)
        if cam_near_far is not None:
            nears = torch.maximum(nears, cam_near_far[:, 0])
            fars = torch.minimum(fars, cam_near_far[:, 1])
        
        if bg_color is None:
            bg_color = 1

        results = {}

        if self.training:
            
            xyzs, dirs, ts, rays = raymarching.march_rays_train(rays_o, rays_d, self.real_bound, self.opt.contract, self.density_bitfield, self.cascade, self.grid_size, nears, fars, perturb, self.opt.dt_gamma, self.opt.max_steps)

            dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True)
            with torch.cuda.amp.autocast(enabled=self.opt.fp16):
                outputs = self(xyzs, dirs, shading=shading)
                sigmas = outputs['sigma']
                rgbs = outputs['color']
                if self.opt.sc:
                    temps = outputs['temp']                
            if self.opt.rgbt:
                weights, weights_sum, depth, image = raymarching.composite_rays_train_rgbt(sigmas, rgbs, ts, rays, self.opt.T_thresh)
            elif self.opt.sc:
                weights, weights_sum, depth, image, temp_image = raymarching.composite_rays_train_sc(sigmas, rgbs, temps, ts, rays, self.opt.T_thresh)
            elif self.opt.ts:
                rgbs = rgbs.squeeze()
                weights, weights_sum, depth, image = raymarching.composite_rays_train_ts(sigmas, rgbs, ts, rays, self.opt.T_thresh)

            results['num_points'] = xyzs.shape[0]
            results['weights'] = weights
            results['weights_sum'] = weights_sum
        
        else:
            
            dtype = torch.float32
            
            weights_sum = torch.zeros(N, dtype=dtype, device=device)
            depth = torch.zeros(N, dtype=dtype, device=device)

            if self.opt.rgbt:
                image = torch.zeros(N, 4, dtype=dtype, device=device)
            elif self.opt.sc:
                image = torch.zeros(N, 3, dtype=dtype, device=device)
                temp_image = torch.zeros(N, dtype=dtype, device=device)
            elif self.opt.ts:
                image = torch.zeros(N, dtype=dtype, device=device)
            
            n_alive = N
            rays_alive = torch.arange(n_alive, dtype=torch.int32, device=device) # [N]
            rays_t = nears.clone() # [N]

            step = 0
            
            while step < self.opt.max_steps:

                # count alive rays 
                n_alive = rays_alive.shape[0]
                
                # exit loop
                if n_alive <= 0:
                    break

                # decide compact_steps
                n_step = max(min(N // n_alive, 8), 1)

                xyzs, dirs, ts = raymarching.march_rays(n_alive, n_step, rays_alive, rays_t, rays_o, rays_d, self.real_bound, self.opt.contract, self.density_bitfield, self.cascade, self.grid_size, nears, fars, perturb if step == 0 else False, self.opt.dt_gamma, self.opt.max_steps)
                
                dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True)
                with torch.cuda.amp.autocast(enabled=self.opt.fp16):
                    outputs = self(xyzs, dirs, shading=shading)
                    sigmas = outputs['sigma']
                    rgbs = outputs['color']
                    if self.opt.sc:
                        temps = outputs['temp']

                if self.opt.rgbt:
                    raymarching.composite_rays_rgbt(n_alive, n_step, rays_alive, rays_t, sigmas, rgbs, ts, weights_sum, depth, image, self.opt.T_thresh)
                elif self.opt.sc:
                    raymarching.composite_rays_sc(n_alive, n_step, rays_alive, rays_t, sigmas, rgbs, temps, ts, weights_sum, depth, image, temp_image, self.opt.T_thresh)
                elif self.opt.ts:
                    raymarching.composite_rays_ts(n_alive, n_step, rays_alive, rays_t, sigmas, rgbs, ts, weights_sum, depth, image, self.opt.T_thresh)

                rays_alive = rays_alive[rays_alive >= 0]

                step += n_step
        if self.opt.rgbt:
            temp_image = image[...,3:].squeeze()
            temp_image = temp_image + (1 - weights_sum) * bg_color
            image = image[...,:3]

        if self.opt.ts:
            image = image + (1 - weights_sum) * bg_color
        else:
            image = image + (1 - weights_sum).unsqueeze(-1) * bg_color

        
        results['depth'] = depth
        results['image'] = image
        if self.opt.sc or self.opt.rgbt:
            results['temp_image'] = temp_image
            temp_image = temp_image + (1 - weights_sum) * bg_color
        return results

    @torch.no_grad()
    def mark_unseen_triangles(self, vertices, triangles, mvps, H, W):
        # vertices: coords in world system
        # mvps: [B, 4, 4]
        device = self.aabb_train.device

        if isinstance(vertices, np.ndarray):
            vertices = torch.from_numpy(vertices).contiguous().float().to(device)
        
        if isinstance(triangles, np.ndarray):
            triangles = torch.from_numpy(triangles).contiguous().int().to(device)

        mask = torch.zeros_like(triangles[:, 0]) # [M,], for face.

        import nvdiffrast.torch as dr
        glctx = dr.RasterizeGLContext(output_db=False)

        for mvp in tqdm.tqdm(mvps):

            vertices_clip = torch.matmul(F.pad(vertices, pad=(0, 1), mode='constant', value=1.0), torch.transpose(mvp.to(device), 0, 1)).float().unsqueeze(0) # [1, N, 4]

            # ENHANCE: lower resolution since we don't need that high?
            rast, _ = dr.rasterize(glctx, vertices_clip, triangles, (H, W)) # [1, H, W, 4]

            # collect the triangle_id (it is offseted by 1)
            trig_id = rast[..., -1].long().view(-1) - 1
            
            # no need to accumulate, just a 0/1 mask.
            mask[trig_id] += 1 # wrong for duplicated indices, but faster.
            # mask.index_put_((trig_id,), torch.ones(trig_id.shape[0], device=device, dtype=mask.dtype), accumulate=True)

        mask = (mask == 0) # unseen faces by all cameras

        print(f'[mark unseen trigs] {mask.sum()} from {mask.shape[0]}')
        
        return mask # [N]


    @torch.no_grad()
    def mark_untrained_grid(self, dataset, S=64):
        
        # data: reference to the dataset object

        poses = dataset.poses # [B, 4, 4]
        intrinsics = dataset.intrinsics # [4] or [B/1, 4]
        cam_near_far = dataset.cam_near_far if hasattr(dataset, 'cam_near_far') else None # [B, 2]
  
        if isinstance(poses, np.ndarray):
            poses = torch.from_numpy(poses)

        B = poses.shape[0]
        
        if isinstance(intrinsics, np.ndarray):
            fx, fy, cx, cy = intrinsics
        else:
            fx, fy, cx, cy = torch.chunk(intrinsics, 4, dim=-1)
        
        mask_cam = torch.zeros_like(self.density_grid)
        mask_aabb = torch.zeros_like(self.density_grid)

        # pc = []
        poses = poses.to(mask_cam.device)

        X = torch.arange(self.grid_size, dtype=torch.int32, device=self.aabb_train.device).split(S)
        Y = torch.arange(self.grid_size, dtype=torch.int32, device=self.aabb_train.device).split(S)
        Z = torch.arange(self.grid_size, dtype=torch.int32, device=self.aabb_train.device).split(S)

        for xs in X:
            for ys in Y:
                for zs in Z:
                    
                    # construct points
                    xx, yy, zz = custom_meshgrid(xs, ys, zs)
                    coords = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [N, 3], in [0, 128)
                    indices = raymarching.morton3D(coords).long() # [N]
                    world_xyzs = (2 * coords.float() / (self.grid_size - 1) - 1).unsqueeze(0) # [1, N, 3] in [-1, 1]

                    # cascading
                    for cas in range(self.cascade):
                        bound = min(2 ** cas, self.bound)
                        half_grid_size = bound / self.grid_size
                        # scale to current cascade's resolution
                        cas_world_xyzs = world_xyzs * (bound - half_grid_size)

                        # first, mark out-of-AABB region
                        mask_min = (cas_world_xyzs >= (self.aabb_train[:3] - half_grid_size)).sum(-1) == 3
                        mask_max = (cas_world_xyzs <= (self.aabb_train[3:] + half_grid_size)).sum(-1) == 3
                        mask_aabb[cas, indices] += (mask_min & mask_max).reshape(-1)

                        # second, mark out-of-camera region
                        # split pose to batch to avoid OOM
                        head = 0
                        while head < B:
                            tail = min(head + S, B)

                            # world2cam transform (poses is c2w, so we need to transpose it. Another transpose is needed for batched matmul, so the final form is without transpose.)
                            cam_xyzs = cas_world_xyzs - poses[head:tail, :3, 3].unsqueeze(1)
                            cam_xyzs = cam_xyzs @ poses[head:tail, :3, :3] # [S, N, 3]
                            cam_xyzs[:, :, 2] *= -1 # crucial, camera forward is negative now...

                            if torch.is_tensor(fx):
                                cx_div_fx = cx[head:tail] / fx[head:tail]
                                cy_div_fy = cy[head:tail] / fy[head:tail]
                            else:
                                cx_div_fx = cx / fx
                                cy_div_fy = cy / fy
                            
                            min_near = self.opt.min_near if cam_near_far is None else cam_near_far[head:tail, 0].unsqueeze(1)
                            
                            # query if point is covered by any camera
                            mask_z = cam_xyzs[:, :, 2] > min_near # [S, N]
                            mask_x = torch.abs(cam_xyzs[:, :, 0]) < (cx_div_fx * cam_xyzs[:, :, 2] + half_grid_size * 2)
                            mask_y = torch.abs(cam_xyzs[:, :, 1]) < (cy_div_fy * cam_xyzs[:, :, 2] + half_grid_size * 2)
                            mask = (mask_z & mask_x & mask_y).sum(0).bool().reshape(-1) # [N]

                            # for visualization
                            # pc.append(cas_world_xyzs[0][mask])

                            # update mask_cam 
                            mask_cam[cas, indices] += mask
                            head += S
    
        # mark untrained grid as -1
        self.density_grid[((mask_cam == 0) | (mask_aabb == 0))] = -1

        print(f'[mark untrained grid] {((mask_cam == 0) | (mask_aabb == 0)).sum()} from {self.grid_size ** 3 * self.cascade}')

    
    def update_extra_state(self, decay=0.95, S=128):
        # call before each epoch to update extra states.

        if not self.cuda_ray:
            return 
        
        ### update density grid

        with torch.no_grad():

            tmp_grid = - torch.ones_like(self.density_grid)
            
            # full update.
            if self.iter_density < 16:
                X = torch.arange(self.grid_size, dtype=torch.int32, device=self.aabb_train.device).split(S)
                Y = torch.arange(self.grid_size, dtype=torch.int32, device=self.aabb_train.device).split(S)
                Z = torch.arange(self.grid_size, dtype=torch.int32, device=self.aabb_train.device).split(S)

                for xs in X:
                    for ys in Y:
                        for zs in Z:
                            
                            # construct points
                            xx, yy, zz = custom_meshgrid(xs, ys, zs)
                            coords = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [N, 3], in [0, 128)
                            indices = raymarching.morton3D(coords).long() # [N]
                            xyzs = 2 * coords.float() / (self.grid_size - 1) - 1 # [N, 3] in [-1, 1]

                            # cascading
                            for cas in range(self.cascade):
                                bound = min(2 ** cas, self.bound)
                                half_grid_size = bound / self.grid_size
                                # scale to current cascade's resolution
                                cas_xyzs = xyzs * (bound - half_grid_size)
                                # add noise in [-hgs, hgs]
                                cas_xyzs += (torch.rand_like(cas_xyzs) * 2 - 1) * half_grid_size
                                # query density
                                with torch.cuda.amp.autocast(enabled=self.opt.fp16):
                                    sigmas = self.density(cas_xyzs)['sigma'].reshape(-1).detach()
                                # assign 
                                tmp_grid[cas, indices] = sigmas

            # partial update (half the computation)
            else:
                N = self.grid_size ** 3 // 4 # H * H * H / 4
                for cas in range(self.cascade):
                    # random sample some positions
                    coords = torch.randint(0, self.grid_size, (N, 3), device=self.aabb_train.device) # [N, 3], in [0, 128)
                    indices = raymarching.morton3D(coords).long() # [N]
                    # random sample occupied positions
                    occ_indices = torch.nonzero(self.density_grid[cas] > 0).squeeze(-1) # [Nz]
                    rand_mask = torch.randint(0, occ_indices.shape[0], [N], dtype=torch.long, device=self.aabb_train.device)
                    occ_indices = occ_indices[rand_mask] # [Nz] --> [N], allow for duplication
                    occ_coords = raymarching.morton3D_invert(occ_indices) # [N, 3]
                    # concat
                    indices = torch.cat([indices, occ_indices], dim=0)
                    coords = torch.cat([coords, occ_coords], dim=0)
                    # same below
                    xyzs = 2 * coords.float() / (self.grid_size - 1) - 1 # [N, 3] in [-1, 1]
                    bound = min(2 ** cas, self.bound)
                    half_grid_size = bound / self.grid_size
                    # scale to current cascade's resolution
                    cas_xyzs = xyzs * (bound - half_grid_size)
                    # add noise in [-hgs, hgs]
                    cas_xyzs += (torch.rand_like(cas_xyzs) * 2 - 1) * half_grid_size
                    # query density
                    with torch.cuda.amp.autocast(enabled=self.opt.fp16):
                        sigmas = self.density(cas_xyzs)['sigma'].reshape(-1).detach()
                    # assign 
                    tmp_grid[cas, indices] = sigmas

            # ema update
            valid_mask = (self.density_grid >= 0) & (tmp_grid >= 0)
            
        self.density_grid[valid_mask] = torch.maximum(self.density_grid[valid_mask] * decay, tmp_grid[valid_mask])

        self.mean_density = torch.mean(self.density_grid.clamp(min=0)).item() # -1 regions are viewed as 0 density.
        # self.mean_density = torch.mean(self.density_grid[self.density_grid > 0]).item() # do not count -1 regions
        self.iter_density += 1

        # convert to bitfield
        density_thresh = min(self.mean_density, self.density_thresh)
        self.density_bitfield = raymarching.packbits(self.density_grid.detach(), density_thresh, self.density_bitfield)

        # print(f'[density grid] min={self.density_grid.min().item():.4f}, max={self.density_grid.max().item():.4f}, mean={self.mean_density:.4f}, occ_rate={(self.density_grid > density_thresh).sum() / (128**3 * self.cascade):.3f}')
        return None
        