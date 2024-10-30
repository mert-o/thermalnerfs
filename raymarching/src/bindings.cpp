#include <torch/extension.h>

#include "raymarching.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // utils
    m.def("flatten_rays", &flatten_rays, "flatten_rays (CUDA)");
    m.def("packbits", &packbits, "packbits (CUDA)");
    m.def("near_far_from_aabb", &near_far_from_aabb, "near_far_from_aabb (CUDA)");
    m.def("sph_from_ray", &sph_from_ray, "sph_from_ray (CUDA)");
    m.def("morton3D", &morton3D, "morton3D (CUDA)");
    m.def("morton3D_invert", &morton3D_invert, "morton3D_invert (CUDA)");
    // train
    m.def("march_rays_train", &march_rays_train, "march_rays_train (CUDA)");
    m.def("composite_rays_train_ts_forward", &composite_rays_train_ts_forward, "composite_rays_train_ts_forward (CUDA)");
    m.def("composite_rays_train_ts_backward", &composite_rays_train_ts_backward, "composite_rays_train_ts_backward (CUDA)");
    m.def("composite_rays_train_rgbt_forward", &composite_rays_train_rgbt_forward, "composite_rays_train_rgbt_forward (CUDA)");
    m.def("composite_rays_train_rgbt_backward", &composite_rays_train_rgbt_backward, "composite_rays_train_rgbt_backward (CUDA)");
    m.def("composite_rays_train_sc_forward", &composite_rays_train_sc_forward, "composite_rays_train_sc_forward (CUDA)");
    m.def("composite_rays_train_sc_backward", &composite_rays_train_s_backward, "composite_rays_train_sc_backward (CUDA)");
    // infer
    m.def("march_rays", &march_rays, "march rays (CUDA)");
    m.def("composite_rays", &composite_rays, "composite rays (CUDA)");
    m.def("composite_rays_ts", &composite_rays_ts, "composite rays ts (CUDA)");
    m.def("composite_rays_rgbt", &composite_rays_rgbt, "composite rays rgbt (CUDA)");
    m.def("composite_rays_sc", &composite_rays_sc, "composite rays sc (CUDA)");
}