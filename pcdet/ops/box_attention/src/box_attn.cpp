#pragma once
#include <torch/extension.h>


//#ifdef WITH_CUDA

at::Tensor box_attn_cuda_forward(
    const at::Tensor &value,
    const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index,
    const at::Tensor &sampling_loc,
    const at::Tensor &attn_weight,
    const int im2col_step
);

std::vector<at::Tensor> box_attn_cuda_backward(
    const at::Tensor &value,
    const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index,
    const at::Tensor &sampling_loc,
    const at::Tensor &attn_weight,
    const at::Tensor &grad_output,
    const int im2col_step
);

//#endif

at::Tensor box_attn_forward(
    const at::Tensor &value,
    const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index,
    const at::Tensor &sampling_loc,
    const at::Tensor &attn_weight,
    const int im2col_step
)
{
    if (value.is_cuda())
    {
//#ifdef WITH_CUDA
        return box_attn_cuda_forward(
            value, 
            spatial_shapes,
            level_start_index,
            sampling_loc,
            attn_weight,
            im2col_step
        );
//#else
//        AT_ERROR("Not compiled with GPU support");
//#endif
    }
    AT_ERROR("Not implemented on the CPU");
}

std::vector<at::Tensor> box_attn_backward(
    const at::Tensor &value,
    const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index,
    const at::Tensor &sampling_loc,
    const at::Tensor &attn_weight,
    const at::Tensor &grad_output,
    const int im2col_step
)
{
    if (value.is_cuda())
    {
//#ifdef WITH_CUDA
        return box_attn_cuda_backward(
            value, 
            spatial_shapes,
            level_start_index,
            sampling_loc,
            attn_weight,
            grad_output,
            im2col_step
        );
//#else
//        AT_ERROR("Not compiled with GPU support");
//#endif
    }
    AT_ERROR("Not implemented on the CPU");
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &box_attn_forward, "box_attn_forward");
    m.def("backward", &box_attn_backward, "box_attn_backward");
}

