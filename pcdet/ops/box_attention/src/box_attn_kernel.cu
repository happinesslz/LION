#include "box_attn_kernel.cuh"

#include <vector>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


at::Tensor box_attn_cuda_forward(
    const at::Tensor &value,
    const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index,
    const at::Tensor &sampling_loc,
    const at::Tensor &attn_weight,
    const int im2col_step
)
{
    CHECK_INPUT(value);
    CHECK_INPUT(spatial_shapes);
    CHECK_INPUT(level_start_index);
    CHECK_INPUT(sampling_loc);
    CHECK_INPUT(attn_weight);

    const int batch = value.size(0);
    const int spatial_size = value.size(1);
    const int num_heads = value.size(2);
    const int channels = value.size(3);

    const int num_levels = spatial_shapes.size(0);

    const int num_query = sampling_loc.size(1);
    const int num_point = sampling_loc.size(4);

    const int im2col_step_ = std::min(batch, im2col_step);

    AT_ASSERTM(batch % im2col_step_ == 0, "batch(%d) must divide im2col_step(%d)", batch, im2col_step_);

    auto output = at::zeros({batch, num_query, num_heads, channels}, value.options());

    const int batch_n = im2col_step_;
    auto output_n = output.view({batch/im2col_step_, batch_n, num_query, num_heads, channels});
    const int per_value_size = spatial_size * num_heads * channels;
    const int per_attn_weight_size = num_query * num_heads * num_levels * num_point;
    const int per_sample_loc_size = per_attn_weight_size << 1;

    for (int n = 0; n < batch / im2col_step_; ++n) {
        auto output_columns = output_n.select(0, n);
        AT_DISPATCH_ALL_TYPES(value.scalar_type(), "box_attn_forward_cuda", ( [&] {
            box_attn_im2col_cuda(
                at::cuda::getCurrentCUDAStream(),
                value.data_ptr<scalar_t>() + n * im2col_step_ * per_value_size,
                spatial_shapes.data_ptr<int64_t>(),
                level_start_index.data_ptr<int64_t>(),
                sampling_loc.data_ptr<scalar_t>() + n * im2col_step_ * per_sample_loc_size,
                attn_weight.data_ptr<scalar_t>() + n * im2col_step_ * per_attn_weight_size,
                batch_n, spatial_size, num_heads, channels, num_levels, num_query, num_point,
                output_columns.data_ptr<scalar_t>()
            );
        }));
    }

    output = output.view({batch, num_query, num_heads * channels});

    return output;
}


std::vector<at::Tensor> box_attn_cuda_backward(
    const at::Tensor &value,
    const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index,
    const at::Tensor &sampling_loc,
    const at::Tensor &attn_weight,
    const at::Tensor &grad_output,
    const int im2col_step
)
{
    CHECK_INPUT(value);
    CHECK_INPUT(spatial_shapes);
    CHECK_INPUT(level_start_index);
    CHECK_INPUT(sampling_loc);
    CHECK_INPUT(attn_weight);
    CHECK_INPUT(grad_output);

    const int batch = value.size(0);
    const int spatial_size = value.size(1);
    const int num_heads = value.size(2);
    const int channels = value.size(3);

    const int num_levels = spatial_shapes.size(0);

    const int num_query = sampling_loc.size(1);
    const int num_point = sampling_loc.size(4);

    const int im2col_step_ = std::min(batch, im2col_step);

    AT_ASSERTM(batch % im2col_step_ == 0, "batch(%d) must divide im2col_step(%d)", batch, im2col_step_);

    auto grad_value = at::zeros_like(value);
    auto grad_sampling_loc = at::zeros_like(sampling_loc);
    auto grad_attn_weight = at::zeros_like(attn_weight);

    const int batch_n = im2col_step_;
    const int per_value_size = spatial_size * num_heads * channels;
    const int per_sample_loc_size = num_query * num_heads * num_levels * num_point * 2;
    const int per_attn_weight_size = num_query * num_heads * num_levels * num_point;
    auto grad_output_n = grad_output.view({batch/im2col_step_, batch_n, num_query, num_heads, channels});

    for (int n = 0; n < batch / im2col_step_; ++n) {
        auto grad_output_columns = grad_output_n.select(0, n);
        AT_DISPATCH_ALL_TYPES(value.scalar_type(), "box_attn_backward_cuda", ( [&] {
            box_attn_col2im_cuda(
                at::cuda::getCurrentCUDAStream(),
                grad_output_columns.data_ptr<scalar_t>(),
                value.data_ptr<scalar_t>() + n * im2col_step_ * per_value_size,
                spatial_shapes.data_ptr<int64_t>(),
                level_start_index.data_ptr<int64_t>(),
                sampling_loc.data_ptr<scalar_t>() + n * im2col_step_ * per_sample_loc_size,
                attn_weight.data_ptr<scalar_t>() + n * im2col_step_ * per_attn_weight_size,
                batch_n, spatial_size, num_heads, channels, num_levels, num_query, num_point,
                grad_value.data_ptr<scalar_t>() + n * im2col_step_ * per_value_size,
                grad_sampling_loc.data_ptr<scalar_t>() + n * im2col_step_ * per_sample_loc_size,
                grad_attn_weight.data_ptr<scalar_t>() + n * im2col_step_ * per_attn_weight_size
            );
        }));
    }

    return {grad_value, grad_sampling_loc, grad_attn_weight};
}
