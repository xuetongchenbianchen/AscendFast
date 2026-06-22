// add_demo 的 PyTorch 适配层（路线 A：手写 aclnn 两段式）。
//
// 为什么不用 torch_npu 的 EXEC_NPU_CMD 宏：那个宏定义在 torch_npu 源码仓库的
// 内部私有头 op_api_common.h 里，发布的 wheel **没有**打进来。本机 torch_npu
// 2.7.1 的 include 树里既没有 op_api_common.h 也没有 EXEC_NPU_CMD。
// 所以这里直接调 aclnn_add_demo.h 暴露的两段式接口，只依赖 wheel/CANN 里
// **真实存在**的公开头：
//   - aclnn 算子头   : aclnn_add_demo.h（build_out/op_api/include 产出）
//   - aclTensor 构造 : acl/acl.h + aclnn/acl_meta.h（CANN，公开）
//   - 当前 NPU 流    : torch_npu .../core/npu/NPUStream.h（公开）
//   - workspace 分配 : torch_npu .../core/npu/NPUWorkspaceAllocator.h（公开）
#include <vector>

#include <torch/extension.h>

#include "acl/acl.h"
#include "aclnn_add_demo.h"

#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/NPUWorkspaceAllocator.h"

namespace ascendfast {

// at::Tensor -> aclTensor。add_demo 是逐元素、ND、连续，所以 view/storage 同形，
// stride 用 contiguous 行主序即可（调用方负责传连续张量）。
static aclTensor* to_acl(const at::Tensor& t, aclDataType dtype) {
    std::vector<int64_t> dims(t.sizes().begin(), t.sizes().end());
    std::vector<int64_t> strides(t.strides().begin(), t.strides().end());
    return aclCreateTensor(
        dims.data(), dims.size(), dtype,
        strides.data(), /*offset=*/0, ACL_FORMAT_ND,
        dims.data(), dims.size(), t.data_ptr());
}

static aclDataType acl_dtype_of(const at::Tensor& t) {
    switch (t.scalar_type()) {
        case at::kHalf:  return ACL_FLOAT16;
        case at::kFloat: return ACL_FLOAT;
        default:
            TORCH_CHECK(false, "add_demo: unsupported dtype ", t.scalar_type(),
                        " (only float16/float32)");
    }
}
at::Tensor add_demo(const at::Tensor& x, const at::Tensor& y) {
    TORCH_CHECK(x.scalar_type() == y.scalar_type(), "add_demo: x/y dtype mismatch");
    TORCH_CHECK(x.sizes() == y.sizes(), "add_demo: x/y shape mismatch");

    // 算子要求连续输入；非连续就先 contiguous 一份。
    at::Tensor xc = x.contiguous();
    at::Tensor yc = y.contiguous();
    at::Tensor out = at::empty_like(xc);

    aclDataType dt = acl_dtype_of(xc);
    aclTensor* ax = to_acl(xc, dt);
    aclTensor* ay = to_acl(yc, dt);
    aclTensor* az = to_acl(out, dt);

    aclrtStream stream = c10_npu::getCurrentNPUStream();

    // 第一段：问工作区大小 + 拿 executor。
    uint64_t wsSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus ret = aclnnAddDemoGetWorkspaceSize(ax, ay, az, &wsSize, &executor);
    TORCH_CHECK(ret == ACL_SUCCESS, "aclnnAddDemoGetWorkspaceSize failed: ", ret);

    // workspace 走 torch_npu 的 NPU 分配器（绑定当前流，自动随流回收）。
    // 本机 torch_npu 2.7.1 导出的是 at_npu::native::allocate_workspace（返回
    // at::Tensor 持有这块显存）；它必须存活到 aclnnAddDemo 执行完，故声明在外层。
    at::Tensor ws;
    void* wsPtr = nullptr;
    if (wsSize > 0) {
        ws = at_npu::native::allocate_workspace(wsSize, stream);
        wsPtr = ws.data_ptr();
    }

    // 第二段：真正下发到 device。
    ret = aclnnAddDemo(wsPtr, wsSize, executor, stream);
    TORCH_CHECK(ret == ACL_SUCCESS, "aclnnAddDemo failed: ", ret);

    aclDestroyTensor(ax);
    aclDestroyTensor(ay);
    aclDestroyTensor(az);
    return out;
}

}  // namespace ascendfast

// schema：与 add_demo 的输入输出一致。注册进 torch.ops.ascendfast.add_demo。(声明)
TORCH_LIBRARY(ascendfast, m) {
    m.def("add_demo(Tensor x, Tensor y) -> Tensor");
}

// NPU 走 PrivateUse1 dispatch key。
TORCH_LIBRARY_IMPL(ascendfast, PrivateUse1, m) {
    m.impl("add_demo", &ascendfast::add_demo);
}
