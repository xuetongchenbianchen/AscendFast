// C2 适配层模板：把一个 aclnn 自定义算子接进 torch.ops.ascendfast 命名空间。
//
// 以 add_custom 交付物为具体例子（aclnnAddCustomTemplate）。接新算子时照此改：
//   1. #include 换成交付的 autogen/aclnn_<op>.h
//   2. EXEC_NPU_CMD 第一个参数换成 aclnn<Op>（不带 GetWorkspaceSize 后缀，宏自己拼）
//   3. schema 字符串和函数签名按算子的输入输出改
//
// 编译：用 kernels/csrc/build_adapter.py（torch.utils.cpp_extension），它会自动
// 带上 torch_npu 的 include/lib 和交付的 libcust_opapi.so。
//
// 关键点：EXEC_NPU_CMD 是 torch_npu 提供的宏，封装了 aclnn 的两段式调用
// （GetWorkspaceSize → 分配 workspace → 执行），把 at::Tensor 自动转 aclTensor。
// 你不用手写 workspace 分配，这正是适配层省事的地方。

#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

// torch_npu 的算子下发宏（含 aclnn 两段式封装）。
#include "op_api_common.h"

// 交付的 aclnn 头：声明 aclnnAddCustomTemplateGetWorkspaceSize / aclnnAddCustomTemplate
#include "aclnn_add_custom_template.h"

namespace ascendfast {

// add_custom 语义：z = x + y，逐元素，x/y/z 同形同 dtype（ND, fp16/fp32）。
at::Tensor add_custom(const at::Tensor& x, const at::Tensor& y) {
    // 输出张量：形状/dtype 跟随输入（与算子 IR 的 InferShape 一致）。
    at::Tensor out = at::empty_like(x);

    // EXEC_NPU_CMD(aclnn算子名, 输入..., 输出...)：
    // 宏内部依次调 aclnnAddCustomTemplateGetWorkspaceSize 拿 workspaceSize，
    // 在当前 NPU stream 上分配 workspace，再调 aclnnAddCustomTemplate 执行。
    EXEC_NPU_CMD(aclnnAddCustomTemplate, x, y, out);

    return out;
}

}  // namespace ascendfast

// 注册 schema 到 dispatcher：算子名 add_custom 进 ascendfast 命名空间。
TORCH_LIBRARY(ascendfast, m) {
    m.def("add_custom(Tensor x, Tensor y) -> Tensor");
}

// 把 NPU 实现绑到 PrivateUse1（torch_npu 把 NPU 注册成这个 DispatchKey）。
TORCH_LIBRARY_IMPL(ascendfast, PrivateUse1, m) {
    m.impl("add_custom", &ascendfast::add_custom);
}
