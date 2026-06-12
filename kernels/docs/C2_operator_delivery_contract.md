# C2 算子交付契约 —— 别的团队交算子给你时，必须满足什么

> 你不写 kernel，你负责"把别人写好的高性能 NPU 算子接进 AscendFast"。
> 本文定义**交付物格式**和**接入机制**。实地参考样例：
> `/models/share/miaoliuyang/ascendc/add_custom/custom_op/build_out/`

## 一、交付物清单：别的团队必须给你这 3 样

一个用 Ascend C 写、`msopgen` 工程编译出来的自定义算子，编译后在 `build_out/`
产出以下东西。你接入**只需要其中 3 类**：

| 交付物 | 路径（相对 build_out/） | 作用 |
|---|---|---|
| **算子包** | `custom_opp_<os>_<arch>.run` | 装到 CANN，把 kernel + tiling 注册进 ASCEND_OPP_PATH |
| **aclnn 头文件** | `autogen/aclnn_<op>.h` | 对外 C 接口签名（你调用的契约） |
| **op_api 库** | `op_host/libcust_opapi.so` | 含 `aclnn<Op>` 符号的动态库，链接用 |

其余（`op_proto`、`*.ini`、kernel 源码）是算子内部产物，接入时不用碰。

## 二、对外接口契约：每个 aclnn 算子都是"两段式"

打开任意 `autogen/aclnn_<op>.h`，会看到**雷打不动的两个函数**（这是所有 aclnn
算子的统一范式，不是某个算子特有的）：

```c
// 第一段：算 workspace 大小 + 构造 executor。入参是所有 tensor。
aclnnStatus aclnn<Op>GetWorkspaceSize(
    const aclTensor *x, const aclTensor *y, const aclTensor *out,  // ← 算子的输入输出
    uint64_t *workspaceSize,    // 输出：需要多大 workspace
    aclOpExecutor **executor);  // 输出：执行句柄

// 第二段：拿着 workspace 和 executor 真正下发到 stream 执行。
aclnnStatus aclnn<Op>(
    void *workspace, uint64_t workspaceSize,
    aclOpExecutor *executor, aclrtStream stream);
```

**这就是你对交付方的硬性要求**：
1. 算子名用大驼峰（`AddCustomTemplate`），生成的接口即 `aclnnAddCustomTemplate*`。
2. tensor 参数顺序 = 先所有 input，再所有 output（看 `.h` 注释里的顺序）。
3. 头文件里写明每个 tensor 的 `param_type`（required/optional）。
4. 算子的 dtype/format 支持范围在算子工程的 IR json 里声明（见 add_custom.json：
   `format: ["ND"]`，`type: ["float16","float"]`）——你接入时传的 tensor 必须落在
   这个范围内，否则 GetWorkspaceSize 直接报错。
5. 必须 `AddConfig("ascend910_93")`（本机芯片 Ascend910_9382），否则装不上。

## 三、接入机制：aclnn 接口 → torch.ops.ascendfast

aclnn 是裸 C 接口，要变成 `torch.ops.ascendfast.<op>` 还差一层 C++ 适配。整条链：

```
别人交付              你写的适配层                          AscendFast 调用
─────────            ──────────────                       ──────────────
.run 包      装到   →  EXEC_NPU_CMD(aclnnXxx, ...)   注册→  torch.ops.ascendfast.xxx
aclnn_x.h    CANN     把 at::Tensor 转 aclTensor 下发        ↑ build_model 的 patch 调它
libcust_opapi.so      TORCH_LIBRARY 注册 schema + 内核
```

适配层做两件事（torch_npu 的 op_plugin 对官方 292 个算子做的也是这件事）：
1. **下发**：`EXEC_NPU_CMD(aclnnAddCustomTemplate, x, y, out)` —— torch_npu 提供的宏，
   自动处理 GetWorkspaceSize→分配 workspace→执行 两段式，把 at::Tensor 转 aclTensor。
2. **注册**：`TORCH_LIBRARY(ascendfast, m)` 声明 schema，
   `TORCH_LIBRARY_IMPL(ascendfast, PrivateUse1, m)` 把 NPU 内核绑上去。

模板见 `kernels/csrc/adapter_template.cpp`。

## 四、装算子包

```bash
# 安装交付的 .run 到 CANN 的自定义算子目录
bash custom_opp_<os>_<arch>.run
# 装完出现在 $ASCEND_OPP_PATH/vendors/customize/，import torch_npu 时可被找到
```

## 五、给交付方的一页纸要求（拷给别的团队）

> 1. 用 Ascend C + msopgen 工程，`AddConfig("ascend910_93")`。
> 2. 交付 `build_out/` 里的：`custom_opp_*.run` + `autogen/aclnn_<op>.h` +
>    `op_host/libcust_opapi.so`。
> 3. 算子命名大驼峰；tensor 参数顺序 input 在前、output 在后。
> 4. 在 `aclnn_<op>.h` 或单独文档里写明每个 tensor 的 dtype/format/shape 约束、
>    required/optional。
> 5. 附一组参考输入输出（供我做 allclose 回归）。
