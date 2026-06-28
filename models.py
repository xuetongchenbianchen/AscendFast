# Data entities. LEVER_KINDS is the single source of truth for optimization levers;
# ExecutionMode + ChangeRecord realize the workspace model; StageOutcome + RunLedger
# realize the observability layer.
from __future__ import annotations
from dataclasses import dataclass, field


# --------------------------------------------------------------------------- #
# 优化杠杆（lever）的权威枚举——单一真相源。
# strategy 选 lever、apply 记 kind、ledger 归因都引用这里；新增/改名 lever 只动
# 这一处，其余 Python 文件 import 本常量，不再各自硬编码字符串（文档侧手工对齐）。
#
# 四个 canonical lever 对应 build_model 的四个改动层级（详见 npu-strategy skill）：
#   forward_patch   — monkey-patch 某个 nn.Module.forward（最窄，治单算子）
#   operator_fusion — 改 config / attn_implementation，整条路径切融合后端
#   graph_rewrite   — 在 build_model() 里包整模型（torch.compile / NPU 图模式）
#   loading_time    — 加载期一次性处理（权重 ND→NZ、dtype 清理、静态 KV cache、padding）
# kvcache / quantize / config / parallelism 不是平级 lever：前三者都是 loading_time
# 的子情况，parallelism 单卡 NPU 用不到——不要把它们当独立 kind。
# --------------------------------------------------------------------------- #
LEVER_KINDS = ("forward_patch", "operator_fusion", "graph_rewrite", "loading_time")
# apply 侧合法的 kind 全集：四个 lever + custom（实在无法归类时的兜底，strategy 不用）。
CHANGE_KINDS = LEVER_KINDS + ("custom",)


@dataclass
class ChangeRecord:
    """一次优化动作的自描述记录，喂给下一轮 Agent 和人阅读。

    优化方案本身是异构的（落在四个 lever 之一：forward_patch / operator_fusion /
    graph_rewrite / loading_time，或尚未归类的 custom），但每一步都收敛成这条统一
    记录：做了什么（summary/details）、动了哪些文件（files）、怎么回退（revert_cmd）。
    """
    mode_uid: str               # 引入本次修改的 ExecutionMode.uid
    strategy_uid: str           # 来源 OptimizationStrategy.uid
    kind: str                   # CHANGE_KINDS 之一：forward_patch | operator_fusion
                                #  | graph_rewrite | loading_time | custom
    summary: str                # 一句话：这一步做了什么
    details: str                # 详细：动了哪些模块/算子、为什么、有何约束
    files: list[str]            # 本步新增/修改的文件（相对 workspace_dir）
    revert_cmd: str | None = None
    metadata: dict | None = None


@dataclass
class StageOutcome:
    """一个环节（benchmark/profile/analyze/strategy/apply/correctness/agent_call）
    的成败判定，与 ChangeRecord 同构：每个环节"已经在做但形态不一"的成败判断，
    都收敛成这一条统一记录——做了什么环节（stage）、过没过（ok）、为什么（reason）。

    门禁是喂给 stage() 的纯函数；它们的返回值落到 ok/reason。异常被 stage()
    捕获后也落成一条 ok=False 的 StageOutcome，不再带着 stacktrace 炸穿整条 run。
    """
    stage: str                  # benchmark|profile|analyze|strategy|apply|correctness|agent_call|decision
    ok: bool
    reason: str = ""            # 失败原因；成功时为 ""
    mode_uid: str | None = None # 该环节作用/产出的 ExecutionMode.uid
    metadata: dict | None = None


@dataclass
class RunLedger:
    """一次 optimize() 的 run 级记录：这一次探索了哪棵树、每个环节成败、为什么停。

    mode 级产物（manifest/report）记录单个变体；RunLedger 记录贯穿整条递归的
    决策轨迹。确定性、离线可用，不沾 agent——agent_call 只是 outcomes 里一种
    stage，用来把"为什么没效果"从黑盒里捞出来。
    """
    run_uid: str
    model_id: str
    outcomes: list["StageOutcome"] = field(default_factory=list)
    stop_reason: str | None = None      # reached_2x|max_depth|no_strategies|exhausted|stage_failed:<stage>
    best_mode_uid: str | None = None
    best_latency: float | None = None
    baseline_latency: float | None = None


@dataclass
class OptimizationStrategy:
    uid: str
    local_speedup_ratio: float
    measures: list[str]
    prompt_instruction: str
    extra: dict | None = None


# --------------------------------------------------------------------------- #
# 自定义算子的请求/产物：坐在 apply 的两阶段之间，由 operator-agent 消费/产出。
#
# 拆分动机：apply-agent 的 HOW 其实混了两种性质迥异的事——「写 AscendC kernel +
# 编译 + 装进 CANN」(分钟级、要编译、改全局)和「把算子接进 build_model()」(秒级、
# 只动 workspace)。OperatorSpec/Artifact 把前者从 apply 里剥出来交给 operator-agent。
#
# 谁产 spec：spec 的作者是 **apply-agent**，不是 strategy。理由——
# strategy 只看得到 profile 的热点算子名，看不到真实 forward 源码/真实 dtype/shape；
# apply-agent fork 出 workspace 后能读真实代码与 model/config.json，产出的 spec 带准确
# 的 arch_params 和一段可执行的 torch_reference(数值金标准)。所以流程是两阶段握手：
# apply(phase1) 读代码→发布 OperatorSpec → operator-agent 据此 design+compile+install+
# 数值自检 → apply(phase2) 把已验证的 OperatorArtifact 接进 build_model()。strategy 仍
# 可在 extra.custom_operator 里给一个「提示」，但采不采纳由 apply-agent 读完真实代码定。
# operator 生成失败 ≠ strategy 失败：artifact 缺省为 None，apply 退回官方/eager 算子。
# --------------------------------------------------------------------------- #
@dataclass
class OperatorSpec:
    """apply-agent(phase1) → operator-agent 的请求：要一个什么算子，及为什么官方不够。

    WHAT/WHY 级别，不含 kernel 实现细节(那是 operator-agent 的 HOW)。arch_params 是
    apply-agent 从本模型架构(workspace 的 model/config.json)读出的特化参数(hidden_size /
    num_heads / head_dim / dtype / eps ...)，让 operator-agent 能为「本模型」而非
    「通用情况」特化 kernel——这正是自定义算子相对官方通用算子的价值来源。torch_reference
    是 apply-agent 从真实 forward 抽出的一段可执行参考，给 operator-agent 当数值 oracle。
    """
    op_name: str                        # 期望的算子名(下划线小写，如 rms_norm_residual)，
                                        #  最终注册为 torch.ops.ascendfast.<op_name>
    semantic: str                       # 算子数学语义(一句话/伪代码)，operator-agent 据此写 kernel
    why_custom: str                     # 为什么官方 torch_npu 没有合适实现(融合点/特化点)
    fusion_targets: list[str] = field(default_factory=list)  # 想融进一个 kernel 的算子序列
    arch_params: dict = field(default_factory=dict)          # 本模型架构特化参数
    expected_signature: str | None = None                    # 期望调用签名(自然语言/伪签名)
    torch_reference: str | None = None  # 算子的 I/O 契约 + 数值金标准：一段自包含 torch
                                        #  参考(class Model + get_inputs)，由 apply-agent 从
                                        #  build_model() 接线点**真实要被替换掉的那段 eager
                                        #  代码**抽出——forward 是精确语义，get_inputs 给出真实
                                        #  流经该点的 shape/dtype。operator-agent 据此设计 kernel
                                        #  的 tiling/签名，并 exec 它当 oracle 做数值自检。


@dataclass
class OperatorArtifact:
    """operator-agent → apply 的产物：一个已注册、已数值自检的 torch.ops.ascendfast.<op>。

    apply-agent 拿到它就像拿到一个「已知签名、已验证」的算子，把它接进 build_model()
    即可(并保留官方/eager fallback)。installed=False 或数值不过关时，optimization 的
    gate_operator 会拦下、artifact 不传给 apply，apply 退回官方算子。
    """
    op_name: str                        # 算子名(同 OperatorSpec.op_name)
    qualified_name: str                 # 完整调用名，如 "torch.ops.ascendfast.rms_norm_residual"
    signature: str                      # 实际调用签名，如 "rms_norm_residual(x, residual, gamma, eps) -> (y, new_residual)"
    installed: bool = False             # B链.run 已装 + A链.so 已编 + import 后 op 真实存在
    supported_dtypes: list[str] = field(default_factory=list)   # 如 ["float16", "float32"]
    numeric_max_rel_err: float | None = None    # 对 fp32 参考的最大相对误差(数值自检结果)
    usage_note: str = ""                # 给 apply-agent 的接入提示(形状约束、reshape、返回 tuple 等)
    files: list[str] = field(default_factory=list)              # kernels/ 下新增/改动的文件
    metadata: dict | None = None


@dataclass
class AnalysisResult:
    uid: str
    top_ops: list[str]
    hot_groups: dict[str, list[str]]
    extra: dict | None = None
    model_id: str | None = None
    device_kind: str | None = None
    device_name: str | None = None
    dtype: str | None = None
    profile_report_path: str | None = None
    dataset: dict | None = None
    top_kernels: list[dict] = field(default_factory=list)
    op_type_totals: dict[str, dict] = field(default_factory=dict)
    roofline_summary: dict[str, float] = field(default_factory=dict)
    profile_findings: list[str] = field(default_factory=list)


@dataclass
class ExecutionMode:
    """一个自包含、可运行的"模型变体快照"。

    workspace_dir 是 fork 自父 mode 的完整可运行目录；entrypoint 暴露统一入口
    build_model() -> (model, tokenizer)，无论里面是何种优化，correctness/profile
    都只通过这个入口加载，对优化方案本身无知。change_log 是从 root 累积到本步
    的全部修改（append-only），下一轮 apply 会把它注入 Agent 以便叠加优化。
    """
    uid: str
    model_id: str                                   # 基础模型标识
    strategy_uid: str                               # 产生本步的 strategy（baseline 为 "baseline"）
    workspace_dir: str                              # 自包含可运行目录（物化后的优化模型）
    parent_uid: str | None = None                   # 父 mode.uid；None = baseline 根节点
    entrypoint: str = "build_model.py"              # 统一入口文件（相对 workspace_dir）
    change_log: list[ChangeRecord] = field(default_factory=list)
    correctness_passed: bool | None = None
    extra: dict | None = None


@dataclass
class ProfileResult:
    """Profile 的结果：算子级性能分析数据。

    profile 只负责定位热点（算子占比、roofline），不产出延迟。
    真实的 end-to-end 延迟由 benchmark.py 在自己的数据集上测量，
    两者数据集口径不同，不可混用。
    """
    uid: str
    execution_mode_uid: str
    extra: dict | None = None
    profile_report_path: str | None = None
    profiler_output_dir: str | None = None
    profile_report: dict | None = None
