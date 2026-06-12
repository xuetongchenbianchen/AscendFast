# 接口文档

## 1. 数据实体定义

### 1.1 OptimizationStrategy（优化方案）

| 字段 | 类型 | 说明 |
|------|------|------|
| uid | str | 唯一标识 |
| local_speedup_ratio | float | 超参数局部的预期加速比 |
| measures | list[str] | 具体措施列表 |
| prompt_instruction | str | 执行该优化的 Agent prompt 指令 |
| extra | dict \| None | 预留扩展字段 |

### 1.2 AnalysisResult（分析结果）

| 字段 | 类型 | 说明 |
|------|------|------|
| uid | str | 唯一标识 |
| total_latency | float | 端到端总延迟（ms） |
| top_ops | list[str] | 耗时 Top 算子列表 |
| hot_groups | dict[str, list[str]] | 算子热点聚合分组，key 为组名 |
| extra | dict \| None | 预留扩展字段 |
| model_id | str \| None | 模型标识 |
| device_kind | str \| None | 设备类型 |
| device_name | str \| None | 设备名称 |
| dtype | str \| None | profile 使用的数据类型 |
| op_type_totals | dict[str, dict] | 按 op_type 聚合后的耗时统计 |
| roofline_summary | dict[str, float] | roofline 分类耗时统计 |
| profile_findings | list[str] | profile 分析阶段生成的优化提示 |

### 1.3 ExecutionMode（执行模式 / 模型变体快照）

每个 ExecutionMode 是一个**自包含、可运行的模型变体快照目录**，fork 自父 mode。
统一入口 `build_model() -> (model, tokenizer)` 屏蔽优化方案的异构性，correctness /
profile 只通过该入口加载，对优化内容本身无知。

| 字段 | 类型 | 说明 |
|------|------|------|
| uid | str | 唯一标识 |
| model_id | str | 被优化模型的标识 |
| strategy_uid | str | 产生本步的 OptimizationStrategy.uid（baseline 为 "baseline"） |
| workspace_dir | str | 自包含可运行目录（物化后的优化模型，绝对路径） |
| parent_uid | str \| None | 父 mode.uid；None 表示 baseline 根节点 |
| entrypoint | str | 统一入口文件，默认 `build_model.py`（相对 workspace_dir） |
| change_log | list[ChangeRecord] | 从 root 累积到本步的全部修改（append-only） |
| correctness_passed | bool \| None | 正确性测试结果，None 表示未测试 |
| extra | dict \| None | 预留扩展字段 |

### 1.4 ChangeRecord（单步修改记录）

一次优化动作的自描述记录。优化方案异构（算子融合 / 图改写 / forward patch /
kvcache / 并行 / 量化 / 尚未想到的方案），但每步都收敛成这条统一记录，供下一轮
Agent 叠加优化时阅读。

| 字段 | 类型 | 说明 |
|------|------|------|
| mode_uid | str | 引入本次修改的 ExecutionMode.uid |
| strategy_uid | str | 来源 OptimizationStrategy.uid |
| kind | str | forward_patch \| operator_fusion \| graph_rewrite \| kvcache \| parallelism \| quantize \| config \| custom |
| summary | str | 一句话：这一步做了什么 |
| details | str | 详细：动了哪些模块/算子、为什么、有何约束 |
| files | list[str] | 本步新增/修改文件（相对 workspace_dir） |
| revert_cmd | str \| None | 回退命令 |
| metadata | dict \| None | 预留扩展字段 |

### 1.5 ProfileResult（Profile 结果）

| 字段 | 类型 | 说明 |
|------|------|------|
| uid | str | 唯一标识 |
| execution_mode_uid | str | 来源 ExecutionMode.uid |
| latency_before | float | 优化前延迟（ms） |
| latency_after | float | 优化后延迟（ms） |
| extra | dict \| None | 预留扩展字段 |

---

## 2. 函数接口

### 2.1 规则生成优化策略

```python
def generate_optimization_strategies(
    analysis: AnalysisResult,
    max_count: int = 5,
) -> list[OptimizationStrategy]:
    """
    规则版 strategy_agent：读取 AnalysisResult，
    生成候选 OptimizationStrategy 列表。

    Args:
        analysis:    当前模型的分析结果
        max_count:   返回策略数量上限

    Returns:
        基于热点规则生成的 OptimizationStrategy 列表
    """
```

## 2.2 应用优化（Agent 调用，fork 叠加）
```python
def ensure_baseline_mode(model_id: str, model_dir: str | Path) -> ExecutionMode:
    """物化 baseline ExecutionMode：workspace 硬链接原始模型 + 标准入口。
    baseline 本身即可运行的根 mode，后续 apply 永远从某个 base_mode 出发。
    """

def apply_optimization(
    strategy: OptimizationStrategy,
    base_mode: ExecutionMode,
) -> ExecutionMode:
    """在 base_mode 的快照之上叠加 strategy，返回新的 ExecutionMode。

    1. fork base_mode.workspace_dir → 新 work_dir（大权重硬链接，零拷贝）。
    2. 把 base_mode.change_log 注入 prompt，要求 apply-agent 在已有
       优化之上叠加，不撤销/重复，并保证 build_model() 仍可运行。
    3. Agent 原地修改 work_dir，返回一条 ChangeRecord。
    4. 框架追加 ChangeRecord、写 manifest，新 mode 的 change_log =
       base_mode.change_log + [新记录]，parent_uid = base_mode.uid。

    Args:
        strategy:   待执行的优化策略
        base_mode:  在其快照之上叠加优化的基础 mode（baseline 或上一轮产物）

    Returns:
        ExecutionMode，correctness_passed 尚未填写
    """
```

## 2.3 正确性测试
```python
def run_correctness_test(
    mode: ExecutionMode,
) -> ExecutionMode:
    """
    对 ExecutionMode 执行正确性验证，填写 correctness_passed 字段。
    测试不通过时直接丢弃（调用方应跳过后续步骤）。
    
    Args:
        mode:  待测试的执行模式
    
    Returns:
        填写了 correctness_passed 的 ExecutionMode
    """
```

## 2.4 Profile（诊断，Agent 调用）
```python
def run_profile(
    mode: ExecutionMode,
) -> ProfileResult:
    """对通过正确性测试的 ExecutionMode 做诊断性 profile（Agent 优先 + 确定性 fallback）。

    用 data/ 模拟数据衡量算子热点，产出 profile_report.json，供 analyze_profile
    生成下一轮策略。模型与 tokenizer 均来自 mode.workspace_dir 的 build_model()
    统一入口（唯一真相源），不再从原始权重目录重载。

    与 run_real_benchmark 的区别：
    - run_profile        —— 模拟数据，做诊断（算子级热点）。
    - run_real_benchmark —— 真实领域数据集，测目标延迟（2x 加速比标尺）。

    Args:
        mode:  correctness_passed=True 的执行模式

    Returns:
        ProfileResult（profile_report + latency_stats）

    Raises:
        ValueError: 若 mode.correctness_passed != True
    """

def run_real_benchmark(
    mode: ExecutionMode,
    dataset_path: str | None = None,
) -> float:
    """用真实领域数据集测端到端延迟（ms），即 2x 加速比的标尺。

    同样通过 mode.workspace_dir 的 build_model() 加载，唯一区别是喂入真实领域
    数据集而非模拟数据。（真实数据集接入为后续工作，当前为桩。）

    Returns:
        端到端平均延迟（ms）
    """
```

## 2.5 分析整理（analysis.py）
```python
def analyze_profile(
    profile: ProfileResult,
) -> AnalysisResult:
    """
    将 ProfileResult 整理汇总为 AnalysisResult，
    结果可反馈至下一轮策略生成。
    
    Args:
        profile:  本轮 profile 结果
    
    Returns:
        AnalysisResult
    """
```

## 3. 主流程伪代码
```python
# 顶层：物化 baseline → 从 depth=0 统一迭代，无首轮特例
def run(model_id, model_dir, top_k=5):
    baseline = ensure_baseline_mode(model_id, model_dir)
    baseline.correctness_passed = True          # 原始模型按定义正确
    return optimize(baseline, top_k=top_k)      # -> (best_mode, best_latency)


def optimize(base_mode, baseline_latency=None, depth=0, top_k=5):
    # ① 诊断：每个 mode（含 baseline）进来先 profile + analyze（唯一测延迟处）
    analysis = analyze_profile(run_profile(base_mode))
    latency = analysis.total_latency
    if baseline_latency is None:
        baseline_latency = latency              # baseline 自己定标尺
    if latency <= baseline_latency / 2 or depth >= MAX_DEPTH:
        return base_mode, latency               # 达成 2x 或到底，止步

    # ② 策略生成（唯一一处）；胜出子 mode 作为新 base_mode 叠加优化
    strategies = generate_optimization_strategies(analysis, top_k)
    best_mode, best_lat = base_mode, latency
    for strategy in strategies[:top_k]:
        child = run_correctness_test(apply_optimization(strategy, base_mode))
        if not child.correctness_passed:
            continue
        cand_mode, cand_lat = optimize(child, baseline_latency, depth + 1, top_k)
        if cand_lat < best_lat:
            best_mode, best_lat = cand_mode, cand_lat
            if best_lat <= baseline_latency / 2:
                break                           # 提前命中 2x
    return best_mode, best_lat
```

要点：
- baseline 是 depth=0 的 base_mode，与深层子节点完全同构，无"循环外先生成策略"特例。
- 延迟只有 `run_profile → analysis.total_latency` 一个真相源；`run_real_benchmark`
  在真实领域数据集就位后用于最终目标衡量，二者语义分离。
- 模型 + tokenizer 唯一来自 workspace 的 `build_model()`，apply / profile / benchmark
  / correctness 全部经此加载。
```
