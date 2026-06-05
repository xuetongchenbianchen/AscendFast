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

### 1.3 ExecutionMode（执行模式）

| 字段 | 类型 | 说明 |
|------|------|------|
| uid | str | 唯一标识 |
| model_id | str | 被优化模型的标识 |
| strategy_uid | str | 来源 OptimizationStrategy.uid |
| correctness_passed | bool \| None | 正确性测试结果，None 表示未测试 |
| extra | dict \| None | 预留扩展字段 |

### 1.4 ProfileResult（Profile 结果）

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

## 2.2 应用优化（Agent 调用）
```python
def apply_optimization(
    strategy: OptimizationStrategy,
    model_id: str,
) -> ExecutionMode:
    """
    将 strategy.prompt_instruction 传给 Agent，Agent 修改模型后
    返回对应的 ExecutionMode（correctness_passed=None）。
    
    Args:
        strategy:  待执行的优化策略
        model_id:   目标模型标识
    
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

## 2.4 Profile
```python
def run_profile(
    mode: ExecutionMode,
) -> ProfileResult:
    """
    对通过正确性测试的 ExecutionMode 执行性能 profile。
    
    Args:
        mode:  correctness_passed=True 的执行模式
    
    Returns:
        ProfileResult，包含优化前后延迟数据
    
    Raises:
        ValueError: 若 mode.correctness_passed != True
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
def optimization_pipeline(
    strategies: list[OptimizationStrategy],
    model_id: str,
    top_k: int = 5,
    baseline_latency: float | None = None,
    _depth: int = 0,
) -> ExecutionMode | None:
    if _depth >= 5 or top_k <= 0:
        return None
    best: tuple[float, ExecutionMode] | None = None  # (latency, mode)
    for strategy in strategies[:top_k]:
        mode = apply_optimization(strategy, model_id)
        mode = run_correctness_test(mode)
        if not mode.correctness_passed:
            continue
        current_latency = run_real_benchmark(model_id)
        # 记录当前最优
        if best is None or current_latency < best[0]:
            best = (current_latency, mode)
        # 达到 2x 加速比，提前返回
        if baseline_latency and current_latency <= baseline_latency / 2:
            return mode
        profile = run_profile(mode)
        analysis = analyze_profile(profile)
        next_strategies = generate_optimization_strategies(analysis, top_k)
        result = optimization_pipeline(
            next_strategies, model_id, top_k, baseline_latency, _depth + 1,
        )
        if result is not None:
            return result
    # 未达到 2x，返回本轮最优（可能为 None，即全部正确性不通过）
    return best[1] if best else None
```
