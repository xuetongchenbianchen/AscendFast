"""verify：把每个环节"已经在做但形态不一"的成败判断，收敛成一个 StageOutcome

可观测层：stage() 吞异常落成 StageOutcome、gate_* 纯函数门禁、
RunLedger 落盘决策轨迹。gate_apply 与 apply 的运行 forward gate 配合。
+ 一个 stage() 机制记录下来；门禁是喂给它的纯函数。

不发明新观测层。这里只做三件事，全部围绕 RunLedger / StageOutcome 两个实体：

1. stage()   —— 上下文管理器：运行环节体 → 记一条 StageOutcome → **吞掉异常**
                （记成 ok=False, reason="<ExcType>: ..."），让单点失败不再带着
                stacktrace 炸穿整条 run。调用方读 outcome.ok 决定继续还是收尾。
2. gate_*()  —— 纯函数门禁：把两道本就该存在却隐式缺失的判断显式化：
                  - gate_strategy：策略列表为空 → run 该明确停，而非静默零循环。
                  - gate_apply  ：apply 是否产出了真 ChangeRecord（None bug 的根因），
                                  现在是 gate 的自然结果，不再是补丁。
3. ledger    —— set_current_ledger / record_agent_call / write_ledger：agent_client
                在返回 None 前记一条 agent_call 事件（区分 disabled/timeout/...），
                这是"为什么没效果"的第一元凶，此前完全黑盒。

determinstic、离线可用：记录与门禁不沾 agent。agent_call 只是 outcomes 里的一种
stage，永不参与决策。
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

from models import ExecutionMode, OperatorArtifact, OptimizationStrategy, RunLedger, StageOutcome

_LEDGER_NAME = "run_ledger.json"
_PROJECT_ROOT = Path(__file__).parent
_RUNS_DIR = _PROJECT_ROOT / "runs"

# 当前 run 的 ledger。agent_client 等底层模块靠它记录而无需层层透传 ledger 引用，
# 与项目"行为模块只对实体操作、靠引用解耦"的方式一致。run() 进入时 set、退出时清。
_CURRENT_LEDGER: RunLedger | None = None


def set_current_ledger(ledger: RunLedger | None) -> None:
    global _CURRENT_LEDGER
    _CURRENT_LEDGER = ledger


# --------------------------------------------------------------------------- #
# stage()：运行一个环节，记一条 StageOutcome，吞掉异常（落成 ok=False）。
# --------------------------------------------------------------------------- #
class _Stage:
    """stage() 产出的句柄。环节体把结果写进 .value，门禁结论写进 .ok/.reason。

    默认 ok=True（环节体跑完没抛异常即视为通过）；若有门禁，调用方拿到 value 后
    自行调 gate_* 并 fail()。异常路径由 __exit__ 兜底成 ok=False。
    """

    __slots__ = ("name", "mode_uid", "ok", "reason", "value", "metadata")

    def __init__(self, name: str, mode_uid: str | None) -> None:
        self.name = name
        self.mode_uid = mode_uid
        self.ok = True
        self.reason = ""
        self.value = None
        self.metadata = None       # 环节体可选填的归因信息，透传进 StageOutcome.metadata

    def fail(self, reason: str) -> None:
        self.ok = False
        self.reason = reason


@contextmanager
def stage(ledger: RunLedger | None, name: str, mode_uid: str | None = None):
    """运行一个环节并记录其 StageOutcome；环节体抛出的任何异常都被吞掉并落成失败。

    用法：
        with stage(ledger, "benchmark", mode.uid) as st:
            st.value = run_real_benchmark(mode)        # 抛异常 → st.ok=False，不外泄
        if not st.ok:
            ...收尾...                                  # 调用方据 st.ok 分支
    """
    st = _Stage(name, mode_uid)
    try:
        yield st
    except BaseException as exc:  # noqa: BLE001 - 故意兜住一切，避免炸穿整条 run
        st.ok = False
        st.reason = f"{type(exc).__name__}: {exc}"
    finally:
        _record(ledger, StageOutcome(
            stage=st.name, ok=st.ok, reason=st.reason, mode_uid=st.mode_uid,
            metadata=st.metadata,
        ))


# --------------------------------------------------------------------------- #
# 门禁：纯函数，输入实体 → (ok, reason)。不沾 agent、不抛异常。
# --------------------------------------------------------------------------- #
def gate_strategy(strategies: list[OptimizationStrategy] | None) -> tuple[bool, str]:
    """策略列表非空门禁。空列表此前会让 optimize 的 for 循环静默零次执行。"""
    if not strategies:
        return False, "strategy agent produced no candidates"
    return True, ""


# 数值自检阈值：与 operator.py 保持一致(fp16 舍入噪声约 2e-3 相对，留舒适余量)。
_OPERATOR_REL_ERR_MAX = 5e-2


def gate_operator(artifact: "OperatorArtifact | None") -> tuple[bool, str]:
    """自定义算子门禁：operator-agent 产出的算子是否真可用。

    与 gate_strategy/gate_apply 同为纯函数门禁。算子缺席本身是允许的降级(调用方据此把
    artifact 置 None、apply 退回官方算子)，所以这里 ok=False 不代表 run 失败，只代表
    「这个自定义算子不可用、别当成已验证算子喂给 apply」。三关：拿到 artifact 了吗、
    声称装上了吗、数值自检过关吗。"""
    if artifact is None:
        return False, "operator agent produced no artifact"
    if not artifact.installed:
        return False, "operator not installed (compile/install/register failed)"
    err = artifact.numeric_max_rel_err
    if err is None:
        return False, "operator missing numeric self-check result"
    if err > _OPERATOR_REL_ERR_MAX:
        return False, f"operator numeric error too large: {err:.4g} > {_OPERATOR_REL_ERR_MAX:.4g}"
    return True, ""


def gate_apply(child: ExecutionMode | None, base: ExecutionMode) -> tuple[bool, str]:
    """apply 产出了真 ChangeRecord 门禁——None-record bug 的根因，现在显式拦下。

    apply 在 base.change_log 之上恰好追加一条新记录；若那条是 None（agent 失败），
    说明这次根本没产出可叠加的优化，不该继续递归（也避免 asdict(None) 崩溃）。
    """
    if child is None:
        return False, "apply returned no ExecutionMode"
    if len(child.change_log) <= len(base.change_log):
        return False, "apply produced no new ChangeRecord"
    if child.change_log[-1] is None:
        return False, "apply produced a None ChangeRecord (agent failed to report)"
    return True, ""


# --------------------------------------------------------------------------- #
# agent_call 记录：agent_client 返回 None 前记一条，区分失败种类。
# --------------------------------------------------------------------------- #
def record_agent_call(agent_name: str, status: str, detail: str = "") -> None:
    """记一条 agent_call StageOutcome。status: ok|disabled|timeout|subprocess_error|
    unexpected|agent_error|bad_json。ok=(status=="ok")；永不抛异常、永不影响调用方的
    None 契约。"""
    _record(_CURRENT_LEDGER, StageOutcome(
        stage="agent_call",
        ok=(status == "ok"),
        reason="" if status == "ok" else f"{status}: {detail}".strip(": "),
        metadata={"agent": agent_name, "status": status},
    ))


# --------------------------------------------------------------------------- #
# 落盘
# --------------------------------------------------------------------------- #
def write_ledger(ledger: RunLedger) -> Path:
    _RUNS_DIR.mkdir(parents=True, exist_ok=True)
    safe = ledger.run_uid.replace(":", "_").replace("/", "_")
    path = _RUNS_DIR / f"{safe}.json"
    payload = {
        "run_uid": ledger.run_uid,
        "model_id": ledger.model_id,
        "stop_reason": ledger.stop_reason,
        "best_mode_uid": ledger.best_mode_uid,
        "best_latency": ledger.best_latency,
        "baseline_latency": ledger.baseline_latency,
        "outcomes": [
            {"stage": o.stage, "ok": o.ok, "reason": o.reason,
             "mode_uid": o.mode_uid, "metadata": o.metadata}
            for o in ledger.outcomes
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _record(ledger: RunLedger | None, outcome: StageOutcome) -> None:
    if ledger is not None:
        ledger.outcomes.append(outcome)
