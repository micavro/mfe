# MFE 系统 API 参考文档

本文档描述 **MFE（Multi-Flow-Execute）** 系统的 API、请求/响应格式、核心类与实现路径。MFE 面向「多工作流、单请求到达即执行、不批处理」场景：每条询问可携带不同的工作流模板（YAML），请求到达即被调度并沿对应 DAG 执行，结果与性能统计按请求返回。

---

## 目录

1. [系统架构与实现路径](#1-系统架构与实现路径)
2. [Server API：请求与响应](#2-server-api请求与响应)
3. [数据结构](#3-数据结构)
4. [工作流 YAML 配置](#4-工作流-yaml-配置)
5. [调度器](#5-调度器)
6. [优化器 OptimizerMFE](#6-优化器-optimizermfe)
7. [Worker 实现](#7-worker-实现)
8. [使用示例与入口](#8-使用示例与入口)
9. [环境变量与性能指标](#9-环境变量与性能指标)
10. [注意事项](#10-注意事项)

---

## 1. 系统架构与实现路径

### 1.1 整体流程

```
Client → request_queue → [Server 进程] → OptimizerMFE.execute_one(query)
                                    → 按 query.template 加载 DAG → schedule_rr
                                    → 向各 Worker 派发任务 (cmd_queues)
                                    → 从各 Worker 收集结果 (result_queues)
                                    → 更新 query.op_output / benchmark
         ← response_queue ← 构造响应 (op_output, benchmark, total_answer_time)
```

- **不批处理**：每次只处理一条请求，执行完整条 DAG 后返回。
- **多工作流**：不同请求可带不同 `template`（YAML 文件名），Optimizer 按需加载并缓存 DAG。

### 1.2 实现路径（代码位置）

| 模块 | 路径 | 说明 |
|------|------|------|
| **Server** | `halo/serve/mfe_server.py` | `run_mfe_server()`：从 request_queue 取请求，调用 OptimizerMFE.execute_one，结果写入 response_queue |
| **Optimizer** | `halo/optimizers/mfe_v.py` | `OptimizerMFE`：管理 Worker 进程与队列，按 template 加载 DAG，`execute_one(query)` 执行单请求 |
| **调度器** | `halo/schedulers/rr.py` | `schedule_rr()`：按拓扑序将 OP 轮询分配到多设备，生成每设备 workflow |
| **Parser** | `halo/parser.py` | `load_config()`、`build_ops_from_config()`：从 YAML 构建 Operator DAG |
| **vLLM Worker** | `halo/workers/worker_v.py` | `vLLMWorker`：绑定 GPU，执行 vLLM 推理，与 Optimizer 通过 cmd/result 队列通信 |
| **Test Worker** | `halo/workers/worker_test.py` | `TestWorker`：不依赖 vLLM/GPU，输出=输入，用于流程验证 |
| **Query / Operator / ExecuteInfo** | `halo/components/` | 请求、图节点、执行信息等数据结构 |
| **测试脚本** | `scripts/test_mfe.py` | 构建请求、启动 Server 子进程、发送请求并统计结果（P50/P95 等） |

---

## 2. Server API：请求与响应

Server 通过 **双队列** 与调用方通信（如 `multiprocessing.Queue`），无需 HTTP。

### 2.1 启动 Server

```python
# 函数签名（halo/serve/mfe_server.py）
def run_mfe_server(
    request_queue,      # 读：请求
    response_queue,     # 写：响应
    templates_dir: str = "templates",
    use_test_worker: bool | None = None,
    verbose: bool = False,
) -> None
```

- **request_queue**：调用方 `put` 请求字典或 `None` / `{"command": "exit"}` 表示结束。
- **response_queue**：Server 对每个请求 `put` 一个响应字典。
- **templates_dir**：工作流 YAML 所在目录（与 `template` 文件名拼接得到路径）。
- **use_test_worker**：为 `True` 时使用 TestWorker；否则从环境变量 `MFE_USE_TEST_WORKER` 读取。
- **verbose**：为 `True` 时打印每条请求/完成的简要日志。

### 2.2 请求格式

每条请求为一个字典，放入 `request_queue`：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `id` | str | 建议 | 请求唯一标识，用于响应与日志 |
| `prompt` | str | 是 | 用户输入文本，作为工作流起点输入 |
| `template` | str | 是 | 工作流 YAML 文件名（如 `adv_reason_3.yaml`）或相对/绝对路径 |

示例：

```python
{"id": "req-001", "prompt": "What is 2+2?", "template": "adv_reason_3.yaml"}
```

### 2.3 响应格式

Server 对每个请求向 `response_queue` 放入一个字典：

**成功时**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | str | 与请求的 `id` 一致 |
| `ok` | bool | `True` |
| `result` | dict | 见下表 |
| `error` | None | 成功时为 `None` |

`result` 内容：

| 字段 | 类型 | 说明 |
|------|------|------|
| `op_output` | dict[str, str] | 各节点 ID 到该请求在该节点输出的映射 |
| `benchmark` | dict[str, [float, float]] | 各节点 ID 到 `[start_time, end_time]` 的映射（秒） |
| `total_answer_time` | float | 从请求创建到最后一个节点结束的秒数 |

**失败时**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | str | 与请求的 `id` 一致 |
| `ok` | bool | `False` |
| `result` | None | 无结果 |
| `error` | str | 错误信息（如 template 不存在、执行异常） |

### 2.4 结束 Server

向 `request_queue` 放入以下任一项即可让 Server 退出并调用 `opt.exit()`：

- `None`
- `{"command": "exit"}`

---

## 3. 数据结构

MFE 复用 Halo 的 `Query`、`Operator`、`ExecuteInfo`、`ModelConfig`、`Benchmark` 等，仅对 MFE 相关用法做摘要。

### 3.1 Query（MFE 相关）

| 属性 | 类型 | 说明 |
|------|------|------|
| `id` | Any | 唯一标识（建议 str/UUID） |
| `prompt` | str | 用户输入 |
| `template` | str | 工作流 YAML 文件名或路径，决定使用哪个 DAG |
| `op_output` | dict[str, str] | 执行过程中各 op_id → 该节点输出 |
| `step` | int | 已完成的 OP 数量 |
| `create_time` | float | 创建时 `time.perf_counter()`，用于算 total_answer_time |
| `benchmark` | dict[str, (float, float)] | op_id → (start, end) 时间戳 |

**实现路径**：`halo/components/query.py`

### 3.2 Operator

工作流图节点，含 `id`、`input_ops`、`output_ops`、`model_config`、`benchmark` 等。由 parser 从 YAML 构建。

**实现路径**：`halo/components/operator.py`

### 3.3 ExecuteInfo

Optimizer 发给 Worker 的单次执行任务：

| 属性 | 类型 | 说明 |
|------|------|------|
| `op` | Operator | 要执行的节点 |
| `query_ids` | list | 本批查询的 ID 列表（MFE 单请求时通常长度为 1） |
| `prompts` | list[str] | 与 query_ids 一一对应，已含父 OP 输出拼接 |

**实现路径**：`halo/components/exe_info.py`

---

## 4. 工作流 YAML 配置

模板文件放在 `templates_dir` 下，由 `halo/parser.py` 的 `load_config` + `build_ops_from_config` 解析。**MFE 不在 YAML 中写数据集**，数据集由测试脚本或调用方在外部指定。

### 4.1 配置结构

```yaml
start_ops: [<op_id>, ...]   # 入口节点
end_ops:   [<op_id>, ...]   # 出口节点

ops:
  <op_id>:
    model: <str>                    # 必填，模型名（如 HuggingFace ID）
    input_ops: [<op_id>, ...]       # 可选，前驱
    output_ops: [<op_id>, ...]      # 可选，后继
    prompt: <str>                    # 可选，系统/角色提示
    temperature: <float>             # 默认 0.7
    top_p: <float>                   # 默认 0.9
    max_tokens: <int>                # 默认 256
    dtype: "bfloat16" | "float16"    # 默认 "bfloat16"
    use_chat_template: <bool>        # 默认 False（parser 中常为 True）
    # 其他可选：max_model_len, quantization, lora_config, min_tokens, keep_cache 等
```

### 4.2 实现路径

- 加载：`halo/parser.py` → `load_config(path)` → `build_ops_from_config(config)`
- Optimizer 内按 template 解析路径：`halo/optimizers/mfe_v.py` → `_resolve_template_path(template)`、`_get_dag(template)`，并带缓存 `_template_cache`

---

## 5. 调度器

MFE 当前**仅使用轮询调度**。

### 5.1 schedule_rr

| 项目 | 说明 |
|------|------|
| **位置** | `halo/schedulers/rr.py` |
| **签名** | `schedule_rr(device_cnt: int, all_ops: List[Operator], queries: List[Query]) -> List[List[Dict]]` |
| **含义** | 对 OP 做拓扑排序，按序轮询分配到 `device_cnt` 个设备；每个设备得到一份 workflow，即 `List[Dict]`，每项为 `{"command": "execute", "params": (op, query_ids)}` |
| **复杂度** | O(V+E) |

Optimizer 只支持 `strategy="rr"`，其他策略会 `ValueError`。

---

## 6. 优化器 OptimizerMFE

### 6.1 类与构造

**位置**：`halo/optimizers/mfe_v.py`

```python
class OptimizerMFE:
    def __init__(
        self,
        templates_dir: str = "templates",
        use_test_worker: bool | None = None,
        verbose: bool = False,
        **kwargs
    ) -> None
```

- **templates_dir**：工作流 YAML 根目录。
- **use_test_worker**：为 `True` 使用 TestWorker；`None` 时从环境变量 `MFE_USE_TEST_WORKER` 读取。
- **verbose**：为 `True` 时打印 DAG、派发/回收日志。

构造时会：根据 `torch.cuda.device_count()` 与 `_visible_physical_gpu_ids()` 创建若干 Worker 进程及每进程的 `cmd_queues[i]`、`result_queues[i]`；若 `use_test_worker` 且 GPU 数为 0，则设备数按 1 处理。

### 6.2 主要方法

| 方法 | 说明 |
|------|------|
| `_resolve_template_path(template)` | 将 template 解析为绝对路径，不存在则抛 `FileNotFoundError` |
| `_get_dag(template)` | 按 template 加载并缓存 DAG，返回 `(ops, start_ops, end_ops, models)` |
| `schedule(queries, strategy="rr")` | 仅支持 `"rr"`，调用 `schedule_rr` 得到 `self.workflows` |
| `execute_one(query: Query) -> Query` | 用 `query.template` 的 DAG 执行该请求：_get_dag → schedule → 按 workflows 向 Worker 派发任务、收集结果，更新 `query.op_output`、`query.benchmark`，返回该 Query |
| `exit()` | 向所有 Worker 发 exit，关闭队列并 join 进程 |

### 6.3 与 Worker 的通信

- **下发给 Worker**：`cmd_queues[i].put(task)`，task 为经 `_cmd_transfer` 后的字典，如 `{"command": "execute", "params": (ExecuteInfo,)}`。
- **从 Worker 接收**：`result_queues[i].get(timeout=0.1)`，期望格式 `{"command": "execute"|"error"|"exit", "result": ...}`；`execute` 的 result 需含 `op_name`、`item`（每项含 `id`、`output`、`benchmark`）、以及可选的 `benchmark` 聚合。

---

## 7. Worker 实现

### 7.1 对比

| Worker | 类名 | 说明 | 使用场景 |
|--------|------|------|----------|
| **vLLM Worker** | `vLLMWorker` | 绑定 GPU，用 vLLM 做推理；按 OP 切换模型，支持 chat template | 生产/压测，需真实推理 |
| **Test Worker** | `TestWorker` | 不加载模型，`execute` 时直接返回输入 prompt 作为 output | 无 GPU 或未装 vLLM 时验证流程 |

### 7.2 统一接口（两者一致）

| 方法/行为 | 说明 |
|-----------|------|
| `__init__(id, physical_gpu_id, cmd_queue, result_queue)` | 初始化；TestWorker 不依赖 GPU |
| `run(debug=True)` | 主循环：从 cmd_queue 取消息，解析 command/params，调用对应方法，向 result_queue 写 `{"command", "result", "elapsed_time"}` |
| `execute(exe_info: ExecuteInfo) -> dict` | 执行一批；返回 `{"item": [{"id", "output", "benchmark": (start,end)}, ...], "op_name": str, "benchmark": {"init_time", "prefill_time", "generate_time"}}` |
| `exit() -> str` | 清理资源并返回退出信息 |

**实现路径**：`halo/workers/worker_v.py`（vLLM）、`halo/workers/worker_test.py`（Test）；导出在 `halo/workers/__init__.py`。

---

## 8. 使用示例与入口

### 8.1 测试脚本（推荐入口）

```bash
# 使用 TestWorker，5 条请求，详细日志
uv run python scripts/test_mfe.py -n 5 --test-worker -v

# 使用真实 vLLM Worker（需 GPU）
uv run python scripts/test_mfe.py -n 5 -v
```

**脚本路径**：`scripts/test_mfe.py`  
功能：从 `--data-dir`（默认 `data/`）或 HuggingFace 准备数据（JSONL），构建 `[prompt, template]` 请求列表；启动 Server 子进程（`run_mfe_server`），顺序发请求、收响应；统计 Total answer time 与各 Node 的 P50/P95。

### 8.2 在代码中调用 Server

```python
import multiprocessing as mp
from halo.serve.mfe_server import run_mfe_server

request_queue = mp.Queue()
response_queue = mp.Queue()

server_proc = mp.Process(
    target=run_mfe_server,
    args=(request_queue, response_queue, "templates", False, False),
    daemon=False,
)
server_proc.start()

# 发请求
request_queue.put({"id": "1", "prompt": "Question?", "template": "adv_reason_3.yaml"})
resp = response_queue.get()
# resp["ok"], resp["result"]["op_output"], resp["result"]["total_answer_time"], ...

# 结束
request_queue.put(None)
server_proc.join()
```

### 8.3 直接使用 OptimizerMFE（单进程）

```python
from halo.components import Query
from halo.optimizers.mfe_v import OptimizerMFE

opt = OptimizerMFE(templates_dir="templates", use_test_worker=True)
query = Query(id="q1", prompt="What is 2+2?", template="adv_reason_3.yaml")
query = opt.execute_one(query)
print(query.op_output, query.benchmark)
opt.exit()
```

---

## 9. 环境变量与性能指标

### 9.1 环境变量

| 变量 | 含义 | 取值 |
|------|------|------|
| `MFE_USE_TEST_WORKER` | 是否使用 TestWorker | `1` / `true` / `yes` 时使用 TestWorker |
| `MFE_VERBOSE` | 是否打印详细中间日志（Server/Opt/Worker） | `1` / `true` / `yes` 时开启 |
| `KMP_DUPLICATE_LIB_OK` | 允许多份 OpenMP 运行时（常见于 Windows） | `TRUE` 可避免 libiomp5 冲突 |
| `CUDA_VISIBLE_DEVICES` | 可见 GPU（Optimizer 用 `_visible_physical_gpu_ids()` 读取） | 如 `0,1` |

### 9.2 性能指标

- **total_answer_time**：从 `query.create_time` 到该请求最后一个节点 `benchmark[op_id][1]` 的差值（秒）。
- **benchmark 各节点**：每个节点 `(start_time, end_time)` 由 Worker 在 execute 内记录并写回；可得到各节点耗时与顺序。
- **P50 / P95**：测试脚本对「总答案时间」及「各节点耗时」做 50%、95% 分位数统计。

---

## 10. 注意事项

1. **template 与 DAG**：每条请求的 `template` 决定使用的 YAML；同一 template 的 DAG 会被缓存，不同 template 可并存。
2. **单请求不批处理**：MFE 每次只执行一个 Query 的整条 DAG，不做多请求 batching。
3. **Worker 常驻**：`execute_one` 执行期间不向 Worker 发 exit；仅在 Server 退出时 `opt.exit()` 关闭 Worker。
4. **通信协议**：Python 内置多进程队列（如 `multiprocessing.Queue`），无 HTTP；若需 HTTP 需在外部再包一层。
5. **TestWorker**：仅用于流程与接口验证；真实延迟与吞吐需使用 vLLM Worker。

---

**文档版本**: 1.0  
**最后更新**: 2025-02
