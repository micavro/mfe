# MFE (Multi-Flow Execution)

MFE（Multi-Flow Execution）是一个面向**多请求、多推理实例**的 LLM 工作流执行系统。  
它支持用户通过 YAML 定义 DAG 工作流，并以多请求异步方式调度到 GPU 池，实现复杂工作流的高效并行执行，同时输出推理结果与性能统计指标。

## 1. 核心能力

- **YAML 定义 DAG 工作流**：支持多种模板统一执行。
- **多请求异步调度**：多个请求并发进入系统，由调度器统一管理。
- **GPU 池化执行**：将可执行节点下发到空闲 GPU 对应 Worker。
- **端到端状态查询**：支持 `submit` / `status` 两阶段接口。
- **性能可观测**：记录节点级 benchmark、请求时延、吞吐等指标。

## 2. 整体架构

系统包含四个核心角色：

- **Client**：模拟客户端，负责发请求、收结果。
- **Server**：服务进程，负责与 Client 的进程间通信。
- **Optimizer**：调度器，负责 DAG 解析、节点就绪判断、GPU 调度、结果回收。
- **Worker**：单 GPU 执行进程（`vLLM` 实例），执行具体推理任务。

### 请求流转（简化）

1. Client 调用 `submit(template, input)` 提交请求，获得 `uid`。  
2. Server 将请求转给 Optimizer。  
3. Optimizer 解析 DAG，找到 ready 节点，分配给空闲 Worker。  
4. Worker 执行推理，将结果写回 `result_queue`。  
5. Optimizer 更新请求 `op_output` 与 `benchmark`。  
6. Client 通过 `status(uid)` 轮询直到 `completed`，获取最终结果与统计信息。

## 3. 代码结构

```text
mfe/
├── components/            # 组件数据结构定义
│   ├── operator.py        # 操作节点
│   ├── model_config.py    # 模型推理参数
│   ├── query.py           # 请求对象
│   └── exe_info.py        # 执行信息
├── optimizers/            # 调度器
│   └── multi_request.py
├── workers/               # 推理 Worker（vLLM / TestWorker）
│   ├── worker_v.py
│   └── worker_test.py
├── serve/                 # 服务进程
│   └── server.py
├── scripts/               # 用户与测试脚本
│   ├── client.py
│   ├── download_datasets.py
│   ├── process_datasets.py
│   └── benchmark_compare.py
├── templates/             # YAML 工作流模板
│   ├── adv_reason_3.yaml
│   └── adv_reason_4m.yaml
├── data/                  # 数据与结果
├── parser.py              # YAML 模板解析
├── util.py                # 工具函数
└── config.py              # 全局配置
```

## 4. 环境准备

### 4.1 Python 版本

- 推荐 Python `>=3.12`（见 `pyproject.toml`）。

### 4.2 安装依赖

在 `mfe/` 目录下执行：

```bash
pip install -e .
```

> 说明：项目依赖包含 `torch`、`vllm`、`transformers`、`datasets`、`pandas` 等。

### 4.3 模型与模板

- 工作流模板位于 `templates/`，例如 `adv_reason_3.yaml`。
- 模板中的 `model` 字段需要指向你本地可用模型路径（或可访问模型名）。

## 5. 快速运行

以下命令均在 `mfe/` 根目录执行。

### 5.1 下载数据（可选）

```bash
python scripts/download_datasets.py --datasets gsm8k --limit 100
```

会生成类似文件：`data/gsm8k/gsm8k.parquet`。

### 5.2 运行一次多请求测试

```bash
python scripts/client.py --dataset gsm8k -n 20 --yaml adv_reason_3.yaml --send-interval 0.0 -v
```

常用参数：

- `--dataset`：`drop | gsm8k | hotpotqa | math`
- `-n / --num`：测试题目数
- `--yaml`：工作流模板名
- `--send-interval`：请求提交间隔（秒）
- `-v / --verbose`：输出详细日志

输出结果默认写入：`data/{dataset}/{dataset}_{yaml}_result_{n}.json`

### 5.3 无 GPU 调试（TestWorker）

若本地没有可用 GPU，可先用测试 Worker 验证流程：

```bash
python scripts/client.py --dataset gsm8k -n 5 --yaml adv_reason_3.yaml --test-worker --worker-delay 0.2 -v
```

## 6. 结果与评测

- 单次运行结果包含：
  - `mfe_answer`：最终答案
  - `benchmark`：节点执行时间区间
  - `worker_assignments`：节点到 Worker 的分配
  - `latency` / `run_time` / `service_time` / `idle_time` 等指标

- 基准图绘制脚本：

```bash
python data/benchmarks/plot_mfe_benchmark_figures.py
```

图像默认输出到 `data/benchmarks/figures/`。

## 7. 常见问题

- **报错 `No visible GPUs.`**
  - 当前环境没有可见 GPU；可先使用 `--test-worker` 跑通逻辑。

- **模板找不到**
  - 确认 `--yaml` 对应文件存在于 `templates/`，或检查 `--templates-dir` 参数。

- **模型加载失败**
  - 检查模板中的模型路径/名称是否正确，且依赖环境与显存满足要求。

