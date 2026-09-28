# 更换 Planner / VLM 模型指南

日期：2026-09-28
相关代码：`tools/apis/llm_endpoint.py`、`entrypoints/run_vsibench_agent.py`、
`entrypoints/collect_vsibench_evidence.py`、`workflow/agentic/planner_loop.py`

## 1. 为什么要拆成两个角色

原实现只有一个模型（`AGENT_COT_REASONER_*`），Planner 和所有视觉调用共用它。
Qwen3-VL-235B-A22B-Thinking 因此承担了它并不需要承担的工作。

现在拆成两个逻辑角色：

| 角色 | 环境变量前缀 | 是否需要视觉 | 用途 |
| --- | --- | --- | --- |
| `planner` | `AGENT_PLANNER_*` | 否 | 证据规划、工具选择、输出 JSON 决策 |
| `vlm` | `AGENT_VLM_*` | 是 | `verify_candidate` 候选框检查、`--detector vlm` 检测回退、`count_entities_in_video` 的 VLM 路径 |

**Planner 完全不看像素。** 它只看到 JSON 形式的场景状态、约束、工具列表和工具返回值。
所以 Planner 可以是任意一个 OpenAI 兼容的纯文本模型，不需要 VL 能力。

视觉部分在现在的默认链路里已经很少：

- 检测用 GroundingDINO / SAM3；
- 分割用 SAM2；
- 3D 用 VGGT，尺度用 MoGe；
- Qwen 只在 `verify_candidate` 检查少量可疑 track，以及显式开启时才做多帧计数。

所以 `vlm` 角色可以用比 235B 小得多的模型。

## 2. 配置方式

`AGENT_PLANNER_*` 和 `AGENT_VLM_*` 各自需要 `MODEL` / `BASE_URL` / `API_KEY`
三个变量同时存在才算有效。优先级：

```
planner:  AGENT_PLANNER_*  ->  AGENT_COT_REASONER_*  ->  AGENT_CODE_GENERATOR_*
vlm:      AGENT_VLM_*      ->  AGENT_COT_REASONER_*  ->  AGENT_CODE_GENERATOR_*
```

即：

- **不配置新变量** → 行为与现在完全一致，两个角色都用 `AGENT_COT_REASONER_*`；
- **只配置 `AGENT_PLANNER_*`** → Planner 换成新模型，视觉仍用旧模型；
- **某个新角色变量只写了一半**（例如漏了 API key）→ 自动回退到 `AGENT_COT_REASONER_*`，
  并在启动日志里打印实际生效的来源。

模板见仓库根目录 `API.txt.example`。

### 只换 Planner（最省钱的改法）

在 `API.txt` 里追加：

```bash
export AGENT_PLANNER_MODEL='qwen3-max'
export AGENT_PLANNER_BASE_URL='https://dashscope.aliyuncs.com/compatible-mode/v1'
export AGENT_PLANNER_API_KEY='sk-...'
```

保留原来的 `AGENT_COT_REASONER_*` 给视觉用。

### 两个角色都换

```bash
export AGENT_PLANNER_MODEL='qwen3-max'
export AGENT_PLANNER_BASE_URL='...'
export AGENT_PLANNER_API_KEY='sk-...'

export AGENT_VLM_MODEL='qwen3-vl-plus'
export AGENT_VLM_BASE_URL='...'
export AGENT_VLM_API_KEY='sk-...'
```

启动日志会打印：

```
[Agent] Planner endpoint: {"role": "planner", "source": "AGENT_PLANNER_*", "model": "qwen3-max", ...}
[Agent] VLM endpoint    : {"role": "vlm", "source": "AGENT_VLM_*", "model": "qwen3-vl-plus", ...}
```

API key 只显示末四位，不会完整写入日志或 `agent_result.json`。

## 3. 兼容性与自动降级

不同厂商对参数支持不一致，`llm_endpoint` 会自动处理：

| 情况 | 处理方式 |
| --- | --- |
| 模型不支持 `temperature` | 捕获 400 错误，去掉该参数重试一次 |
| 模型不支持 `top_p` | 同上 |
| 只接受 `max_completion_tokens` | 自动改用该字段重试 |
| 答案放在 `reasoning_content`、`content` 为空 | 从 `reasoning_content` 里恢复最后一个 JSON 块 |
| 思考内联为 `<think>...</think>` | 自动剥离，只取思考后的正文 |
| 返回结构化 content parts | 拼接为纯文本 |
| 模型返回空内容 | 明确抛错，`agent_result.json` 记录 `Planner API call failed` |

注意：部分模型别名（例如 `qwen-plus`、`qwen3.7-plus`）**首次调用冷启动很慢**，
实测出现过 90~120s 无响应。默认超时已从 90s 提高到 180s，仍不够时：

```bash
export GCA_LLM_TIMEOUT=300
```

冷启动只影响第一次调用，之后通常恢复到正常速度。若长时间不恢复，可改用带日期的
固定快照（例如 `qwen3.7-plus-2026-05-26`）。

可用环境变量微调超时与重试：

```bash
export GCA_LLM_TIMEOUT=180     # 单次请求超时（秒），默认 180
export GCA_LLM_MAX_RETRIES=2   # 传输层重试次数，默认 2
```

## 4. 选型建议

Planner 的任务是「读 JSON 状态、选工具、输出固定 schema 的 JSON」，
不需要 235B 级别的模型。建议按以下顺序试：

| 类型 | 候选 | 说明 |
| --- | --- | --- |
| 国内 API（百炼兼容模式） | `qwen-flash`、`qwen-plus` | 最便宜，先测 JSON 稳定性 |
| 国内 API（推理更强） | `qwen3-max`、`deepseek-v3` 系列、`glm-4.5` | 工具选择更稳，单价仍远低于 235B-Thinking |
| 本项目当前选择 | Planner=`qwen3.7-plus`，VLM=`qwen3-vl-plus` | 见 `API.txt` |
| 本地 vLLM | `Qwen/Qwen3-32B` 等 | 零边际成本；`scripts/launch_agent.sh` 已支持 `BASE_URL='vllm'` 路径 |
| 视觉角色 | `qwen3-vl-plus`、`qwen3-vl-30b-a3b` | 只在候选框检查时调用，成本占比低 |

建议的验证顺序：

1. 固定 `vlm` 不变，只用新 Planner 跑 2~3 道方向题 + 距离题；
2. 检查 `agent_result.json` 里的 `steps`、`api_budget.used`、`verification` 是否正常；
3. 确认 JSON 决策稳定后，再扩大题型。

## 5. 换模型后需要重点看什么

Planner 换小模型最常见的失败是**不按 schema 输出**，而不是算错。
所以换模型后先看这几项：

- `agent_result.json` 的 `done` 是否为 `true`；
- `operation_results` 里是否有 `verification_status: verified` 的条目；
- `steps` 是否比原来明显增加（说明决策反复）；
- `api_budget.used` 是否接近上限（说明陷入循环）。

注意：约束层的 `make_done_validator` 会拦住没有已验证结果的 `done`。
新 Planner 如果不会调用 `execute_operation`，表现为轮次用尽后 `done=false`，
而不是给出错误答案——这是预期行为，便于定位是策略问题而非计算问题。

## 6. 相关测试

```bash
# 角色解析 / 回退 / 响应提取 / 参数降级（9 项，需要 gca 环境）
source scripts/gca_env.sh
python tests/test_llm_endpoint.py
```

覆盖：legacy 配置服务两个角色、角色变量覆盖、半配置回退、key 脱敏、
`reasoning_content` 恢复、`<think>` 剥离、`temperature` / `max_tokens` 降级重试。
