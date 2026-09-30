# 更换 Planner / VLM 模型指南

日期：2026-09-28
相关代码：`tools/apis/llm_endpoint.py`、`entrypoints/run_vsibench_agent.py`、
`entrypoints/collect_vsibench_evidence.py`、`workflow/agentic/planner_loop.py`

## 1. 只有 Planner 一个角色

**Agent 只负责推理。** 它读 JSON 状态、选工具、输出决策，**从来不接触图像**，
所以只需要一个文本模型。

感知全部由专用模型承担：

| 环节 | 模型 |
|---|---|
| 检测 + 分割 | SAM3（或 GroundingDINO + SAM2） |
| 3D 重建 | VGGT |
| metric scale | MoGe |
| 计数 / 尺寸 / 距离 | 由 3D track 和点云几何计算 |

因此配置里只有一个角色：

| 角色 | 环境变量前缀 | 是否需要视觉 |
| --- | --- | --- |
| `planner` | `AGENT_PLANNER_*` | 否 |

> **历史说明**：早期版本还有一个 `vlm` 角色（`AGENT_VLM_*`），用来做
> `verify_candidate` 候选框检查、`--detector vlm` 检测回退和
> `count_entities_in_video` 的 VLM 计数。这三处都已删除：
> `verify_candidate` 工具不存在了，`--detector` 只支持 `grounding_dino` 和
> `sam3`，计数只走 3D track。`llm_endpoint` 也不再认识 `vlm` 角色，
> 误用会直接报错。

## 2. 配置方式

`AGENT_PLANNER_*` 需要 `MODEL` / `BASE_URL` / `API_KEY` 三个变量同时存在。
优先级：

```
planner:  AGENT_PLANNER_*  ->  AGENT_COT_REASONER_*  ->  AGENT_CODE_GENERATOR_*
```

即：不配置 `AGENT_PLANNER_*` 就沿用 `AGENT_COT_REASONER_*`；
只配一半（例如漏了 API key）会自动回退并在启动日志里打印实际来源。

模板见仓库根目录 `API.txt.example`。当前用的是 `qwen3.7-plus`。

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
| 本项目当前选择 | Planner=`qwen3.7-plus`（文本模型） | 见 `API.txt` |
| 本地 vLLM | `Qwen/Qwen3-32B` 等 | 零边际成本；`scripts/launch_agent.sh` 已支持 `BASE_URL='vllm'` 路径 |

建议的验证顺序：

1. 用新 Planner 跑 2~3 道方向题 + 距离题；
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
