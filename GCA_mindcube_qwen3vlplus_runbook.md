# MindCube 小样本运行手册：Qwen3-VL-Plus API

## 已完成修改

```text
tools/utils/vlm_as_detector.py
- 已加入 qwen3-vl-plus 和 qwen/qwen3-vl-plus

entrypoints/agent.py
- 已加入 --limit 参数
- 已加入 benchmark 子集截取逻辑

scripts/gca_env.sh
- 已配置 HF 缓存、U2Net、EasyOCR、Numba、Matplotlib 可写缓存
```

## 1. 激活环境

```bash
cd /data3/Agentic-Spatial-Reasoning/gca-main
conda activate gca
source scripts/gca_env.sh
```

## 2. 配置 API

将下面内容按实际 API 修改：

```bash
export AGENT_COT_REASONER_MODEL='qwen3-vl-plus'
export AGENT_COT_REASONER_BASE_URL='https://your-api.example.com/v1'
export AGENT_COT_REASONER_API_KEY='your-api-key'
export AGENT_COT_REASONER_PROXY=''

export AGENT_CODE_GENERATOR_MODEL='qwen3-vl-plus'
export AGENT_CODE_GENERATOR_BASE_URL='https://your-api.example.com/v1'
export AGENT_CODE_GENERATOR_API_KEY='your-api-key'
export AGENT_CODE_GENERATOR_PROXY=''
```

如果 API 的模型名是 `Qwen/Qwen3-VL-Plus`，就填写完整名称。

## 3. 检查数据

```bash
test -f data/mindcube/raw/MindCube_tinybench.jsonl && echo DATA_OK
wc -l data/mindcube/raw/MindCube_tinybench.jsonl
```

## 4. 运行 1 个 rotation 样本

```bash
python -m entrypoints.agent \
  --benchmark mindcube \
  --question_type rotation \
  --limit 1 \
  --concurrency 1 \
  --work_dir work_dir/mindcube_qwen3vlplus_1
```

## 5. 运行 10 个 rotation 样本

```bash
python -m entrypoints.agent \
  --benchmark mindcube \
  --question_type rotation \
  --limit 10 \
  --concurrency 1 \
  --work_dir work_dir/mindcube_qwen3vlplus_10
```

## 6. 运行 around 和 among

```bash
python -m entrypoints.agent \
  --benchmark mindcube \
  --question_type around \
  --limit 10 \
  --concurrency 1 \
  --work_dir work_dir/mindcube_around_10

python -m entrypoints.agent \
  --benchmark mindcube \
  --question_type among \
  --limit 10 \
  --concurrency 1 \
  --work_dir work_dir/mindcube_among_10
```

## 7. 查看结果

```bash
cat work_dir/mindcube_qwen3vlplus_1/predictions.jsonl
cat work_dir/mindcube_qwen3vlplus_1/results_summary.json
```

查看 session：

```bash
find work_dir/mindcube_qwen3vlplus_1 \
  -name 'trace.jsonl' -o -name 'msg.jsonl'
```

HTML 报告：

```text
work_dir/mindcube_qwen3vlplus_1/session-<sample_id>/session_report.html
```

## 8. 恢复中断任务

```bash
python -m entrypoints.agent \
  --benchmark mindcube \
  --question_type rotation \
  --limit 10 \
  --concurrency 1 \
  --resume \
  --work_dir work_dir/mindcube_qwen3vlplus_10
```
