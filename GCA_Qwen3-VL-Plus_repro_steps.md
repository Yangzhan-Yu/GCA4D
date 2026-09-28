# GCA 复现步骤：Qwen3-VL-Plus API + MindCube 小样本

## 1. 启动环境

```bash
cd /data3/Agentic-Spatial-Reasoning/gca-main
conda activate gca
source scripts/gca_env.sh
```

确认 GPU：

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

## 2. 配置 Qwen3-VL-Plus API

模型名按 API 服务商的实际 `model id` 填写。

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

如果服务商模型名是 `Qwen/Qwen3-VL-Plus` 或 `Qwen3-VL-Plus`，就使用其准确名称。

检查 API：

```bash
curl -sS "$AGENT_COT_REASONER_BASE_URL/chat/completions" \
  -H "Authorization: Bearer $AGENT_COT_REASONER_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$AGENT_COT_REASONER_MODEL\",
    \"messages\": [{\"role\":\"user\",\"content\":\"Reply OK\"}],
    \"max_tokens\": 8
  }"
```

## 3. 注册 Qwen3-VL-Plus 为目标检测模型

编辑：

```text
tools/utils/vlm_as_detector.py
```

在 `VLM_AS_DETECTOR` 中加入：

```python
'qwen3-vl-plus': (qwen3_prompt, qwen3_parse_detection),
'qwen/qwen3-vl-plus': (qwen3_prompt, qwen3_parse_detection),
```

这样可以避免额外依赖 GroundingDINO。

## 4. 增加小样本参数 `--limit`

编辑：

```text
entrypoints/agent.py
```

增加参数：

```python
parser.add_argument('--limit', type=int, default=None)
```

创建 benchmark 后增加：

```python
benchmark = BenchmarkFactory.create_benchmark(
    benchmark_name=config.benchmark,
    question_type=config.question_type,
)

if args.limit is not None:
    if args.limit <= 0:
        raise ValueError('--limit must be greater than 0')
    benchmark.data = benchmark.data.iloc[:args.limit].copy()
```

## 5. 下载 MindCube 数据集

```bash
export GCA_ROOT=/data3/Agentic-Spatial-Reasoning/gca-main
export TMP_MIND=/data3/Agentic-Spatial-Reasoning/tmp_mindcube

mkdir -p "$TMP_MIND"
```

如果 `hf` 命令不存在：

```bash
python -m pip install -U "huggingface_hub[cli]"
```

下载：

```bash
hf download MLL-Lab/MindCube \
  --repo-type dataset \
  --local-dir "$TMP_MIND"
```

如果 Hugging Face 不通：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

解压：

```bash
cd "$TMP_MIND"
unzip data.zip
```

移动到 GCA 数据目录：

```bash
mkdir -p "$GCA_ROOT/data/mindcube"

mv "$TMP_MIND/data/raw" "$GCA_ROOT/data/mindcube/"
mv "$TMP_MIND/data/other_all_image" "$GCA_ROOT/data/mindcube/"
```

确认结构：

```text
data/mindcube/
├── raw/
│   └── MindCube_tinybench.jsonl
└── other_all_image/
    ├── around/
    ├── among/
    └── rotation/
```

检查：

```bash
cd "$GCA_ROOT"

wc -l data/mindcube/raw/MindCube_tinybench.jsonl
find data/mindcube/other_all_image -maxdepth 2 -type d | sort
```

## 6. 先运行 1 个样本

```bash
cd /data3/Agentic-Spatial-Reasoning/gca-main

python -m entrypoints.agent \
  --benchmark mindcube \
  --question_type rotation \
  --limit 1 \
  --concurrency 1 \
  --work_dir work_dir/mindcube_qwen3vlplus_1
```

## 7. 再运行 10 个样本

```bash
python -m entrypoints.agent \
  --benchmark mindcube \
  --question_type rotation \
  --limit 10 \
  --concurrency 1 \
  --work_dir work_dir/mindcube_qwen3vlplus_10
```

稳定后扩大：

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

## 8. 检查结果

```bash
cat work_dir/mindcube_qwen3vlplus_1/predictions.jsonl
cat work_dir/mindcube_qwen3vlplus_1/results_summary.json
```

查看日志：

```bash
find work_dir/mindcube_qwen3vlplus_1 \
  -name 'trace.jsonl' -o -name 'msg.jsonl'
```

查看报告：

```text
work_dir/mindcube_qwen3vlplus_1/session-<sample_id>/session_report.html
```

## 9. 常见检查点

```text
serve.json not found
-> base_url 不要填 vllm

模型不支持 image_url
-> Qwen3-VL-Plus API 不具备多图视觉能力

要求 GroundingDINO
-> 检查 VLM_AS_DETECTOR 是否正确加入模型名

API 报 temperature/top_p/max_tokens
-> 在 API 网关中做参数兼容

启动失败或 CUDA OOM
-> 先使用 --concurrency 1

单样本成功后
-> 再逐步增加样本数量和并发
```

## 10. 后续完整复现顺序

```text
1. MindCube-tiny rotation 小样本
2. MindCube-tiny around/among 小样本
3. MindCube-tiny 全部样本
4. MMSI-Bench
5. OmniSpatial
6. SPBench
7. CV-Bench
8. CoT baseline 对照
```
