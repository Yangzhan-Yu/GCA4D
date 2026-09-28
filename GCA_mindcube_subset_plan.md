# GCA 小样本复现方案：MindCube-tiny

本文档用于在 `/data3/Agentic-Spatial-Reasoning/gca-main` 中，通过少量 MindCube-tiny 样本熟悉 GCA 的完整推理和评测流程。

当前目标不是复现论文总表，而是确认以下链路能够正常工作：

```text
数据加载
-> VLM 任务形式化
-> 工具编排
-> VGGT / SAM2 / Orient-Anything 等视觉工具调用
-> Python 几何计算
-> 最终答案
-> 自动评测和日志保存
```

---

## 1. 为什么选择 MindCube-tiny

论文在 5 个 benchmark 上评测：

- MMSI-Bench
- MindCube-tiny
- OmniSpatial
- SPBench
- CV-Bench

首个复现实验建议选择 **MindCube-tiny**，原因如下：

1. MindCube-tiny 本身就是 MindCube 的小规模子集。
2. 它直接测试多视图空间建模、视角变换和场景重建，是 GCA 最核心的能力。
3. 默认配置会覆盖：
   - SemanticDetector
   - GeometricReconstructor
   - ObjPoseEstimator
   - PythonTool
   - LanguageToCamera
4. 默认 `config/agent_mindcube.json` 不依赖 EasyOCR，也不启用 MetricScaleEstimator。
5. 先跑 `rotation` 子任务，再扩展到 `around` 和 `among`。

建议的第一阶段运行规模：

```text
Benchmark:     mindcube
Question type: rotation
Samples:       1 -> 5 -> 10
Concurrency:   1
```

---

## 2. 当前环境检查结果

### 2.1 已确认正常的项目

Conda 环境：

```text
/data2/conda_envs/gca
```

关键版本：

```text
Python                 3.11.16
PyTorch                2.5.1
torchvision            0.20.1
torchaudio             2.5.1
PyTorch CUDA           12.4
ray                    2.48.0
langgraph              0.6.6
transformers           4.55.2
easyocr                1.7.2
opencv-python          4.11.0
numpy                  1.26.4
pandas                 2.3.1
pyarrow                21.0.0
```

第三方仓库均已存在：

```text
tools/third_party/vggt
tools/third_party/sam2
tools/third_party/Orient-Anything
tools/third_party/MoGe
```

模型权重均已存在：

```text
/data3/Agentic-Spatial-Reasoning/hf_cache/hub/models--facebook--VGGT-1B
/data3/Agentic-Spatial-Reasoning/hf_cache/hub/models--Ruicheng--moge-2-vitl-normal
/data3/Agentic-Spatial-Reasoning/hf_cache/hub/models--Viglong--Orient-Anything
/data3/Agentic-Spatial-Reasoning/hf_cache/hub/models--facebook--dinov2-large
```

SAM2 checkpoint：

```text
tools/third_party/sam2/checkpoints/sam2.1_hiera_large.pt
```

U2Net：

```text
/data3/Agentic-Spatial-Reasoning/u2net/u2net.onnx
```

EasyOCR：

```text
/data3/Agentic-Spatial-Reasoning/easyocr/model/craft_mlt_25k.pth
/data3/Agentic-Spatial-Reasoning/easyocr/model/zh_sim_g2.pth
```

离线 Hugging Face 缓存加载测试全部通过：

```text
VGGT                 OK
MoGe-2               OK
Orient-Anything      OK
DINOv2 config        OK
DINOv2 processor     OK
```

### 2.2 已修复的问题

预检时 `rembg` 和 `tools.apis` 导入失败：

```text
RuntimeError: cannot cache function '_make_tree':
no locator available for file
```

原因是 Numba、Matplotlib、Fontconfig 默认缓存目录不可写。

已在 `scripts/gca_env.sh` 中增加仓库内可写缓存：

```text
GCA_RUNTIME_CACHE=/data3/Agentic-Spatial-Reasoning/gca-main/.cache
XDG_CACHE_HOME=$GCA_RUNTIME_CACHE/xdg
MPLCONFIGDIR=$GCA_RUNTIME_CACHE/matplotlib
NUMBA_CACHE_DIR=$GCA_RUNTIME_CACHE/numba
PYTHONPYCACHEPREFIX=$GCA_RUNTIME_CACHE/pycache
```

修复后重新验证：

```text
workflow.config     OK
workflow.workflow   OK
tools.apis          OK
vggt.models.vggt    OK
sam2.build_sam      OK
moge.model.v2       OK
easyocr             OK
rembg               OK
```

以后每次运行前都应执行：

```bash
cd /data3/Agentic-Spatial-Reasoning/gca-main
conda activate gca
source scripts/gca_env.sh
```

### 2.3 仍需确认或补充的项目

#### GPU 可见性

Codex 当前沙箱中没有 `/dev/nvidia*`，因此这里得到：

```text
torch.cuda.is_available() = False
torch.cuda.device_count() = 0
```

这不代表服务器真实终端一定没有 GPU。请在普通服务器终端执行：

```bash
nvidia-smi
conda activate gca
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

预期应看到 GPU 列表，并且：

```text
True
1 或更大
```

如果仍然是 `False`，必须先解决 GPU 可见性，否则 VGGT、SAM2、Orient-Anything 和 EasyOCR 都无法按默认配置运行。

#### API/VLM 配置

当前 `config/agent_*.json` 只配置了 benchmark 相关参数，没有写死模型 API。运行前需要设置：

```bash
export AGENT_COT_REASONER_MODEL='Qwen/Qwen3-VL-235B-A22B-Thinking'
export AGENT_COT_REASONER_BASE_URL='https://your-api.example.com/v1'
export AGENT_COT_REASONER_API_KEY='sk-REPLACE_ME'
export AGENT_COT_REASONER_PROXY=''

export AGENT_CODE_GENERATOR_MODEL='Qwen/Qwen3-VL-235B-A22B-Thinking'
export AGENT_CODE_GENERATOR_BASE_URL='https://your-api.example.com/v1'
export AGENT_CODE_GENERATOR_API_KEY='sk-REPLACE_ME'
export AGENT_CODE_GENERATOR_PROXY=''
```

如果使用多模态代理服务，模型名可能不是完整的 `Qwen/Qwen3-VL-235B-A22B-Thinking`。此时需要注意：

```text
Qwen/Qwen3-VL-235B-A22B-Thinking
zai-org/GLM-4.5V-FP8
```

这两个名字默认可以直接走 VLM-as-detector，不需要 GroundingDINO。

如果 API 只接受其他短名，应当：

1. 将短名加入 `tools/utils/vlm_as_detector.py`；或
2. 下载 GroundingDINO，并关闭 VLM-as-detector。

#### 数据集

当前仓库还没有：

```text
data/
```

MindCube 数据需要补充。

---

## 3. 准备 MindCube-tiny 数据

数据集页面：

```text
https://huggingface.co/datasets/MLL-Lab/MindCube
```

建议将临时下载目录放在：

```text
/data3/Agentic-Spatial-Reasoning/tmp_mindcube
```

下载并解压：

```bash
export GCA_ROOT=/data3/Agentic-Spatial-Reasoning/gca-main
export TMP_MIND=/data3/Agentic-Spatial-Reasoning/tmp_mindcube

mkdir -p "$TMP_MIND"
hf download MLL-Lab/MindCube \
  --repo-type dataset \
  --local-dir "$TMP_MIND"

cd "$TMP_MIND"
unzip data.zip
```

建立 GCA 需要的目录：

```bash
mkdir -p "$GCA_ROOT/data/mindcube"

mv "$TMP_MIND/data/raw" "$GCA_ROOT/data/mindcube/"
mv "$TMP_MIND/data/other_all_image" "$GCA_ROOT/data/mindcube/"
```

最终结构应为：

```text
data/mindcube/
├── raw/
│   ├── MindCube.jsonl
│   ├── MindCube_train.jsonl
│   └── MindCube_tinybench.jsonl
└── other_all_image/
    ├── around/
    ├── among/
    └── rotation/
```

检查数据：

```bash
wc -l data/mindcube/raw/MindCube_tinybench.jsonl

find data/mindcube/other_all_image -maxdepth 2 -type d | sort
```

评测代码固定读取：

```text
data/mindcube/raw/MindCube_tinybench.jsonl
```

因此不要只复制 `MindCube.jsonl`。

---

## 4. 增加 `--limit` 小样本参数

当前 `entrypoints/agent.py` 没有 `--limit`，只能按 `question_type` 过滤，不能直接限制样本数量。

为了只跑 1～10 个样本，建议增加：

```python
parser.add_argument('--limit', type=int, default=None)
```

在创建 benchmark 之后、开始 resume 和调度之前加入：

```python
if args.limit is not None:
    if args.limit <= 0:
        raise ValueError('--limit must be greater than 0')
    if not hasattr(benchmark, 'data') or not hasattr(benchmark.data, 'iloc'):
        raise TypeError('--limit currently requires a pandas-backed benchmark')
    benchmark.data = benchmark.data.iloc[:args.limit].copy()
```

位置示意：

```python
benchmark = BenchmarkFactory.create_benchmark(...)
if args.limit is not None:
    benchmark.data = benchmark.data.iloc[:args.limit].copy()
```

这种实现会让：

- `len(benchmark)`
- `benchmark.__getitem__`
- `benchmark.evaluate`

都只看到前 N 个样本，避免把未运行的样本算成错误答案。

该改动只用于小样本调试，不会影响完整数据集运行。

---

## 5. 启动前的 API 连通性检查

先加载统一环境：

```bash
cd /data3/Agentic-Spatial-Reasoning/gca-main
conda activate gca
source scripts/gca_env.sh
```

然后设置 API 环境变量，并检查变量：

```bash
python - <<'PY'
import os
for name in [
    'AGENT_COT_REASONER_MODEL',
    'AGENT_COT_REASONER_BASE_URL',
    'AGENT_COT_REASONER_API_KEY',
    'AGENT_CODE_GENERATOR_MODEL',
    'AGENT_CODE_GENERATOR_BASE_URL',
    'AGENT_CODE_GENERATOR_API_KEY',
]:
    value = os.environ.get(name)
    print(name, '<set>' if value else '<unset>')
PY
```

测试 API：

```bash
curl -sS "$AGENT_COT_REASONER_BASE_URL/chat/completions" \
  -H "Authorization: Bearer $AGENT_COT_REASONER_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$AGENT_COT_REASONER_MODEL\",
    \"messages\": [{\"role\": \"user\", \"content\": \"Reply OK\"}],
    \"max_tokens\": 8
  }"
```

如果只有本地 vLLM：

```bash
python -m entrypoints.launch_vllm \
  --model Qwen/Qwen3-VL-235B-A22B-Thinking \
  --tp 8
```

并设置：

```bash
export AGENT_COT_REASONER_BASE_URL=vllm
export AGENT_CODE_GENERATOR_BASE_URL=vllm
```

---

## 6. 小样本运行流程

### 第 1 次：运行 1 个 rotation 样本

目的：

- 检查模型 API 是否连通；
- 检查 Ray Serve 是否能启动；
- 检查 VGGT、SAM2、Orient-Anything 是否能加载；
- 检查 PythonTool 是否能生成并执行代码；
- 检查 session 日志和可视化是否生成。

命令：

```bash
python -m entrypoints.agent \
  --benchmark mindcube \
  --question_type rotation \
  --limit 1 \
  --concurrency 1 \
  --work_dir work_dir/mindcube_smoke_1
```

预期输出目录：

```text
work_dir/mindcube_smoke_1/
├── config.json
├── predictions.jsonl
├── results.csv
├── results_summary.json
└── session-*/
    ├── trace.jsonl
    ├── msg.jsonl
    ├── session_report.html
    └── visualizations/
```

### 第 2 次：运行 5 个样本

```bash
python -m entrypoints.agent \
  --benchmark mindcube \
  --question_type rotation \
  --limit 5 \
  --concurrency 1 \
  --work_dir work_dir/mindcube_smoke_5
```

第一次运行会加载大模型，速度较慢。第二次开始通常会更快。

### 第 3 次：运行 10 个样本

```bash
python -m entrypoints.agent \
  --benchmark mindcube \
  --question_type rotation \
  --limit 10 \
  --concurrency 1 \
  --work_dir work_dir/mindcube_smoke_10
```

### 第 4 次：扩展到 around

```bash
python -m entrypoints.agent \
  --benchmark mindcube \
  --question_type around \
  --limit 10 \
  --concurrency 1 \
  --work_dir work_dir/mindcube_around_10
```

### 第 5 次：扩展到 among

```bash
python -m entrypoints.agent \
  --benchmark mindcube \
  --question_type among \
  --limit 10 \
  --concurrency 1 \
  --work_dir work_dir/mindcube_among_10
```

当单个样本稳定后，再将 `concurrency` 提高到 2～4。

---

## 7. 结果检查方法

查看预测：

```bash
cat work_dir/mindcube_smoke_1/predictions.jsonl
```

查看汇总：

```bash
cat work_dir/mindcube_smoke_1/results_summary.json
```

查看每个样本的推理轨迹：

```bash
find work_dir/mindcube_smoke_1/session-* -name trace.jsonl -o -name msg.jsonl
```

查看 HTML 报告：

```text
work_dir/mindcube_smoke_1/session-<sample_id>/session_report.html
```

重点检查：

1. Task Formalization 是否正确生成 `C_R` 和 `C_O`。
2. 是否选择了正确的参考系。
3. 检测框是否正确。
4. VGGT 重建是否正常。
5. Orient-Anything 是否给出合理的物体姿态。
6. PythonTool 是否真正执行了几何计算，而不是直接猜答案。
7. 最终答案是否正确。
8. 是否出现频繁的 API timeout、CUDA OOM 或工具调用循环。

---

## 8. 预期工具调用链

MindCube-tiny 的一个典型样本可能包含：

```text
Semantic Analyst
-> 生成 C_task = (C_R, C_O)

Tool Orchestrator
-> SemanticDetector
-> GeometricReconstructor
   -> VGGTModel
   -> SAM2Model
-> ObjPoseEstimator
   -> OrientationAnythingModel
   -> rembg
   -> u2net.onnx
-> PythonTool
-> FinalAnswerGenerator
```

如果使用 MindCube 的 `around` 或 `among`，`LanguageToCamera` 也可能参与视角关系计算。

这些调用链是判断“是否真正复现 GCA”的关键，不应只看最终答案。

---

## 9. 合理的第一步成功标准

小样本熟悉阶段不要求达到论文准确率。只要满足以下条件，就说明流程基本跑通：

- `predictions.jsonl` 中每个样本都有非空答案。
- `results_summary.json` 能正常生成。
- session 日志中存在完整的 graph 阶段。
- 至少有一个样本成功调用 VGGT 或 GeometricReconstructor。
- 至少有一个样本成功调用 Orient-Anything。
- 至少有一个样本成功执行 PythonTool。
- 最终答案格式能被 MindCube evaluator 解析。
- 没有持续的 CUDA OOM、权重缺失或 API 555/超时错误。

---

## 10. 当前仍需补齐的项目

开始运行前需要完成：

```text
[ ] 在普通服务器终端确认 nvidia-smi 正常
[ ] 在 gca 环境中确认 torch.cuda.is_available() 为 True
[ ] 设置 AGENT_COT_REASONER_* 环境变量
[ ] 设置 AGENT_CODE_GENERATOR_* 环境变量
[ ] 测试 VLM API 或启动本地 vLLM
[ ] 下载 MindCube-tiny 到 data/mindcube
[ ] 给 entrypoints/agent.py 增加 --limit
[ ] 运行 1 个 rotation 样本
[ ] 运行 5～10 个 rotation 样本
[ ] 根据日志排查缺失工具或路径问题
```

---

## 11. 推荐执行顺序

```text
1. 确认 GPU
2. source scripts/gca_env.sh
3. 配置并测试 VLM API
4. 下载 MindCube-tiny
5. 增加 --limit 参数
6. 跑 1 个 rotation 样本
7. 检查 session_report.html 和 trace.jsonl
8. 跑 5～10 个 rotation 样本
9. 再跑 around / among
10. 流程稳定后再考虑完整 MindCube-tiny 和 MMSI-Bench
```

最终目标不是一次性跑完所有 benchmark，而是先确认：

```text
数据 -> Task Formalization -> 工具 -> 几何计算 -> 答案 -> 评测
```

这条链路在本地环境中完整且可解释地工作。
