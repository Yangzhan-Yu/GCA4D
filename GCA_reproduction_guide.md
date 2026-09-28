我读了 `README.md`、`docs/install.md`、`docs/dataset.md` 和 `docs/usage.md`，并核对了 `entrypoints/agent.py`、`workflow/config.py` 和 `evals/factory.py`。

先说结论：这个仓库复现的是 **GCA 在若干空间推理 benchmark 上的推理和评测流程**，没有看到训练流程。建议先完整复现一个 benchmark，例如 MMSI-Bench，再扩展到 5 个数据集。整个流程是：

**环境 → 第三方视觉模型 → Python 依赖 → VLM/LLM 服务 → 数据集 → 配置 → 启动 agent → 查看结果**

## 1. 创建 Python 环境并安装 PyTorch

在仓库根目录执行：

```bash
cd /data3/Agentic-Spatial-Reasoning/gca-main

conda create -n gca python=3.11 -y
conda activate gca

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu124
```

要求是 Python >= 3.11、PyTorch >= 2.5.1；README 的安装文档使用 CUDA 12.4 版本。先用 `nvidia-smi` 确认驱动能支持 CUDA 12.4。

## 2. 准备第三方视觉模型

README 中的路径写的是 `visual-agent/`，但当前仓库根目录已经是 `gca-main`。实际代码会从 `tools/third_party/` 找这些仓库，所以应执行：

```bash
mkdir -p tools/third_party
cd tools/third_party

git clone --depth=1 https://github.com/facebookresearch/vggt.git
cd vggt
pip install .
cd ..

git clone --depth=1 https://github.com/facebookresearch/sam2.git
cd sam2
pip install .
cd checkpoints
sh download_ckpts.sh
cd ../..

git clone --depth=1 https://github.com/SpatialVision/Orient-Anything.git
git clone --depth=1 https://github.com/microsoft/MoGe.git

cd ../..
```

至少需要 SAM2 的：

```text
tools/third_party/sam2/checkpoints/sam2.1_hiera_large.pt
```

VGGT、MoGe、Orient-Anything 的部分权重会在第一次运行时从 Hugging Face 下载。如果机器不能联网，需要提前下载并通过配置中的 `cache_dir` 指向权重缓存目录。

## 3. 安装 GCA Python 依赖

README 文档里的命令漏了 `-r`，应改成：

```bash
cd /data3/Agentic-Spatial-Reasoning/gca-main
pip install -r requirements/gca.txt
```

## 4. 准备 VLM/LLM

GCA 有两个模型角色：

- `cot_reasoner`：规划、推理、工具调用，通常是多模态模型；
- `code_generator`：根据规划生成几何计算 Python 代码。

最简单可以直接使用 OpenAI 兼容 API，也可以用 vLLM 本地部署。README 和启动脚本默认使用：

```text
Qwen/Qwen3-VL-235B-A22B-Thinking
```

如果使用 vLLM，建议单独创建环境：

```bash
conda create -n gca-vllm python=3.11 -y
conda activate gca-vllm

pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128

cd /data3/Agentic-Spatial-Reasoning/gca-main
pip install -r requirements/vllm.txt
```

然后启动模型服务：

```bash
python -m entrypoints.launch_vllm \
  --model Qwen/Qwen3-VL-235B-A22B-Thinking \
  --tp 8
```

服务会把自己的 IP、端口等信息写入：

```text
logs/serve.json
```

注意：`scripts/serve_qwen3_vl_235b_thinking.sh` 里面引用了未定义的 `$SERVED_NAME`，可能会启动失败，所以建议先使用文档中的手动启动命令，而不是直接运行这个脚本。

之后配置两个模型角色，例如：

```bash
export AGENT_COT_REASONER_MODEL='Qwen/Qwen3-VL-235B-A22B-Thinking'
export AGENT_COT_REASONER_BASE_URL='vllm'
export AGENT_COT_REASONER_API_KEY='bearer'

export AGENT_CODE_GENERATOR_MODEL='Qwen/Qwen3-VL-235B-A22B-Thinking'
export AGENT_CODE_GENERATOR_BASE_URL='vllm'
export AGENT_CODE_GENERATOR_API_KEY='bearer'
```

README 中的 `scripts/launch_agent.sh` 已经帮你设置了这些环境变量。

## 5. 准备评测数据集

README 支持：

- MMSI-Bench
- MindCube
- OmniSpatial
- SPBench
- CVBench

数据集必须放在仓库根目录的 `data/<benchmark>` 下，因为 `evals/factory.py` 会自动寻找：

```text
data/mmsi
data/mindcube
data/omnispatial
data/spbench
data/cvbench
```

建议先只准备 MMSI-Bench：

```bash
mkdir -p data/mmsi

huggingface-cli download RunsenXu/MMSI-Bench \
  --repo-type dataset \
  --local-dir data/mmsi
```

最终至少应有：

```text
data/mmsi/MMSI_Bench.parquet
```

MMSI 第一次运行时会自动生成 `data/mmsi/images/`。不过当前代码 `evals/mmsi.py` 中有一个拼写错误：

```python
os.makedirs(image_dir, exists_ok=True)
```

正确参数应为 `exist_ok=True`。运行 MMSI 前需要修正，或者提前手工创建：

```bash
mkdir -p data/mmsi/images
```

其他数据集的下载和解压步骤见 `docs/dataset.md`，重点是保证目录结构与文档一致。

## 6. 检查配置

每个 benchmark 都有默认配置，例如：

```text
config/agent_mmsi.json
config/agent_mindcube.json
config/agent_omnispatial.json
config/agent_spbench.json
config/agent_cvbench.json
```

这些配置主要设置 benchmark、启用的工具、问题类型、并发和日志等，但通常不包含模型 API 地址和密钥，所以需要通过环境变量、JSON 或启动脚本补充。

配置优先级是：

```text
CLI 参数 > JSON 配置 > 环境变量
```

## 7. 启动 MMSI 评测

最直接的启动方式：

```bash
python -m entrypoints.agent \
  --benchmark mmsi \
  --concurrency 16
```

也可以直接用脚本：

```bash
bash scripts/launch_agent.sh mmsi
```

如果只想先验证流程，建议先用一个 question type：

```bash
python -m entrypoints.agent \
  --benchmark mmsi \
  --question_type "MSR" \
  --concurrency 1
```

确认单类型能正常完成后，再跑全部 MMSI，最后运行其他四个数据集。

中断后可以恢复：

```bash
python -m entrypoints.agent \
  --benchmark mmsi \
  --concurrency 16 \
  --resume
```

## 8. 查看结果

默认结果目录类似：

```text
work_dir/mmsi_<model-name>_v1/
```

其中通常包含：

```text
config.json
predictions.jsonl
results.csv
results_summary.json
session-<sample_id>/
```

每个 session 里还包括 `trace.jsonl`、`msg.jsonl`、可视化结果和 `session_report.html`。

如果推理已经完成，只想重新计算指标：

```bash
python -m entrypoints.summary_results \
  --benchmark mmsi \
  --work_dir work_dir/mmsi_<model-name>_v1
```

## 最终建议顺序

```text
1. Conda + PyTorch
2. tools/third_party 中的 VGGT/SAM2/OrientAnything/MoGe
3. pip install -r requirements/gca.txt
4. 启动 Qwen3-VL vLLM 或配置外部 API
5. 下载 MMSI-Bench 到 data/mmsi
6. 修正 MMSI 的 exists_ok 拼写并启动评测
7. 检查 predictions.jsonl 和 results_summary.json
8. 再依次复现 MindCube、OmniSpatial、SPBench、CVBench
9. 如需论文中的 CoT 对照，再运行 entrypoints.cot_baseline
```

另外，这个仓库不是“一键复现论文所有表格”的脚本。要得到论文级别的完整复现结果，还需要保证模型版本、vLLM 参数、prompt version、并发设置、数据集版本和评测配置都与论文一致。
