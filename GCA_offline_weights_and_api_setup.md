# GCA 权重下载、路径配置与 API 接入

## 1. 路径规划

```bash
export GCA_ROOT=/data3/Agentic-Spatial-Reasoning/gca-main
export GCA_HF_HOME=/data3/Agentic-Spatial-Reasoning/hf_cache
export GCA_HF_HUB=$GCA_HF_HOME/hub
export U2NET_HOME=/data3/Agentic-Spatial-Reasoning/u2net
export EASYOCR_MODULE_PATH=/data3/Agentic-Spatial-Reasoning/easyocr

mkdir -p "$GCA_HF_HUB" "$U2NET_HOME" "$EASYOCR_MODULE_PATH/model"
```

规则：

```bash
export AGENT_CACHE_DIR="$GCA_HF_HUB"     # GCA 配置里的 cache_dir
export HF_HUB_CACHE="$GCA_HF_HUB"        # MoGe / GroundingDINO processor 会走这里
export HF_HOME="$GCA_HF_HOME"
export TRANSFORMERS_CACHE="$GCA_HF_HUB"
```

`AGENT_CACHE_DIR` 必须指向 HF cache 根目录，不能指向单个模型目录。两个变量都指向同一个 `hub` 目录最稳。

---

## 2. 下载清单

| 组件 | 下载地址 | 需要的文件 | 服务器最终位置 |
|---|---|---|---|
| VGGT | `https://huggingface.co/facebook/VGGT-1B/tree/main` | 整个仓库 | `$GCA_HF_HUB/models--facebook--VGGT-1B` |
| MoGe | `https://huggingface.co/Ruicheng/moge-2-vitl-normal/resolve/main/model.pt` | `model.pt` | `$GCA_HF_HUB/models--Ruicheng--moge-2-vitl-normal` |
| Orient-Anything | `https://huggingface.co/Viglong/Orient-Anything/resolve/main/ronormsigma1/dino_weight.pt` | `ronormsigma1/dino_weight.pt` | `$GCA_HF_HUB/models--Viglong--Orient-Anything` |
| DINOv2 | `https://huggingface.co/facebook/dinov2-large/tree/main` | `config.json`、`preprocessor_config.json` | `$GCA_HF_HUB/models--facebook--dinov2-large` |
| GroundingDINO（可选） | `https://huggingface.co/IDEA-Research/grounding-dino-base/tree/main` | 整个仓库 | `$GCA_HF_HUB/models--IDEA-Research--grounding-dino-base` |
| SAM2 | `https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt` | `sam2.1_hiera_large.pt` | `$GCA_ROOT/tools/third_party/sam2/checkpoints/` |
| rembg | `https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx` | `u2net.onnx` | `$U2NET_HOME/u2net.onnx` |
| EasyOCR | `https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/craft_mlt_25k.zip` | `craft_mlt_25k.pth` | `$EASYOCR_MODULE_PATH/model/` |
| EasyOCR | `https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/zh_sim_g2.zip` | `zh_sim_g2.pth` | `$EASYOCR_MODULE_PATH/model/` |

单文件模型可以直接浏览器打开直链下载。VGGT / GroundingDINO 建议在联网机器上用 `hf download` 整仓下载，再传 cache 目录；不要手工拼 HF cache 的 `refs/`、`blobs/`、`snapshots/` 结构。

GroundingDINO 只有在这两种情况下才需要：

- 不使用 `Qwen/Qwen3-VL-235B-A22B-Thinking` 或 `zai-org/GLM-4.5V-FP8`；
- 要显式关闭 VLM-as-detector，改回 GroundingDINO。

---

## 3. 方案 A：服务器能联网，直接在服务器下载

```bash
cd /data3/Agentic-Spatial-Reasoning/gca-main
conda activate gca

export GCA_ROOT=/data3/Agentic-Spatial-Reasoning/gca-main
export GCA_HF_HOME=/data3/Agentic-Spatial-Reasoning/hf_cache
export GCA_HF_HUB=$GCA_HF_HOME/hub
export U2NET_HOME=/data3/Agentic-Spatial-Reasoning/u2net
export EASYOCR_MODULE_PATH=/data3/Agentic-Spatial-Reasoning/easyocr

export HF_HOME="$GCA_HF_HOME"
export HF_HUB_CACHE="$GCA_HF_HUB"
export TRANSFORMERS_CACHE="$GCA_HF_HUB"

unset HF_HUB_OFFLINE
unset TRANSFORMERS_OFFLINE
mkdir -p "$GCA_HF_HUB" "$U2NET_HOME" "$EASYOCR_MODULE_PATH/model"
```

### 3.0 先测网络：huggingface.co 不通时用镜像

```bash
timeout 10 curl -vI https://huggingface.co
timeout 10 curl -vI https://hf-mirror.com
```

如果 `huggingface.co` 一直卡在：

```text
* Trying 103.200.31.172:443...
```

说明 TCP 443 被墙。只要 `hf-mirror.com` 返回 `HTTP/2 200`，就使用镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
unset HF_HUB_OFFLINE
unset TRANSFORMERS_OFFLINE
```

注意：`HF_ENDPOINT` 必须在运行 `hf download` 之前设置。已经卡住的下载先 `Ctrl+C`，再重新执行。

如果 `hf-mirror.com` 也不通，但服务器有内网代理：

```bash
export HTTP_PROXY=http://proxy-host:port
export HTTPS_PROXY=http://proxy-host:port
```

如果两个地址都不通，直接使用第 4 节的离线搬运方案。

### 3.1 下载 HF 权重

网络不稳定或使用镜像时，建议加 `--max-workers 1`。

```bash
cd "$GCA_ROOT"

hf download facebook/VGGT-1B \
  --cache-dir "$GCA_HF_HUB" \
  --max-workers 1

hf download Ruicheng/moge-2-vitl-normal model.pt \
  --cache-dir "$GCA_HF_HUB" \
  --max-workers 1

hf download Viglong/Orient-Anything ronormsigma1/dino_weight.pt \
  --cache-dir "$GCA_HF_HUB" \
  --max-workers 1

hf download facebook/dinov2-large config.json preprocessor_config.json \
  --cache-dir "$GCA_HF_HUB" \
  --max-workers 1
```

可选：

```bash
hf download IDEA-Research/grounding-dino-base \
  --cache-dir "$GCA_HF_HUB" \
  --max-workers 1
```

另开一个终端监控是否真的在下载：

```bash
watch -n 5 '
  du -sh /data3/Agentic-Spatial-Reasoning/hf_cache
  find /data3/Agentic-Spatial-Reasoning/hf_cache \
    -name "*.incomplete" \
    -printf "%s %p\n"
'
```

正常表现：

- `hf_cache` 体积持续变大；
- `.incomplete` 文件持续变大。

如果 5 分钟还没有 `.incomplete` 文件，说明仍然卡在网络连接，需要继续排查镜像或代理。

### 3.2 下载 SAM2

```bash
mkdir -p "$GCA_ROOT/tools/third_party/sam2/checkpoints"
wget -c -O "$GCA_ROOT/tools/third_party/sam2/checkpoints/sam2.1_hiera_large.pt" \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

如果 `dl.fbaipublicfiles.com` 无法访问，使用代理，或按第 4 节在联网机器下载后传输。

### 3.3 下载 u2net

```bash
mkdir -p "$U2NET_HOME"
wget -c -O "$U2NET_HOME/u2net.onnx" \
  https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx
```

如果 `github.com` 无法访问，使用代理，或按第 4 节离线传输。

### 3.4 下载 EasyOCR 模型

```bash
mkdir -p "$EASYOCR_MODULE_PATH/model"

python - <<'PY'
import os
import easyocr

model_dir = os.path.join(os.environ['EASYOCR_MODULE_PATH'], 'model')
easyocr.Reader(['ch_sim', 'en'], gpu=False, model_storage_directory=model_dir)
print('EasyOCR models:', model_dir)
PY
```

---

## 4. 方案 B：服务器无网，先手动下载，再传到服务器

### 4.1 在联网机器上下载 HF 权重

联网机器可以是你的本机、另一台服务器或 WSL。先安装 HF CLI：

```bash
python -m pip install -U "huggingface_hub[cli]"
```

建立中转目录并下载：

```bash
export TRANSFER_ROOT=$HOME/gca_transfer
export TRANSFER_HF_HOME=$TRANSFER_ROOT/hf_cache
export TRANSFER_HF_HUB=$TRANSFER_HF_HOME/hub
mkdir -p "$TRANSFER_HF_HUB"

hf download facebook/VGGT-1B \
  --cache-dir "$TRANSFER_HF_HUB"

hf download Ruicheng/moge-2-vitl-normal model.pt \
  --cache-dir "$TRANSFER_HF_HUB"

hf download Viglong/Orient-Anything ronormsigma1/dino_weight.pt \
  --cache-dir "$TRANSFER_HF_HUB"

hf download facebook/dinov2-large config.json preprocessor_config.json \
  --cache-dir "$TRANSFER_HF_HUB"
```

可选：

```bash
hf download IDEA-Research/grounding-dino-base \
  --cache-dir "$TRANSFER_HF_HUB"
```

### 4.2 在联网机器上下载 SAM2 和 u2net

```bash
mkdir -p "$TRANSFER_ROOT/sam2"
mkdir -p "$TRANSFER_ROOT/u2net"

wget -c -O "$TRANSFER_ROOT/sam2/sam2.1_hiera_large.pt" \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

wget -c -O "$TRANSFER_ROOT/u2net/u2net.onnx" \
  https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx
```

### 4.3 在联网机器上下载 EasyOCR 模型

```bash
export EASYOCR_MODULE_PATH="$TRANSFER_ROOT/easyocr"
mkdir -p "$EASYOCR_MODULE_PATH/model"

EASYOCR_MODULE_PATH="$EASYOCR_MODULE_PATH" python - <<'PY'
import os
import easyocr

model_dir = os.path.join(os.environ['EASYOCR_MODULE_PATH'], 'model')
easyocr.Reader(['ch_sim', 'en'], gpu=False, model_storage_directory=model_dir)
PY
```

最终中转目录结构应为：

```text
$TRANSFER_ROOT/
├── hf_cache/
│   └── hub/
│       ├── models--facebook--VGGT-1B/
│       ├── models--Ruicheng--moge-2-vitl-normal/
│       ├── models--Viglong--Orient-Anything/
│       ├── models--facebook--dinov2-large/
│       └── models--IDEA-Research--grounding-dino-base/   # 可选
├── sam2/
│   └── sam2.1_hiera_large.pt
├── u2net/
│   └── u2net.onnx
└── easyocr/
    └── model/
        ├── craft_mlt_25k.pth
        └── zh_sim_g2.pth
```

### 4.4 打包

HF cache 里是 symlink，必须用 `tar` 或 `rsync -a`，不要用普通网盘同步。

```bash
cd "$TRANSFER_ROOT"

tar -czf gca_hf_cache.tar.gz hf_cache
tar -czf gca_extra_assets.tar.gz sam2 u2net easyocr
```

如果传输链路不支持 symlink，可以解引用后打包：

```bash
tar -czhf gca_hf_cache_deref.tar.gz hf_cache
```

代价是 blobs 和 snapshots 会重复占空间。

### 4.5 传到服务器并解压

```bash
scp "$TRANSFER_ROOT/gca_hf_cache.tar.gz" \
    "$TRANSFER_ROOT/gca_extra_assets.tar.gz" \
    user@server:/data3/Agentic-Spatial-Reasoning/
```

服务器上：

```bash
cd /data3/Agentic-Spatial-Reasoning

tar -xzf gca_hf_cache.tar.gz
tar -xzf gca_extra_assets.tar.gz

mkdir -p gca-main/tools/third_party/sam2/checkpoints
mv sam2/sam2.1_hiera_large.pt \
   gca-main/tools/third_party/sam2/checkpoints/
```

解压后服务器路径应为：

```text
/data3/Agentic-Spatial-Reasoning/hf_cache/hub/models--...
/data3/Agentic-Spatial-Reasoning/u2net/u2net.onnx
/data3/Agentic-Spatial-Reasoning/easyocr/model/craft_mlt_25k.pth
/data3/Agentic-Spatial-Reasoning/easyocr/model/zh_sim_g2.pth
/data3/Agentic-Spatial-Reasoning/gca-main/tools/third_party/sam2/checkpoints/sam2.1_hiera_large.pt
```

---

## 5. 离线运行时环境变量

离线机器在启动 GCA 前执行：

```bash
export GCA_ROOT=/data3/Agentic-Spatial-Reasoning/gca-main
export GCA_HF_HOME=/data3/Agentic-Spatial-Reasoning/hf_cache
export GCA_HF_HUB=$GCA_HF_HOME/hub

export HF_HOME="$GCA_HF_HOME"
export HF_HUB_CACHE="$GCA_HF_HUB"
export TRANSFORMERS_CACHE="$GCA_HF_HUB"
export AGENT_CACHE_DIR="$GCA_HF_HUB"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export U2NET_HOME=/data3/Agentic-Spatial-Reasoning/u2net
export EASYOCR_MODULE_PATH=/data3/Agentic-Spatial-Reasoning/easyocr
export NUMBA_CACHE_DIR=/data3/Agentic-Spatial-Reasoning/numba_cache
mkdir -p "$NUMBA_CACHE_DIR"
```

注意：

- 下载阶段不要设 `HF_HUB_OFFLINE=1`；
- 只有所有文件都放到服务器后，才设置 `HF_HUB_OFFLINE=1`；
- `HF_HUB_OFFLINE=1` 时缺任何一个文件都会直接报错，不会再联网等待。

---

## 6. API 模式 VLM / LLM 配置

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

API 要求：

- 必须是 OpenAI-compatible 接口；
- 实际请求地址是 `{base_url}/chat/completions`；
- `base_url` 一般要带 `/v1`；
- `cot_reasoner` 必须是多模态模型，能接收 `image_url`；
- `code_generator` 可以用同一个模型，也可以换成纯文本 coding model；
- 不要用 `scripts/launch_agent.sh`，它写死了 `base_url='vllm'`。

模型名与 GroundingDINO：

```text
Qwen/Qwen3-VL-235B-A22B-Thinking   -> 不需要 GroundingDINO
zai-org/GLM-4.5V-FP8               -> 不需要 GroundingDINO
其他模型名 / 服务商别名              -> 需要 GroundingDINO
```

如果 API 只接受短名，例如 `qwen3-vl-235b-a22b-thinking`，就直接用短名，然后下载 GroundingDINO；或者把短名加入 `tools/utils/vlm_as_detector.py` 的 `VLM_AS_DETECTOR`。

API 连通性测试：

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

---

## 7. 启动前验证

```bash
cd /data3/Agentic-Spatial-Reasoning/gca-main
conda activate gca
```

检查配置：

```bash
python - <<'PY'
from workflow.config import AgentConfig

c = AgentConfig()
print('cache_dir =', c.cache_dir)
print('cot_model =', c.cot_reasoner_model)
print('cot_url   =', c.cot_reasoner_base_url)
print('code_model=', c.code_generator_model)
print('code_url  =', c.code_generator_base_url)
PY
```

检查 HF 文件：

```bash
python - <<'PY'
from huggingface_hub import hf_hub_download

for repo, filename in [
    ('Ruicheng/moge-2-vitl-normal', 'model.pt'),
    ('Viglong/Orient-Anything', 'ronormsigma1/dino_weight.pt'),
    ('facebook/dinov2-large', 'config.json'),
    ('facebook/dinov2-large', 'preprocessor_config.json'),
]:
    print('OK', repo, hf_hub_download(repo, filename))
PY
```

检查非 HF 文件：

```bash
test -f tools/third_party/sam2/checkpoints/sam2.1_hiera_large.pt && echo SAM2_OK
test -f "$U2NET_HOME/u2net.onnx" && echo U2NET_OK
ls "$EASYOCR_MODULE_PATH/model"
```

---

## 8. 启动

先跑单类型：

```bash
python -m entrypoints.agent \
  --benchmark mmsi \
  --question_type "MSR" \
  --concurrency 1
```

再跑全量：

```bash
python -m entrypoints.agent \
  --benchmark mmsi \
  --concurrency 8 \
  --resume
```

---

## 9. 常见问题

| 问题 | 直接处理 |
|---|---|
| MoGe 仍然联网下载 | 设置 `HF_HUB_CACHE=$GCA_HF_HUB`；MoGe 不读 `AGENT_CACHE_DIR` |
| GroundingDINO 仍然联网下载 | 设置 `HF_HUB_CACHE`；它只下载完整仓库，processor 不走 `AGENT_CACHE_DIR` |
| VGGT 报缺 `model.safetensors` | 重新完整下载 `facebook/VGGT-1B` |
| 离线报 `couldn't connect to huggingface.co` | cache 缺文件；在有网机器补齐后重新传 |
| `hf download` 卡住、只有 `Trying ...:443` | huggingface.co 的 TCP 443 被墙；执行 `export HF_ENDPOINT=https://hf-mirror.com` 后重试 |
| EasyOCR 运行时下载 | 启动 Python 前设置 `EASYOCR_MODULE_PATH` |
| rembg 运行时下载 | 确保 `$U2NET_HOME/u2net.onnx` 存在 |
| `serve.json not found` | API 模式不能把 `base_url` 设成 `vllm` |
| 搬运后 symlink 失效 | 用 `tar` / `rsync -a` 重新传，不要用普通网盘 |
| MMSI 报 `exists_ok` | 把 `evals/mmsi.py` 改成 `exist_ok=True`，或预先 `mkdir -p data/mmsi/images` |
