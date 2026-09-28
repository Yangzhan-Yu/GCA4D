# 从 GCA 到 4D-GCA：训练无关的动态空间推理方案

## 0. 参考工作

### GCA

核心思想：

```text
C_task = (C_R, C_O)
```

- `C_R`：参考系约束
- `C_O`：目标约束

GCA 先完成 Task Formalization，再在约束下调用几何工具和 PythonTool，解决静态 3D 场景中的 semantic-to-geometric gap。

局限：

- 主要面向静态图像；
- 缺少跨时间状态；
- 无法处理视频中的相机轨迹和物体状态；
- 没有持久 Scene Memory；
- 没有过程级 Agent Memory。

### S-Agent

S-Agent 将空间推理建模为连续的时空证据积累，而非孤立的单帧预测。

核心组成：

```text
VLM Planner
+ Hierarchical Spatial Tools
+ Scene Memory
+ Agent Memory
```

工具层级：

```text
Level 1: 2D 视觉证据
- keyframe selection
- open-vocabulary detection
- VLM grounding / verification
- image-level depth

Level 2: 2D -> 3D geometric lifting
- metric depth
- 3D coordinates
- camera pose
- repeated geometry evidence

Level 3: spatial knowledge experts
- metric measurement
- counting
- visual orientation
- relative position
- object-centric view
```

双记忆更新：

```text
S_{t+1} = Merge(S_t, e_t)
H_{t+1} = Append(H_t, c_t)
```

- `S_t`：Scene Memory，保存可复用的场景实体、空间事实和几何证据。
- `H_t`：Agent Memory，保存 planner thoughts、工具调用、观测、失败和中间结论。

当前 S-Agent 官方代码尚未开源，本文方案只参考论文公开的方法和接口设计。

---

# 1. 4D-GCA 的核心思想

在 GCA 的正式任务约束上增加时间和记忆：

```text
C_task(t) = (C_R(t), C_O(t), C_T, C_M)
```

- `C_R(t)`：随时间变化的参考系约束。
- `C_O(t)`：指定时刻或时间段内的目标约束。
- `C_T`：时间约束，包括事件顺序、区间和持续时间。
- `C_M`：记忆约束，指定需要保存或检索的实体、时间范围、分辨率和置信度。

推理过程变为：

```text
r_t = Planner(C_task(t), SceneMemory, AgentMemory)
o_t = Tool(r_t)
e_t, c_t = Split(o_t)
S_{t+1} = Merge(S_t, e_t)
H_{t+1} = Append(H_t, c_t)
```

最终回答前增加验证：

```text
a = Verify(C_task(t), S_t, H_t)
```

核心原则：

```text
冻结所有模型参数
+ 显式外部记忆
+ 时间约束
+ 几何工具
+ 可验证计算
```

---

# 2. 4D 状态定义

## 2.1 世界状态

```text
State(t) = {
    WorldMap,
    CameraState(t),
    ObjectStates(t),
    SceneGraph(t),
    EventTimeline(t)
}
```

## 2.2 相机状态

```text
CameraState(t) = {
    T_world_cam(t),
    intrinsic,
    timestamp,
    confidence
}
```

## 2.3 物体状态

```text
ObjectState = {
    object_id,
    category,
    aliases,
    T_world_obj(t),
    position(t),
    orientation(t),
    velocity(t),
    mask,
    point_cloud,
    first_seen,
    last_seen,
    confidence
}
```

## 2.4 时空事实

```text
SpatialFact = {
    subject,
    relation,
    object,
    time_interval,
    reference_frame,
    value,
    confidence,
    evidence
}
```

例如：

```text
(chair, left_of, sofa, [42s, 47s], front_view, confidence=0.91)
```

---

# 3. 双记忆设计

## 3.1 Scene Memory

Scene Memory 保存跨帧、跨视角可复用的场景证据。

建议存储：

```text
objects
observations
camera_poses
tracks
geometry
relations
events
evidence_frames
```

Scene Memory 不保存所有原始帧，只保存：

- 对象注册表；
- 几何属性；
- 轨迹；
- 场景关系；
- 支持该结论的关键帧；
- 置信度和来源。

## 3.2 Agent Memory

Agent Memory 保存推理过程：

```text
planner thoughts
tool requests
tool outputs
failed calls
uncertainties
intermediate conclusions
```

作用是避免：

- 重复调用工具；
- 忘记哪些证据已获取；
- 重复处理同一对象；
- 忽略失败和不确定性；
- 与之前的结论矛盾。

---

# 4. 工具层级

## Level 1：时空 2D 证据

- Keyframe Selection
- SAM2 segmentation
- GroundingDINO or VLM grounding
- CoTracker point tracking
- RAFT optical flow
- Object re-identification

## Level 2：2D -> 3D / 4D lifting

- VGGT / VGGT-Omega
- MonST3R
- CUT3R
- Metric depth：Depth Anything 3
- Camera pose tracking
- Object pose tracking
- Object trajectory fitting
- Chunk alignment and loop closure

## Level 3：Spatial Experts

- Metric Measurement Expert
- Counting Expert
- Visual Orientation Expert
- Relative Position Expert
- Object-Centric View Expert
- Temporal Relation Expert
- Motion/Trajectory Expert

## Level 4：Temporal Knowledge Experts

新增面向 4D 的专家：

- Object permanence expert
- Occlusion / reappearance expert
- Before/after/during expert
- Camera vs object motion expert
- Future state / counterfactual viewpoint expert

---

# 5. GCA 与 S-Agent 的对应关系

| GCA | S-Agent | 4D-GCA |
|---|---|---|
| Semantic Analyst | Planner | Temporal Task Formalizer |
| `C_task` | Tool request | `C_task(t)` |
| Solver Planner | VLM Planner | Memory-aware Temporal Planner |
| Toolbox | Hierarchical tools | 4D hierarchical tools |
| PythonTool | Experts + tools | Geometry + temporal computation |
| Stateless reasoning | Scene/Agent Memory | Persistent 4D dual memory |
| Final answer | Answer summary | Verified temporal answer |

4D-GCA 不是简单复现 S-Agent，而是：

```text
GCA 的正式任务约束
+ S-Agent 的时空记忆和证据积累
+ 4D 几何工具和一致性验证
```

---

# 6. 分阶段实现计划

## Phase 0：VSI-Bench 数据接入

状态：已完成。

已有：

```text
data/vsibench/test.jsonl
data/vsibench/videos/
evals/vsibench.py
config/agent_vsibench.json
```

当前实现：

```text
视频 -> 抽取 8 帧 -> 现有 GCA 多图推理
```

这是静态关键帧 baseline，不是最终 4D 方法。

## Phase 1：场景级 4D Memory

新增：

```text
tools/apis/four_d_memory.py
tools/apis/four_d_memory_types.py
```

实现：

```text
build_scene_memory
write_observation
merge_object
merge_relation
query_by_time
query_by_object
get_evidence_frames
get_camera_pose
get_scene_graph
verify_consistency
```

存储：

```text
SQLite: 实体、事件、关系、索引
Parquet: 观测、轨迹、相机位姿
文件系统: 关键帧和可视化
```

输出：

```text
data/vsibench/memory/<dataset>/<scene_name>/
├── memory.sqlite
├── objects.parquet
├── observations.parquet
├── camera_poses.parquet
├── tracks.parquet
├── events.parquet
├── relations.parquet
└── evidence/
```

验收标准：

```text
同一 scene 的多个问题只构建一次 Memory
不同问题可以复用对象、关系和时间证据
不存在逐问题重复抽帧和重复检测
```

## Phase 2：Video/Scene 输入改造

当前 AgentWorkflow 只接收 images。

需要增加：

```text
VideoInput
VideoPath
SceneId
TimeRange
```

改造：

```text
entrypoints/agent.py
workflow/workflow.py
workflow/state.py
```

目标：

```text
VSIBench sample
-> video_path + scene_id
-> load scene memory
-> question-specific retrieval
```

VSIBench 改为两种模式：

```text
baseline: 关键帧
memory: scene memory
```

## Phase 3：Temporal Task Formalization

新增：

```text
workflow/nodes/temporal_analyst.py
workflow/prompts/temporal_formalization.py
```

把自然语言时间关系解析为：

```text
C_T = {
    time_point,
    time_interval,
    event_order,
    duration,
    before_after_relation
}
```

需要处理：

```text
before
after
while
during
again
first
finally
turns left/right
moves forward/backward
```

输出：

```text
C_task(t) = (C_R(t), C_O(t), C_T, C_M)
```

## Phase 4：Memory-aware Planner

修改：

```text
workflow/nodes/solver/planner.py
```

Planner 的输入增加：

```text
Scene Memory Summary
Agent Memory Summary
Available Time Range
Known Object IDs
Uncertain Evidence
```

Planner 首先判断：

```text
当前还缺什么证据？
该证据在哪个时间段？
需要哪个对象或视角？
是否能直接从 Memory 中查询？
是否需要重新调用感知工具？
```

## Phase 5：4D 工具接入

按优先级加入：

### 第一组：轨迹和记忆

```text
CoTracker
TAPIR
SAM2
```

作用：

```text
物体轨迹
身份保持
遮挡后重现
```

### 第二组：相机和几何

```text
VGGT / VGGT-Omega
MonST3R
CUT3R
Depth Anything 3
```

作用：

```text
相机轨迹
深度
点云
全局地图
```

### 第三组：空间专家

```text
Counting Expert
Distance Expert
Size Expert
Direction Expert
Route Expert
Orientation Expert
```

## Phase 6：时序一致性验证

新增验证器：

```text
workflow/utils/temporal_consistency.py
```

检查：

- 时间戳是否有效；
- 参考系是否一致；
- 物体 ID 是否连续；
- 相机变换是否可组合；
- 静态物体是否稳定；
- 动态物体是否符合轨迹；
- 多帧结论是否矛盾。

失败时：

```text
重新检索 Memory
重新调用工具
标记 uncertainty
不能直接猜最终答案
```

---

# 7. VSI-Bench 上的实施路线

VSI-Bench 主要是静态室内场景和移动相机，因此第一阶段是：

```text
3D scene memory + camera-time memory
```

第二阶段再加入真正动态物体的 4D 推理。

## 第一步：单场景闭环

选择：

```text
arkitscenes/41069025
```

目标：

```text
1. 构建一次 scene memory
2. 复用于 5 个问题
3. 每个问题生成 C_task(t)
4. 从 memory 查询证据
5. 输出答案
6. 使用官方 Accuracy / MRA 评测
```

## 第二步：单题型

优先：

```text
object_counting
object_abs_distance
object_rel_direction
route_planning
```

这些题型能直接验证：

```text
对象去重
全局 3D 位置
视角切换
路径与方向
```

## 第三步：小样本评测

```text
每个题型 10 个问题
共享 scene memory
记录 memory hit rate
记录重复检测率
```

## 第四步：消融实验

```text
A. Video CoT
B. 当前 8 帧 GCA baseline
C. Scene Memory only
D. Scene Memory + C_task(t)
E. Scene + Agent Memory
F. 完整 4D-GCA
```

---

# 8. 推荐代码目录

```text
tools/apis/
├── four_d_memory.py
├── four_d_memory_types.py
├── video_loader.py
├── object_tracker.py
├── camera_tracker.py
└── temporal_experts.py

entrypoints/
├── build_vsibench_memory.py
└── agent.py

workflow/
├── nodes/
│   ├── temporal_analyst.py
│   └── solver/
├── prompts/
│   └── temporal_formalization.py
└── utils/
    └── temporal_consistency.py

evals/
└── vsibench.py
```

---

# 9. 主要风险

1. 4D 重建工具在动态场景中不稳定。
2. 相机运动和物体运动容易混淆。
3. 长视频 memory 会快速膨胀。
4. 物体身份切换会污染 Scene Memory。
5. 时间戳和事件边界难以自动定位。
6. 多轮工具调用可能重复或矛盾。
7. VSI-Bench 主要是静态房间移动相机，不等同于高动态 4D 场景。
8. 最终必须区分：方法问题、工具问题、benchmark 问题。

---

# 10. 最近三步

```text
Step 1:
实现 FourDMemory 数据结构和 SQLite/Parquet 存储

Step 2:
实现单场景 Scene Memory Builder，并复用于 5 个 VSI-Bench 问题

Step 3:
加入 C_task(t)，让 Planner 根据时间、参考系和对象从 Memory 检索证据
```

完成这三步后，再接入 CoTracker、相机位姿和动态 4D 工具。

---

# 11. 修正：问题驱动的按需证据收集

先前实现的 `build_vsibench_objects.py` 会预先检测整个场景的所有类别，这不适合作为主流程。

问题：

- 会产生大量与当前问题无关的对象；
- 低置信度检测会污染 Scene Memory；
- 不同类别可能在相同位置重复；
- 容易错误地把同一物体拆成多个对象；
- 无法根据问题动态决定需要什么证据。

正确方式应遵循 S-Agent 的证据积累思想：

```text
问题
-> Planner 分解需要什么证据
-> 针对目标实体调用 grounding/tracking/geometry 工具
-> 只把问题相关证据合并进 Scene Memory
-> 检查证据是否足够
-> 不足则继续请求
-> 足够后计算答案
```

## 11.1 问题驱动的 Evidence Request

Planner 首先从问题中提取：

```text
target_entities
reference_entities
relation_or_metric
time_constraint
reference_frame
required_evidence
```

例如：

```text
Question:
Measuring from the closest point of each object, what is the distance
between the sofa and the stove?

Evidence Request:
- ground sofa
- ground stove
- recover 3D points for sofa
- recover 3D points for stove
- estimate metric scale
- compute closest-point distance
```

只检测 `sofa` 和 `stove`，不检测 table、chair、tv。

## 11.2 Scene Memory 按需写入

Memory 中只创建当前证据链需要的实体：

```text
object_id
category
supporting_frames
bboxes
masks
3D points
relations
measurements
confidence
```

如果后续问题再次需要同一个对象，则通过 alias、跨帧外观和 3D 位置链接到已有实体。

## 11.3 更新后的推理循环

```text
C_task(t)
-> Planner generates EvidenceRequest r_t
-> Entity Grounding Tool
-> Optional Tracking Tool
-> Optional Geometry Tool
-> Spatial Expert
-> Merge e_t into Scene Memory
-> Append c_t to Agent Memory
-> Check evidence sufficiency
-> Final Answer
```

## 11.4 工具调用原则

```text
不要预先检测全场景类别
只检测问题中出现的目标实体
优先复用 Scene Memory 中已有证据
只有证据不足时才重新调用感知工具
计数问题例外：需要检测目标类别的全部实例
```

## 11.5 当前代码定位

保留：

```text
build_vsibench_memory.py
- 构建帧时间轴，问题无关，保留

build_vsibench_geometry.py
- 构建相机位姿和场景几何，问题无关，保留
```

不作主流程使用：

```text
build_vsibench_objects.py
- 仅作为全场景检测诊断/可视化工具
- 不应作为 VSI-Bench 推理前处理
```

## 11.6 下一步实现

```text
1. Question Evidence Planner
2. Target Entity Extraction
3. On-demand VLM/GroundingDINO grounding
4. SAM2 mask refinement
5. VGGT 3D lifting
6. Merge only relevant evidence into Scene Memory
7. Evidence sufficiency check
8. Final spatial computation
```
