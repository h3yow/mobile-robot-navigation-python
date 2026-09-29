# 移动机器人导航仿真服务 — 设计与使用说明

本地运行、可通过现有 FastAPI HTTP 层驱动的占用栅格导航仿真：
**全局路径规划（A\*）+ 局部轨迹跟踪（纯追踪差速模型）+ 会话级状态管理与并发控制**。
所有规划与执行结果都带可解释的判定依据（错误码、`last_note`、状态快照）。

---

## 1. 总体架构

```
HTTP 层 (nav/api.py, FastAPI)
  └─ Service 层 (nav/service.py)   多会话管理 / TTL / 幂等 / 参数解析
       └─ Session (nav/session.py) 单会话状态机（自带锁）
             ├─ GridPlanner  (nav/planner.py)    A* 全局规划 / 路径通畅性
             ├─ TrajectoryTracker (nav/controller.py) 纯追踪局部跟踪
             └─ OccupancyGrid (nav/grid.py)     占用栅格 / 坐标 / 碰撞 / 膨胀
```

数据流：
`静态地图 + 动态障碍矩形 → 有效占用栅格 → A* 路径 → 逐步跟踪指令（受运动学约束）→ 位姿积分`。

---

## 2. 运动学模型与代价模型

### 2.1 机器人运动学（差速驱动）

状态为 SE(2) 位姿 `(x, y, θ)`，控制量为体坐标系速度 `(v, ω)`，
一步（时长 `dt`）的积分：

```
x'   = x + v·cos(θ)·dt
y'   = y + v·sin(θ)·dt
θ'   = wrap(θ + ω·dt)          # 角度归一化到 [-π, π]
```

### 2.2 速度包络（任何指令都不得违反）

| 约束 | 含义 |
|---|---|
| `min_v ≤ v ≤ max_v` | 线速度限幅（默认 `min_v=0`，即不倒车） |
| `|ω| ≤ max_omega` | 角速度限幅 |
| `|ω / v| ≤ max_curvature` | 曲率上限（最小转弯半径 = `1/max_curvature`） |
| `|v − v_prev| ≤ max_acc·dt` | 加速度限幅 |

大角度偏差（`|α| > 90°`，目标在身后）时先**原地转向**（`v=0, ω≠0`），不前进。
小中角度偏差时使用**纯追踪几何**：

```
曲率 κ = 2·sin(α) / Ld        # α 为前视夹角，Ld 为前视距离
ω    = v·κ (+ 小比例纠偏项)
```

前视距离 `Ld` 在会话创建时取
`max(controller.lookahead_cells, 1.5·resolution, 2/max_curvature)`，
从而在几何上保证 `|κ| ≤ max_curvature`，粗分辨率与大曲率约束下均稳定。

### 2.3 碰撞模型与安全余量

* 机器人近似为半径 `footprint_radius` 的圆；
* 障碍检测用**圆与栅格单元的精确最近点距离**判定，并检查圆是否越出地图世界边界
  （因此任何轨迹都不会穿出地图）；
* 规划前把障碍按 Chebyshev 距离做膨胀，膨胀格数
  `inflation = 用户配置 obstacle_inflation_cells + ceil(footprint_radius/resolution)`，
  使规划路径与障碍保持至少一个外接半径的间距；
* 每个步进在**积分前**（当前位姿）与**积分后**（预测位姿）各做一次足迹碰撞复核；
  预测位姿触障则拒绝该指令（输出零速并进入 `BLOCKED`），从机制上保证
  **任何时刻轨迹都不穿越障碍、不越界、不超速度/曲率限制**。

### 2.5 长时推进的安全保证（advance）

`advance(duration, dt?, max_v?)` 在一次调用内连续执行多个“微步（tick）”，
但安全模型与单步完全相同、且更强：

* **时间细分**：内部把 `duration` 按 `dt`（默认 `controller.dt=0.1s`）切成
  `floor(duration/dt)` 个微步逐帧积分，每个微步都独立施加
  v/ω/曲率/**加速度**限幅，逐帧上报位姿——不是“一帧跳到 duration 之后”；
* **位姿间扫掠复核**：相邻两个上报位姿之间，以“机器人外接圆沿线段扫掠形成的
  **胶囊体**”对障碍栅格做**精确最近点**判定，并检查越界。因此即使单步位移
  大于隔墙厚度或通道宽度（例如 `dt=0.5s, max_v=5` 时一步 2.5m），
  也不可能“跨过”1 格薄墙或从窄口外穿入；
* **制动视距 + 局部通畅性**：每帧只检查当前位姿前方
  `前视距离 + v²/(2·max_acc) + max_v·dt` 弧长内的剩余路径，
  远处被挡时先继续前进（**能走多少走多少**），逼近到制动距离内才触发自动重规划；
  重规划失败则该帧**零速原地停住**（`BLOCKED`），下一帧用最新栅格重试，
  撤障/换环境后自动恢复，会话永不卡死；
* **速度上限临时调高**：`max_v` 只对本次调用生效（不改动会话配置），
  其余约束（ω/曲率/加速度/足迹）不变。

### 2.4 A\* 代价模型

* 邻域：4/8 邻域（`allow_diagonal`，默认 8）；
* 步代价：正交 `1.0`，对角 `√2`（`cost_diagonal`，要求 `≥1` 保证启发式可采纳）；
* 启发式：对角距离（8 邻域）/ 曼哈顿距离（4 邻域），A\* 最优且可采纳；
* **禁止穿墙角**：对角移动时要求两个正交邻格同时自由，避免对角切角；
* 路径后处理：删除共线中间点（直线压缩），首点为真实起点、末点为真实终点，
  并对压缩后的长直段再做连续足迹复核。

---

## 3. 输入形式与参数

### 3.1 地图

`map.cells` 支持两种形式：

* **二维行序列**：`[[0,1,0],[0,0,0]]`（行优先，行 y、列 x）；
* **一维扁平序列**：`[0,1,0,0,0,0]` + 显式 `width` / `height`。

取值仅允许 `0`（自由）/ `1`（障碍）。配套字段：

* `resolution`（米/格）：必须为有限正数；
* `origin`：`[x, y]`，单元 `(0,0)` 左下角的世界坐标，必须为有限数；
* `width` / `height`：仅一维输入需要，与数据不符会被拒绝。

### 3.2 动态障碍

`step` 请求体可选 `dynamic_obstacles`，元素为世界坐标矩形
`[xmin, ymin, xmax, ymax]`。语义为**全量覆盖**（每次步进用最新列表重建动态层），
矩形与地图无交叠时被裁剪为空（不报错）。动态层经栅格化后与静态地图做逐格 OR，
形成“当前有效占用栅格”。

### 3.3 参数（均带默认值，可在创建会话时覆盖）

| 组 | 字段 | 默认 | 说明 |
|---|---|---|---|
| limits | `max_v` / `min_v` | 1.0 / 0.0 | 线速度上下限 (m/s) |
| | `max_omega` | 1.5 | 角速度上限 (rad/s) |
| | `max_curvature` | 2.0 | 曲率上限 (1/m) |
| | `max_acc` | 2.0 | 加速度上限 (m/s²) |
| | `footprint_radius` | 0.3 | 外接圆半径 (m) |
| planner | `obstacle_inflation_cells` | 0 | 额外用户膨胀格数（足迹膨胀自动叠加） |
| | `allow_diagonal` | true | 是否允许对角移动 |
| | `max_expanded_nodes` | 200000 | 节点扩展预算，超限返回 `planning_timeout` |
| controller | `k_linear` / `k_angular` | 1.0 / 2.0 | 比例增益 |
| | `goal_tolerance` | 0.05 | 到点容差 (m) |
| | `lookahead_cells` | 0.6 | 纯追踪前视距离 (m)，会话层有下限保护 |
| | `dt` | 0.1 | 仿真步长 (s) |

---

## 4. 会话状态机

```
                 plan(start,goal)         step(到达)
 CREATED ───────────────────────► READY ──────────────► ARRIVED
                                    │ ▲                  ▲
                          step(执行)│ │replan/自动重规划  │step
                                    ▼ │                  │
                                  RUNNING ───────────────┘
                                    │
                  路径被挡且重规划失败 / 当前位姿或预测位姿触障
                                    ▼
                                  BLOCKED
                                    │ 环境变化（障碍消失/新路径可达）后继续 step
                                    ▼
                          自动重规划成功 → RUNNING → ARRIVED
```

* `CREATED`：已建图，未规划；此时 `step` 返回 `session_state_error (409)`。
* `READY`：规划/重规划成功，等待步进。
* `RUNNING`：执行中。
* `ARRIVED`：到达终点容差内；之后重复 `step` **幂等**（零速、停在原地）。
* `BLOCKED`：动态障碍封死可行运动；状态被保留，**不丢路径与目标**。
  后续 `step` 会在最新有效栅格上自动重试重规划，环境恢复后自动继续。
* `reset`：清空目标/轨迹/动态层回到 `CREATED`（保留地图与参数）。

单步 `step` 内部顺序（整体在会话锁内原子完成）：
重建有效栅格 → 当前位姿安全检查 → 剩余路径通畅性检查 →（被挡则）自动重规划
→ 到点判定 → 选前视点/算受约束指令 → 积分 → 预测位姿碰撞复核 → 更新状态。

---

## 5. 并发与错误语义

### 5.1 并发

* 服务级 `RLock` 保护会话表与幂等表；
* **每个会话一把 `RLock`**：`plan / step / replan / reset / 查询` 都在会话锁内完成，
  同一会话的并发请求按获锁顺序**串行化**，因此不会错序、不会读到撕裂状态
  （例如并发步进的 `step_count` 必为无重复的连续序列）；
* 不同会话用不同锁，**互不阻塞、状态完全隔离**；
* 失败（`BLOCKED`、规划错误）不破坏会话状态，可继续 `step` / `replan` 恢复。

### 5.2 幂等

* `POST /sessions` 与 `POST /sessions/{id}/step` 支持请求头 `Idempotency-Key`；
* 同键 + 同请求体：返回首次结果（重复创建返回同一 `session_id`；
  重复步进返回同一步结果且 `step_count` 不增加）；
* 同键 + 不同请求体：返回 `idempotency_conflict (409)`。

### 5.3 错误码（稳定、可区分）

| code | HTTP | 触发条件 |
|---|---|---|
| `invalid_input` | 422 | JSON 非法、参数结构/取值非法、resolution/origin 非法、栅格值非 0/1 |
| `map_shape_error` | 422 | 二维行长度不一致、一维长度与 width·height 不符、显式尺寸冲突 |
| `coordinate_out_of_bounds` | 422 | 起终点/坐标不在地图世界覆盖范围内（含非有限数值） |
| `start_in_obstacle` | 422 | 起点在障碍上，或与障碍距离小于足迹+膨胀余量 |
| `goal_in_obstacle` | 422 | 终点同上 |
| `goal_unreachable` | 422 | 起终点合法但无无碰撞路径（全封死、窄通道被膨胀封死等） |
| `planning_timeout` | 504 | A\* 扩展节点超过 `max_expanded_nodes`（大图保护） |
| `session_state_error` | 409 | 未规划就步进/推进等非法状态迁移 |
| `invalid_input`（advance/preview） | 422 | `duration` 非正/超过 60s、`dt`/`max_v`/`max_ticks`/`trajectory_capacity` 越界、预演自定义路线缺少 `start` 等 |
| `idempotency_conflict` | 409 | 幂等键复用于不同请求体 |
| `session_not_found` | 404 | 会话不存在或已删除 |
| `session_expired` | 404 | 会话超过空闲 TTL 被回收 |

错误响应统一形如：
`{"error": {"code": "...", "message": "...", "details": {...}}}`。

### 5.4 退化/极端输入的稳定判定

* **空旷地图**：A\* 直接给出对角/直线压缩路径；
* **全封死**：`goal_unreachable`（与“起点/终点在障碍上”区分）；
* **极窄通道**：1 格缺口在足迹膨胀后不可行时明确拒绝（`goal_unreachable`），
  调小膨胀/足迹才可通过，结论可复现；
* **坐标越界 / 非有限数值**：`coordinate_out_of_bounds`（地图边界为半开区间）；
* **分辨率/原点/尺寸不一致**：`map_shape_error` / `invalid_input`；
* **大地图**：300×300 空图规划为亚秒~毫秒级；200×200 单步为毫秒级；
  节点预算防止复杂大图下无限搜索，且全过程无数值异常退出。

---

## 6. 长时间仿真：按时长推进 / 轨迹查询 / 只读预演

### 6.1 按时长推进 `POST /sessions/{id}/advance`

请求体：

| 字段 | 必填 | 默认 | 语义 |
|---|---|---|---|
| `duration` | 是 | — | 本次要模拟的秒数，`(0, 60]`（硬上限，防止单次无限计算） |
| `dt` | 否 | `controller.dt`（0.1s） | 内部微步积分间隔，越小轨迹越密，必须 >0 |
| `max_v` | 否 | 会话 `limits.max_v` | **本次调用临时**线速度上限（可调高），不修改会话配置 |
| `dynamic_obstacles` | 否 | 不变 | 同 `step`，全量覆盖动态障碍 |

返回 `advance`：

```
steps_requested     计划微步数（floor(duration/dt)，封顶 20000）
steps_executed      实际执行微步数
requested_duration / dt / elapsed   请求秒数 / 微步间隔 / 实际模拟秒数
completed           是否把请求时长走完（或提前到达）
arrived / blocked   是否到达 / 是否被挡停
replanned           本次推进中是否发生过自动重规划
poses[]             每个微步对外上报的位姿序列（含中间扫过位姿、含被挡零速步）
samples[]           与 poses 一一对应的轨迹样本（index/t/cmd/标志位）
note                提前停止/完成原因（只走了一部分时说明为什么）
```

语义：

* **部分推进**：`completed=false` 时 `steps_executed < steps_requested`，
  已完成的运动全部保留；`blocked=true` 表示在安全前提下走到不能再走，
  `note` 给出原因（视距内封死且重规划失败 / 预测扫掠触障等）。
  之后可继续 `advance`、更新障碍、`replan` 或 `reset`，会话不卡死。
* **兼容性**：`step` 接口保持不变；`advance(duration=dt)` 默认参数下与
  连续 `duration/dt` 次 `step` 的位姿**逐点一致**（同一套积分与限幅）。
* 支持 `Idempotency-Key`（同键同体重放，步数/轨迹不重复增加）。

### 6.2 执行轨迹查询 `GET /sessions/{id}/trajectory`

每个实际发生的微步（含被挡时的原地零速步、到达那一步）落盘一条样本：

```
index               会话内全局步号（等于记录后的 step_count）
t                   会话内累计仿真时间（秒）
x,y,theta           该步上报位姿
v,omega             该步实际速度指令
replanned/blocked/arrived   该步状态标志
```

**保留上限与超限处理（明确约定）：**

| 项 | 默认 / 范围 |
|---|---|
| 容量 `trajectory_capacity` | 默认 **2000** 条，创建会话时可设 `[1, 100000]` |
| 超限策略 | **FIFO 丢弃最旧样本**（环形缓冲），不报错、不阻塞推进 |
| 查询返回 | `length`（当前条数）、`total`（累计产生）、`dropped=total-length`、`truncated`、`capacity`、`sim_time`、`samples[]` |
| 何时清零 | `plan` / `replan` / `reset` 清空轨迹与 `sim_time`（容量配置保留） |
| 不产生样本 | `ARRIVED` 之后重复的幂等空步（零速、位姿冻结）不占内存 |

* 轨迹样本与 `advance`/`step` 对外上报的**位姿序列、步数严格一一对应**，
  可用于复现与逐段核对；快照 `session.trajectory` 也带相同的 length/total/
  dropped/capacity/truncated 统计。
* 缓冲是**每会话独立**的，不同会话互不影响。

### 6.3 只读预演 `POST /sessions/{id}/preview`

在**不改动会话任何状态**的前提下，按给定路线与运行参数完整模拟一遍纯跟踪：

请求体：`path`（`[[x,y],...]`，可省略=预演当前路径）、
`start`（自定义 path 时必填）、`dt`（默认会话步长）、
`max_v`（默认会话上限）、`max_ticks`（默认 20000，上限 200000）。

返回 `preview`：

```
feasible            能否在预算内安全走完全程
ticks_simulated     模拟的微步数
fail_reason         start_blocked / path_invalid / collision / blocked / step_budget
fail_index          大概在路线第几个顶点出问题（沿途最近顶点下标）
fail_pose           出问题时（被拒绝）的预测位姿
min_clearance       全程机器人外缘到障碍/边界的最近距离（米）
min_clearance_point 最近点位置；clearance_capped=true 表示附近无障，按搜索半径封顶
note                人读判定依据
```

语义保证：

* **确定性 / 可复现**：纯函数（不读时钟、无随机数），相同
  地图+约束+路线+`dt/max_v` 必得相同结果，适合执行前筛查与批量调参；
* **不影响在线服务**：仿真本身在**会话锁外**执行，不阻塞同会话或其它会话的
  推进与查询，也不改变位姿、步数、轨迹、状态（仅查询快照会 touch `updated_at`）；
* 与执行共用同一个跟踪器/约束/碰撞模型：`feasible=true` 时预演微步数与
  真正 `advance` 到达所需微步数一致；
* 动态障碍计入预演（基于当前有效栅格）；大地图只扫描线段邻域栅格，
  1000×1000 地图的典型预演为毫秒级。

---

## 7. HTTP 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 健康检查 |
| POST | `/sessions` | 创建会话（可选 `Idempotency-Key`，可选 `trajectory_capacity`） |
| GET | `/sessions/{id}` | 查询状态快照 |
| POST | `/sessions/{id}/plan` | 设置起终点并全局规划 |
| POST | `/sessions/{id}/step` | 单步执行（可带 `dynamic_obstacles`，可选幂等键） |
| POST | `/sessions/{id}/advance` | 按时长安全推进（`duration`/`dt`/`max_v`，可选幂等键） |
| GET | `/sessions/{id}/trajectory` | 查询保留的执行轨迹（含中间位姿与容量统计） |
| POST | `/sessions/{id}/preview` | 只读预演（不改会话状态） |
| POST | `/sessions/{id}/replan` | 手动重规划 |
| POST | `/sessions/{id}/reset` | 重置 |
| DELETE | `/sessions/{id}` | 删除 |

状态快照包含：`status / pose / goal / path / path_index / step_count /
last_note（关键判定与迁移依据）/ created_at / updated_at / sim_time /
trajectory{length,total,dropped,capacity,truncated}`。

---

## 8. 本地验证方法

```bash
# 1) 安装依赖（需要 Python 3.14，使用 uv）
uv sync

# 2) 运行全部单测（日志会打印输入、关键判定与状态迁移依据）
uv run pytest -s

# 3) 启动服务
uv run uvicorn main:app --port 8000
#    交互文档： http://127.0.0.1:8000/docs
```

典型调用序列：

```bash
# 创建会话（8x10 空图）
curl -s -X POST localhost:8000/sessions -H 'content-type: application/json' \
  -d '{"map":{"cells":[[0,0,0,0,0,0,0,0,0,0],
                       [0,0,0,0,0,0,0,0,0,0],
                       [0,0,0,0,0,0,0,0,0,0],
                       [0,0,0,0,0,0,0,0,0,0],
                       [0,0,0,0,0,0,0,0,0,0],
                       [0,0,0,0,0,0,0,0,0,0]],"resolution":1}}'

# 规划
curl -s -X POST localhost:8000/sessions/<id>/plan -H 'content-type: application/json' \
  -d '{"start":{"x":0.5,"y":0.5,"theta":0},"goal":{"x":9.5,"y":5.5}}'

# 单步（可随时全量更新动态障碍）
curl -s -X POST localhost:8000/sessions/<id>/step -H 'content-type: application/json' \
  -d '{"dynamic_obstacles":[[6.0,0.0,7.0,4.0]]}'

# 一次推进 10 秒、本次速度上限 3 m/s
curl -s -X POST localhost:8000/sessions/<id>/advance -H 'content-type: application/json' \
  -d '{"duration":10.0,"max_v":3.0}'

# 查询这一路走过的轨迹（含中间位姿）
curl -s localhost:8000/sessions/<id>/trajectory

# 真正执行前先预演（不改状态）：自定义路线 + 运行参数
curl -s -X POST localhost:8000/sessions/<id>/preview -H 'content-type: application/json' \
  -d '{"path":[{"x":0.5,"y":0.5},{"x":7.5,"y":5.5}],
       "start":{"x":0.5,"y":0.5,"theta":0},"dt":0.1,"max_v":2.0}'

# 查询
curl -s localhost:8000/sessions/<id>
```

测试覆盖（`tests/`，共 85 个用例）：

* `test_grid.py`：二维/一维建图、形状/取值/分辨率校验、坐标变换、足迹碰撞、
  膨胀封死窄口、越界裁剪、**线段扫掠胶囊碰撞与 clearance 间隙查询**；
* `test_planner.py`：空图/带墙缺口/全封死/起终点障碍/越界/窄通道膨胀/
  节点预算超时/大图耗时/禁止穿墙角；
* `test_controller.py`：直行与反向朝向到达、原地转向、加速斜坡、角度归一化、
  全程 v/ω/曲率约束；
* `test_session.py`：**动态障碍触发自动重规划并绕行**、全程运动学约束与无碰撞、
  **封死→BLOCKED（零速冻结）→障碍消失自动恢复到达**、未规划拒绝、
  到达后幂等步进、手动重规划、大图单步性能；
* `test_service_api.py`：**6 会话并发隔离**、同会话 4 线程并发步进无错序/撕裂、
  步进与重规划并发、创建/步进幂等与冲突、TTL 过期可区分、删除后 404、
  各类非法输入 HTTP 错误码、非零原点/细分辨率建图规划；
* `test_advance_preview.py`：**S 形 1 格薄壁窄通道大跨度（dt=0.1/max_v=5 与
  dt=0.5）推进零穿障且全程满足 v/ω/曲率/加速度**、**薄隔墙封死后能走多少走多少、
  原地停住、撤障后继续到达**、轨迹含密集中间扫过位姿并与上报序列/步号/仿真时间
  一一对应、**超容量 FIFO 丢弃统计**、reset/replan 清空、**预演可复现、不改会话、
  报告失败顶点/原因/最近间隙、与执行逐 tick 一致、预演线程不阻塞其它会话推进**、
  duration/dt/max_v/max_ticks/capacity 非法输入拒绝、advance 幂等、
  老 `step` 调用与响应字段不变。
