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
* `advance` 长时推进且**显式传入 `dt`/`max_v`** 时进一步对整条微步圆弧做**扫掠复核**：沿弧以
  `≤ min(resolution/4, 0.2m)` 弦长采样，并以 `footprint_radius + 间距/2`
  的膨胀半径逐点检测，因此**两次上报位姿之间**（哪怕位移远大于墙厚/通道宽）
  也不穿障、不越界；走不下时二分取最大可行片段，详见 4A.1。

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
| 会话 | `trajectory_limit` | 2000 | 每个会话保留的执行轨迹记录条数上限 |

### 3.4 长时推进参数（`advance`，可选，不传时与逐步 `step` 行为一致）

| 字段 | 默认 | 含义/范围 |
|---|---|---|
| `duration` | （必填） | 本次一次推进要模拟的时长（秒）；`(0, 60]` |
| `dt` | 会话 `controller.dt`（0.1s） | 本次推进的微步间隔（秒），可取秒级 |
| `max_v` | 会话 `limits.max_v`（1.0 m/s） | 本次推进的临时速度上限（m/s），可调高 |

`dt`/`max_v` **只对本次 `advance` 调用生效**，不修改会话的默认配置；
不传它们时，`advance` 的每一步与连续调用 `step` 使用完全相同的参数与运动学
（legacy 语义，见 §8 向后兼容）；显式传入任一参数即启用增强微步
（圆弧积分/整段扫掠复核/部分推进/封死时尽量推进，见 §4A.1）。

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

## 4A. 长时运行的三种能力

### 4A.1 按指定时长推进 `advance`

`POST /sessions/{id}/advance` 在一次调用内模拟最多 `duration` 秒的连续运动，
内部按 `dt` 拆成若干微步顺序执行，避免“每挪一点点就调一次接口”。

请求体：

```json
{ "duration": 10.0, "dt": 0.5, "max_v": 3.0,
  "dynamic_obstacles": [[6.0, 0.0, 7.0, 4.0]] }
```

返回（节选）：

```json
{ "advance": {
    "status": "running|arrived|blocked",
    "pose": {"x": ..., "y": ..., "theta": ...},
    "requested_duration": 10.0,
    "simulated_duration": 8.4,
    "steps_executed": 17,
    "completed": false, "arrived": false, "blocked": true,
    "reason": "推进途中剩余路线被障碍/边界阻挡，已在最后安全位置停住……",
    "results": [ /* 每个微步一个与 step 同构的 StepResult（含 elapsed）*/ ] } }
```

**整段运动的安全保证（与步长无关）。** 即使一次推进很长、`dt` 到秒级、
单步位移比通道宽度或隔墙厚度还大，也不会穿障：

* 每个微步都沿**恒定 v/ω 的精确圆弧**以不超过 `min(resolution/4, 0.2m)`
  的弦长采样“扫过的位姿”，并对每个采样点用
  `footprint_radius + 采样间距/2` 的**膨胀半径**做足迹复核；
  相邻采样之间的“隧穿”缝隙因此被覆盖——5cm 薄墙也无法用 5m 的单步位移跨过。
* 越出地图世界边界与撞障碍同等对待（边界即墙）。
* 每条指令都先经加速度、速度、角速度、曲率包络限幅，再复核执行，
  因此不超速度/曲率/加速度限制。

**能走多少走多少。** 当一整个微步走不下时，会在该微步内迭代收缩执行时长
（二分求最大可行比例），找到最大可行时长并停在安全位置
（对应 `StepResult.blocked=true` 且 `0 < elapsed < dt`）；
任何微小运动都触障则**零速原地停住**（`elapsed=0`，状态 `BLOCKED`）。
`simulated_duration` 给出实际成功模拟的秒数，`reason` 说明为什么没走满。

**部分推进的指令按实际执行时长限幅。** 部分微步上报的速度指令不是
“按整段 `dt` 算好再截断”，而是按该微步**真正执行的时长** `elapsed`
重新施加加速度限幅：`|v − v_prev| ≤ max_acc·elapsed`（速度、角速度、
曲率包络同样满足），因此无论一次推进走满还是只走了一小段，
每一步上报指令都满足全部运动学约束；轨迹记录中的 `cmd/dt` 与之一致。

**前路封死时尽量推进。** 若剩余路径被动态障碍封死且重规划失败
（无可绕行路线），增强微步不会原地停住，而是沿原（被挡）路径继续跟踪，
由整段扫掠复核保证不穿障/不越界，**推进到障碍前最近的安全位置**再停住；
`last_note`/`reason` 会说明“封死且无法绕行、已停在最近安全位置”。
停下后会话状态（目标/路径/位姿）完整保留：清掉动态障碍、更换环境
（`reset` 或更新地图）或重新规划后都能继续 `advance`/`step`，不会卡死。

**停住/被挡后会话继续可用。** 状态、目标与路径都保留：之后可以
再次 `advance` / `step`（环境恢复后自动重规划继续）、更换 `dynamic_obstacles`、
`replan` 或 `reset`，不会卡死。到达终点提前结束也视为 `completed=true`。

保护上限：`duration ≤ 60s`、单次最多 `20000` 个微步（超限返回已安全模拟的部分）。

### 4A.2 执行轨迹查询 `trajectory`

`GET /sessions/{id}/trajectory` 返回本会话实际走过的轨迹，**不只是起止点**：

```json
{ "records": [ {
      "step": 1,
      "pose": {"x": ..., "y": ..., "theta": ...},
      "cmd":  {"v": ..., "omega": ...},
      "dt": 0.1,
      "swept": [ {"x":..,"y":..,"theta":..}, ... ] } ],
  "limit": 2000, "retained": 2000, "dropped": 35, "total": 2035,
  "session": { /* 状态快照 */ } }
```

* 每个**真实发生运动**的微步产生一条 `record`，其 `step` 与历史
  `StepResult` 的 `step_count`、`pose`、`cmd`、`elapsed(=dt)` **一一对应**；
  被挡零速/已到点这类未运动的微步不产生记录。
* `swept` 是该微步扫过的**稠密位姿**（含起点与终点，`swept[-1] == pose`）。
  把各记录的 `swept` 首尾相接即可还原连续轨迹，用于复现/核对是否穿墙。
* 存储采样数每条记录封顶 **256** 点（碰撞复核用更细的采样；长直段记录点更粗，
  但始终含中间扫过位置），避免单条记录过大。

**保留上限与超限处理。** 每个会话持有一个 FIFO 环形缓冲，只保留最近
`trajectory_limit` 条记录，**默认 2000**（创建会话时可用
`trajectory_limit` 配置，整数且 ≥1）。超出后**丢弃最旧记录**，
并在 `dropped` 中累计被丢弃条数；`total = retained + dropped`。
重新 `plan` 与 `reset` 会清空轨迹与计数。**不同会话各存各的缓冲，互不影响。**
查询返回的是副本，外部修改不影响会话内部状态。

### 4A.3 只读预演 `rehearse`

`POST /sessions/{id}/rehearse` 在**不改动会话状态**的前提下，
给定一条路线和一组运行参数，先判断能否安全走完：

```json
{ "route": [{"x":0.5,"y":2.5}, {"x":39.5,"y":2.5}],   // 可省：用当前规划路线
  "start": {"x":0.5,"y":2.5,"theta":0},                // 可省：用当前位姿
  "duration": 60.0, "dt": 0.1, "max_v": 2.0,           // 均可省
  "max_microsteps": 20000 }
```

返回：

```json
{ "report": {
    "feasible": true, "completed": true,
    "duration": 41.1, "steps": 411,
    "min_clearance": 0.2,
    "failure_index": null, "failure_step": null,
    "reason": "可安全走完全程，预计 41.10 秒 / 411 个微步" } }
```

* `feasible`：在预算内能否全程**不触障、不越界、不超 v/ω/曲率/加速度约束**
  地走到路线末端。预演执行与真实 `advance` **同一套**跟踪与扫掠碰撞逻辑，
  但**不做自动重规划**（用于评估“按这条给定路线、这组参数”是否可行）。
* `failure_index`：首个问题大致落在第几个路线点/段；`failure_step`
  为第几个微步；`reason` 区分“路线被阻挡 / 时长预算用尽 / 超约束 /
  微步预算耗尽 / 起点在障碍内”等。
* `min_clearance`：全程**机器人净空**的最小值（到最近障碍表面或地图边界的
  距离减去足迹半径，米）；越接近 0 越贴边，为负表示发生碰撞。
  该值基于栅格离散，精度为栅格量级；碰撞判定本身仍用精确圆-矩形检测。

**可复现与不干扰：**

* 纯函数式计算：相同栅格/路线/起点/参数 → 相同结果，适合执行前检查与批量调参筛选。
* 只读：预演后会话的位姿、`step_count`、状态、轨迹都不变。
* 不阻塞：服务只在会话锁内取栅格/路线/位姿的副本，计算在锁外进行，
  因此预演一个大地图时，同一会话的推进查询与**其它会话**都可正常进行。
* 大图保护：距离查询用 O(W·H) 预建的精确欧氏距离变换索引（建一次、会话内缓存），
  单次净空查询为亚线性小常数；另有 `max_microsteps` 预算，保证大图也能及时返回。

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
| `session_state_error` | 409 | 未规划就步进等非法状态迁移 |
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

## 6. HTTP 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 健康检查 |
| POST | `/sessions` | 创建会话（可选 `Idempotency-Key`） |
| GET | `/sessions/{id}` | 查询状态快照 |
| POST | `/sessions/{id}/plan` | 设置起终点并全局规划 |
| POST | `/sessions/{id}/step` | 单步执行（可带 `dynamic_obstacles`，可选幂等键） |
| POST | `/sessions/{id}/advance` | 按指定时长连续推进（`duration/dt/max_v`，可选幂等键） |
| GET | `/sessions/{id}/trajectory` | 查询保留的执行轨迹（含稠密扫掠位姿） |
| POST | `/sessions/{id}/rehearse` | 只读预演（不改会话状态） |
| POST | `/sessions/{id}/replan` | 手动重规划 |
| POST | `/sessions/{id}/reset` | 重置 |
| DELETE | `/sessions/{id}` | 删除 |

状态快照包含：`status / pose / goal / path / path_index / step_count /
last_note（关键判定与迁移依据）/ created_at / updated_at`。

---

## 8. 向后兼容与默认值

* **不传新参数时行为不变**：`POST /sessions/{id}/step` 的请求/响应保持原样
  （`StepResult` 仅新增一个带默认值的 `elapsed` 字段，旧客户端可忽略）。
  `POST /sessions` 不传 `trajectory_limit` 时使用默认 2000。
* **逐步 `step` 与升级前逐位一致**。`step` 固定使用 legacy 单步语义：
  用户**原始** controller 参数（前视距离不做下限保护）、**欧拉积分**
  （`x += v·cosθ·dt`）、仅对积分后预测位姿做**端点足迹复核**、
  **整步或零速**（无部分推进）、前路封死且重规划失败即原地 BLOCKED。
  同样的起点/目标/参数下，逐步执行的位姿序列与升级前完全一致。
* `advance` **不传 `dt`/`max_v`** 时，等价于在会话锁内连续执行若干次默认参数的
  `step`（相同的 `controller.dt` 与 `limits.max_v`、相同的 legacy 语义），
  逐位姿结果一致。
* **新能力只在显式传入推进参数时启用**：只有 `advance` 显式携带 `dt` 和/或
  `max_v` 时，才使用增强微步（圆弧积分、整段扫掠复核、部分推进且指令按
  实际执行时长限幅、封死时尽量推进到最近安全位置）。这些新能力不改变
  老调用的任何行为。
* 新增端点（`advance` / `trajectory` / `rehearse`）与旧端点并存，不改变
  旧端点路径与字段；`rehearse` 只读，不会对既有工作流产生副作用。
* 轨迹缓冲仅记录“真实发生运动”的微步；旧的逐步 `step` 使用方式同样会留下轨迹
  （legacy 微步的 swept 按欧拉模型采样，保证与上报位姿逐位一致）。

---

## 9. 本地验证方法

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

# 一次推进 10 秒（微步 0.5s、速度上限 3m/s）
curl -s -X POST localhost:8000/sessions/<id>/advance -H 'content-type: application/json' \
  -d '{"duration":10.0,"dt":0.5,"max_v":3.0}'

# 查询执行轨迹（含每条微步的稠密扫掠位姿）
curl -s localhost:8000/sessions/<id>/trajectory

# 只读预演：按当前路线/位姿，用 dt=0.2、max_v=2 评估能否安全走完
curl -s -X POST localhost:8000/sessions/<id>/rehearse -H 'content-type: application/json' \
  -d '{"dt":0.2,"max_v":2.0}'

# 查询
curl -s localhost:8000/sessions/<id>
```

测试覆盖（`tests/`）：

* `test_grid.py`：二维/一维建图、形状/取值/分辨率校验、坐标变换、足迹碰撞、
  膨胀封死窄口、越界裁剪；
* `test_planner.py`：空图/带墙缺口/全封死/起终点障碍/越界/窄通道膨胀/
  节点预算超时/大图耗时/禁止穿墙角；
* `test_controller.py`：直行与反向朝向到达、原地转向、加速斜坡、角度归一化、
  全程 v/ω/曲率约束；
* `test_session.py`：**动态障碍触发自动重规划并绕行**、全程运动学约束与无碰撞、
  **封死→BLOCKED（零速冻结）→障碍消失自动恢复到达**、未规划拒绝、
  到达后幂等步进、手动重规划、大图单步性能；
* `test_service_api.py`：**6 会话并发隔离**、同会话 4 线程并发步进无错序/撕裂、
  步进与重规划并发、创建/步进幂等与冲突、TTL 过期可区分、删除后 404、
  各类非法输入 HTTP 错误码、非零原点/细分辨率建图规划。
* `test_advance.py`：**100m 长走廊一次大跨度（dt=1s、v=5m/s）安全走完**、
  **5cm 薄墙在 5m 单步位移下不被穿透**、窄口高速被安全拦截且会话可继续、
  部分微步二分停在墙前、**部分推进指令的加速度按实际执行时长核算**、
  **封死墙尽量推进到最近安全位置→零速停住→清障恢复**、
  非法 duration/dt/max_v 拒绝、默认参数与逐步 `step` 逐位姿一致。
* `test_step_compat.py`：**不传新参数时逐步 `step` 与升级前参考实现
  （原始前视/欧拉积分/端点复核）逐位一致**（含会发散的转向场景）、
  默认参数 `advance` 与连续 `step` 逐位一致、单步响应结构兼容。
* `test_trajectory.py`：**swept 稠密轨迹与上报位姿/指令/步数完全对齐、
  首尾相接可复现**、大跨度记录含中间扫过点、**保留上限 FIFO 丢弃最旧**
  （limit/dropped/total 语义）、plan/reset 清空、多会话缓冲隔离、上限校验。
* `test_rehearse.py`：**预演结论可复现**、**不改会话位姿/步数/状态/轨迹**、
  跨墙路线给出 failure_index/failure_step 与近零净空、预算不足报障、
  批量调参结论稳定、**预演大地图期间其它会话正常推进（不互斥）**、
  大图预演及时返回、各类非法输入拒绝。
* `test_advance_api.py`：三个新端点的 HTTP 接线、advance 幂等与 409 冲突、
  非法输入 422、未规划 409、不存在会话 404、`trajectory_limit` 创建选项。
