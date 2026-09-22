# 会话状态机、并发与错误语义

## 1. 会话生命周期与状态机

```
                 POST /sessions (auto_plan=true，默认)
   ┌──────────┐         规划成功          ┌────────┐
   │ CREATED  │ ───────────────────────▶ │ ACTIVE │
   └──────────┘   (auto_plan=false 时     └───┬────┘
                  停留 CREATED，需先 /plan)    │ step
                                              ▼
                          ┌──────────────────────────────┐
                          │ ACTIVE：跟踪前进              │
                          │  - 前瞻/弧段发现障碍 → 重规划  │
                          │  - 重规划成功：本周期原地      │
                          │    记录 replanned，下周期走新路│
                          └───┬───────────────┬──────────┘
                  重规划失败   │               │ 到达目标
                              ▼               ▼
                         ┌─────────┐     ┌─────────┐
                         │ BLOCKED │     │ REACHED │（终态）
                         └───┬─────┘     └─────────┘
              更新障碍/重规划成功│ 或移除障碍
                              ▼
                          回到 ACTIVE
```

- **CREATED**：已建会话但无有效路径（`auto_plan=false`）；此时 `/step` 返回 409 `INVALID_REQUEST`，提示先 `/plan`。
- **ACTIVE**：有路径，可步进。
- **BLOCKED**：前方被挡且**重规划仍失败**（不可达或超时）。机器人保持原位置、速度归零，**状态不被破坏**：移除障碍后再次 `/step` 或 `/plan` 即可继续。
- **REACHED**：终态，位置进入 `goal_tolerance` 且（若指定）朝向对准；继续 `/step` 或 `/plan` 返回 409 `SESSION_ALREADY_FINISHED`。
- **FAILED**：机器人当前位置与（新加入的）动态障碍重合。移除该障碍后调用 `/plan` 可恢复。

### 重规划触发与冷却

- 跟踪器通过“路径前瞻 + 弧段采样”发现障碍，返回 `path_blocked_ahead`。
- 会话以**当前机器人位置**为起点、原目标为终点，在最新地图上重新 A*；成功后 `replan_count+1`，记录 `replan_reason`，当周期不移动（新路径下周期生效）。
- 重规划冷却 `replan_cooldown=2` 步：冷却期内即使仍提示阻挡，也只保持停止而不重复规划，杜绝“同一障碍每步重规划”的抖动。
- 每次成功重规划都是一条可解释事件，快照 `last_step.events` 中可见 `replanned:path_blocked_ahead`。

## 2. HTTP 接口

所有接口前缀 `/api/v1`：

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/sessions` | 建会话；默认 `auto_plan=true` 立即规划 |
| GET | `/sessions` | 列出会话 id |
| GET | `/sessions/{id}` | 完整状态快照（位置/朝向/速度、剩余路径、动态障碍、最近一步、事件） |
| POST | `/sessions/{id}/plan` | 显式（重新）规划 |
| POST | `/sessions/{id}/step` | 推进 `steps`（1..10000）个周期；遇到 REACHED/BLOCKED/FAILED 提前停止并置 `stopped_early` |
| POST | `/sessions/{id}/obstacles` | 增量更新动态障碍 `[[col,row,0/1],...]`，`reset:true` 清空 |
| DELETE | `/sessions/{id}` | 删除会话 |

另有 `GET /`、`GET /health`、`GET /docs`（Swagger UI）。

### 动态障碍更新

- 障碍以**栅格坐标**给出（0 表示清除、1 表示占用），叠加在静态地图上，不修改原始地图；`reset` 可一键回到静态地图。
- 越界格 → `COORDINATE_OUT_OF_BOUNDS`；非法值 → `MAP_INVALID_CELL_VALUE`。
- 更新后若机器人正好被障碍覆盖 → 会话进入 FAILED，响应快照可解释。

## 3. 并发语义

- 每个会话持有一把 `asyncio.Lock`：同一会话的 plan/step/obstacle/查询在异步层**严格串行**，并发下不会错序或状态撕裂；不同会话使用不同锁，可完全并行。
- 取锁带超时（`SessionStore.lock_timeout`，默认 5 秒）。超时返回 **409 `SESSION_LOCK_TIMEOUT`**，与业务冲突区分，不会无限排队。
- 状态查询（GET）也取同一把锁，保证读到的是某一步的完整一致快照。
- 会话仓库自身的创建/删除由一把全局短锁保护。

## 4. 幂等语义

所有变更类请求（建会话自动规划、plan、step、obstacles）都支持可选 `request_id`：

- **相同 `request_id` + 相同请求体**：不重复执行，返回首次结果，并额外带 `"idempotent_replay": true`。步进因此不会被重复提交推动两次。
- **相同 `request_id` + 不同请求体**：返回 **409 `IDEMPOTENCY_CONFLICT`**，且不改变任何状态。
- 不带 `request_id`：正常执行（非幂等）。
- 判定依据为请求体的稳定 JSON 指纹（SHA-256 截断）；步进请求的指纹含 `steps`，避免不同批量大小被误复用。

## 5. 错误响应格式

所有业务错误均为统一结构，`code` 可程序化区分：

```json
{ "error": { "code": "GOAL_UNREACHABLE",
             "message": "人类可读说明",
             "detail": { ... } } }
```

| code | HTTP | 触发条件 |
|---|---|---|
| `MAP_EMPTY` / `MAP_SHAPE_INVALID` | 422 | 缺地图、尺寸为 0 |
| `MAP_DIMENSION_MISMATCH` | 422 | 二维行宽不一致、一维长度与 width*height 不符、显式尺寸与数据不符 |
| `MAP_INVALID_CELL_VALUE` / `MAP_INVALID_RESOLUTION` / `MAP_INVALID_ORIGIN` | 422 | 非法格值/字符、非正分辨率、非法原点 |
| `MAP_NO_FREE_SPACE` | 422 | 地图全部占用 |
| `INVALID_KINEMATICS` | 422 | 运动学参数非正/非有限/不自洽 |
| `COORDINATE_OUT_OF_BOUNDS` | 422 | 起终点或动态障碍格越界 |
| `START_IN_OBSTACLE` / `GOAL_IN_OBSTACLE` | 422 | 起/终点与障碍相交 |
| `GOAL_UNREACHABLE` | 422 | 无连通路径（含太窄穿不过去） |
| `PLAN_TIME_BUDGET_EXCEEDED` | 408 | 超时/超过最大扩展数 |
| `SESSION_NOT_FOUND` | 404 | 会话不存在 |
| `SESSION_ALREADY_FINISHED` | 409 | 终态后继续步进/规划 |
| `SESSION_LOCK_TIMEOUT` | 409 | 同会话锁等待超时 |
| `IDEMPOTENCY_CONFLICT` | 409 | request_id 撞号但请求体不同 |
| `INTERNAL_ERROR` | 500 | 未预期错误（兜底，记录堆栈） |

“失败后状态保持可继续”的具体保证：BLOCKED 不回滚已走路径、不清空动态障碍、速度归零；移除障碍或场景变化后 `/step` 自动重新尝试规划，或 `/plan` 显式重规划。
