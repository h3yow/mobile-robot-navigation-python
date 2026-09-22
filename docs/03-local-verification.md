# 本地运行与验证指南

## 1. 环境准备

需要 Python 3.14（`pyproject.toml` 约束）。使用 uv：

```bash
uv sync                 # 安装运行与开发依赖（fastapi、pytest、pytest-asyncio）
```

## 2. 启动服务

```bash
uv run fastapi run main.py --host 127.0.0.1 --port 8000
# 或
uv run uvicorn main:app --reload
```

- 健康检查：`GET http://127.0.0.1:8000/health`
- 交互式文档：`http://127.0.0.1:8000/docs`（可直接发请求）

## 3. curl 端到端示例

### 3.1 建会话并自动规划

ASCII 地图中 `.`/`0`/空格为空闲，`#`/`1`/`x` 为占用：

```bash
SID=$(curl -s -X POST http://127.0.0.1:8000/api/v1/sessions \
  -H 'Content-Type: application/json' -d '{
    "grid": ["....................",
             "....................",
             "....................",
             "....................",
             "....................",
             "....................",
             "....................",
             "....................",
             "....................",
             "....................",
             "....................",
             "...................."],
    "resolution": 0.1,
    "start": {"x": 0.15, "y": 0.15, "theta": 0.0},
    "goal":  {"x": 1.85, "y": 1.05, "theta": 0.0},
    "kinematics": {"robot_radius": 0.0, "dt": 0.1},
    "lookahead": 0.25
  }' | python3 -c 'import sys,json;print(json.load(sys.stdin)["session_id"])')
echo $SID
```

也可用二维数组 `"occupancy": [[0,1],[..]]`，或一维数组配合 `"width"/"height"`。

### 3.2 步进、查询、加动态障碍

```bash
# 推进 10 个周期
curl -s -X POST http://127.0.0.1:8000/api/v1/sessions/$SID/step \
  -H 'Content-Type: application/json' -d '{"steps": 10}'

# 查询完整状态
curl -s http://127.0.0.1:8000/api/v1/sessions/$SID

# 在前方某格放一个动态障碍（col,row,value）
curl -s -X POST http://127.0.0.1:8000/api/v1/sessions/$SID/obstacles \
  -H 'Content-Type: application/json' -d '{"cells": [[10, 7, 1]]}'

# 继续步进；前瞻发现障碍会自动重规划（last_step.replanned=true）
curl -s -X POST http://127.0.0.1:8000/api/v1/sessions/$SID/step \
  -H 'Content-Type: application/json' -d '{"steps": 50}'

# 清空动态障碍
curl -s -X POST http://127.0.0.1:8000/api/v1/sessions/$SID/obstacles \
  -H 'Content-Type: application/json' -d '{"reset": true}'
```

### 3.3 幂等重试

```bash
# 同一 request_id 重复提交不会让机器人多走
curl -s .../step -d '{"steps": 5, "request_id": "abc"}'
curl -s .../step -d '{"steps": 5, "request_id": "abc"}'   # idempotent_replay=true
# 同 id 不同体 -> 409 IDEMPOTENCY_CONFLICT
```

### 3.4 非法/退化输入观察可区分错误

```bash
# 全封死地图
curl -s -X POST .../sessions -d '{"grid": ["###","###"]}'
#   -> 422 MAP_NO_FREE_SPACE

# 终点在封闭盒内
curl -s -X POST .../sessions -d '{"grid":["........","..####..",...], ...}'
#   -> 422 GOAL_UNREACHABLE

# 起点越界
#   -> 422 COORDINATE_OUT_OF_BOUNDS
```

## 4. 自动化测试

```bash
uv run pytest                      # 全部 37 个测试
uv run pytest -k planner           # 只跑规划相关
uv run pytest -k concurrency -v    # 并发隔离/串行/锁超时
```

测试日志默认输出每个用例的【输入】【关键判定】【状态迁移依据】（`navsim.test` logger），可直接复现：

- `test_planner.py::test_distance_transform_matches_brute_force`：距离变换对数百张随机地图逐格对齐暴力结果。
- `test_planner.py::test_narrow_gap_radius_decision`：1 格窄缝对 r=0（可过）、r=0.049（可过）、r=0.06（GOAL_UNREACHABLE）给出稳定区分。
- `test_http_sessions.py::test_dynamic_obstacle_forces_single_replan`：动态障碍恰好触发一次重规划并绕行到达。
- `test_http_sessions.py::test_blocked_then_clear_obstacle_recovers`：封死→BLOCKED（携带 replan_error 原因）→清除障碍→继续到达。
- `test_concurrency.py`：8 会话并发隔离、同会话 40 并发步进计数严格 1..40、锁超时可区分。
- `test_idempotency_http.py`：重复请求回放、冲突不污染状态。

## 5. 性能参考（本机，纯 Python）

| 场景 | 耗时 |
|---|---|
| 500×500 随机障碍(15%) 解析+规划 | < 1 s（默认 2 s 预算内） |
| 300×300 空旷地建会话（含规划） | 亚秒级 |
| 单个跟踪步进（弧段密集碰撞采样） | 约 20 µs |
| 500×500 障碍距离场(EDT) | 约 0.3 s |

## 6. 目录结构

```
app/
  errors.py       统一错误码与异常
  gridmap.py      多格式地图解析、坐标变换、圆盘碰撞
  kinematics.py   差分运动学、圆弧积分、指令钳制
  planner.py      EDT 精确膨胀 + 8 邻接 A*
  tracker.py      纯追踪 + 前瞻/弧段安全闸
  session.py      状态机、并发锁、幂等、动态障碍
  api.py          HTTP 路由与错误处理
  main.py         应用装配
tests/            37 个单元/集成/并发测试
docs/             本目录
```
