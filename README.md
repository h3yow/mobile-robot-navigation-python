# 移动机器人导航仿真（本地 HTTP 服务）

基于占用栅格与差速运动学约束的全局规划（A\*）+ 局部轨迹跟踪（纯追踪），
支持会话级状态管理、并发隔离、动态障碍重规划、幂等请求，
以及**长时推进（按时长/可调速度上限）、执行轨迹查询、只读预演**。

## 快速开始

```bash
uv sync
uv run pytest -s                     # 运行全部单测（含输入/判定/迁移日志）
uv run uvicorn main:app --port 8000  # 启动服务，交互文档见 /docs
```

最小调用：

```bash
SID=$(curl -s -X POST localhost:8000/sessions -H 'content-type: application/json' \
  -d '{"map":{"cells":[[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],"resolution":1}}' \
  | python3 -c 'import sys,json;print(json.load(sys.stdin)["session_id"])')

curl -s -X POST localhost:8000/sessions/$SID/plan -H 'content-type: application/json' \
  -d '{"start":{"x":0.5,"y":0.5,"theta":0},"goal":{"x":3.5,"y":3.5}}'
curl -s -X POST localhost:8000/sessions/$SID/step -H 'content-type: application/json' -d '{}'
curl -s localhost:8000/sessions/$SID

# 一次推进 10 秒（微步 0.5s、速度上限 3m/s），再查轨迹、做只读预演
curl -s -X POST localhost:8000/sessions/$SID/advance -H 'content-type: application/json' \
  -d '{"duration":10.0,"dt":0.5,"max_v":3.0}'
curl -s localhost:8000/sessions/$SID/trajectory
curl -s -X POST localhost:8000/sessions/$SID/rehearse -H 'content-type: application/json' -d '{}'
```

## 目录

```
main.py              FastAPI 入口
nav/grid.py          占用栅格：建图/校验/坐标/足迹碰撞/膨胀/动态层
nav/planner.py       A* 全局规划与路径通畅性检查
nav/controller.py    纯追踪差速轨迹跟踪（v/ω/曲率/加速度包络、精确圆弧/扫掠采样）
nav/session.py       会话状态机：规划/自动重规划/单步/按时长推进/轨迹缓冲/BLOCKED 恢复
nav/rehearse.py      只读预演：可行性/受阻位置/全程净空（不改会话、可复现）
nav/service.py       多会话管理、TTL、幂等、参数解析、advance/trajectory/rehearse
nav/api.py           HTTP 路由与统一错误响应
tests/               86 个单测（长时推进/轨迹/预演/老行为兼容 + 动态重规划/并发隔离/非法输入）
docs/navigation.md   详细设计：运动学与代价模型、参数、状态机、长时运行三能力、并发与错误语义
```

## 关键能力

- 不可达 / 起终点非法 / 越界 / 地图退化均以**稳定可区分错误码**拒绝；
- 执行期动态障碍自动重规划，任何时刻轨迹**不穿障、不越界、不超速度与曲率**；
- **按时长一次推进**（`advance`：秒级 `dt`、可调高速），逐微步做圆弧扫掠复核，
  位移远大于墙厚/通道宽也不穿墙；走不下则“能走多少走多少”——部分微步的
  速度指令按**实际执行时长**重新限幅（加速度不超限），并停在安全位置；
  前路被封死且绕不过去时**尽量推进到最靠近的安全位置**并说明原因，
  清障/换环境/重规划后可继续，不会卡死；
- **执行轨迹查询**（`trajectory`）：每条微步的稠密扫掠位姿与上报位姿/步数严格对齐，
  每会话 FIFO 保留（默认 2000 条，超上限丢最旧），会话间隔离；
- **只读预演**（`rehearse`）：按给定路线与参数判断能否安全走完、报告受阻位置与
  全程最小净空，结果可复现、不改会话状态、不阻塞其它会话，大图可及时返回；
- 多会话并发隔离；同会话步进/重规划/查询串行化，无错序与状态撕裂；
- 重复请求幂等，超时请求返回可区分错误，失败后状态可继续执行。
- **不传新参数时逐步 `step` 行为与旧版逐位姿一致**（新能力只作用于 advance
  微步路径；详见文档“向后兼容与默认值”一节）。

详见 [`docs/navigation.md`](docs/navigation.md)。
