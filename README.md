# 移动机器人导航仿真（本地 HTTP 服务）

基于占用栅格与差速运动学约束的全局规划（A\*）+ 局部轨迹跟踪（纯追踪），
支持会话级状态管理、并发隔离、动态障碍重规划与幂等请求。

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
```

## 目录

```
main.py              FastAPI 入口
nav/grid.py          占用栅格：建图/校验/坐标/足迹碰撞/膨胀/动态层
nav/planner.py       A* 全局规划与路径通畅性检查
nav/controller.py    纯追踪差速轨迹跟踪（v/ω/曲率/加速度包络）
nav/session.py       会话状态机：规划/自动重规划/单步执行/BLOCKED 恢复
nav/service.py       多会话管理、TTL、幂等、参数解析
nav/api.py           HTTP 路由与统一错误响应
tests/               53 个单测（动态重规划/并发隔离/非法与退化输入）
docs/navigation.md   详细设计：运动学与代价模型、参数、状态机、并发与错误语义
```

## 关键能力

- 不可达 / 起终点非法 / 越界 / 地图退化均以**稳定可区分错误码**拒绝；
- 执行期动态障碍自动重规划，任何时刻轨迹**不穿障、不越界、不超速度与曲率**；
- 多会话并发隔离；同会话步进/重规划/查询串行化，无错序与状态撕裂；
- 重复请求幂等，超时请求返回可区分错误，失败后状态可继续执行。

详见 [`docs/navigation.md`](docs/navigation.md)。
