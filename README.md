# 移动机器人导航仿真（本地 HTTP 服务）

基于占用栅格与差速运动学约束的全局规划（A\*）+ 局部轨迹跟踪（纯追踪），
支持会话级状态管理、并发隔离、动态障碍重规划、幂等请求，以及面向长时仿真的
**按时长安全推进 / 执行轨迹查询 / 只读预演**。

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
nav/grid.py          占用栅格：建图/校验/坐标/足迹碰撞/膨胀/动态层/线段扫掠胶囊
nav/planner.py       A* 全局规划与路径通畅性检查
nav/controller.py    纯追踪差速轨迹跟踪（v/ω/曲率/加速度包络）
nav/preview.py       只读预演（确定性纯函数，锁外执行，不改会话）
nav/session.py       会话状态机：规划/自动重规划/单步/按时长推进/轨迹缓冲
nav/service.py       多会话管理、TTL、幂等、参数解析
nav/api.py           HTTP 路由与统一错误响应
tests/               85 个单测（含长跨度不穿障/轨迹一致/预演可复现）
docs/navigation.md   详细设计：运动学/代价/参数/状态机/并发错误语义/长时仿真
```

## 关键能力

- 不可达 / 起终点非法 / 越界 / 地图退化均以**稳定可区分错误码**拒绝；
- 执行期动态障碍自动重规划，任何时刻轨迹**不穿障、不越界、不超速度与曲率**；
- **按时长推进**：`POST /sessions/{id}/advance` 一次模拟 `duration` 秒
  （≤60s），内部按 `dt` 切微步逐帧限幅、相邻位姿间做**扫掠胶囊**精确碰撞复核，
  单步位移比墙厚/通道宽也不会穿障；被挡时“能走多少走多少”，原地停住后仍可
  撤障/重规划/继续；可临时调高 `max_v`；
- **轨迹查询**：`GET /sessions/{id}/trajectory` 返回含中间扫过位姿的轨迹，
  与上报序列/步数一一对应；每会话独立环形缓冲，默认保留 **2000** 条
  （可配 1~100000），超限 FIFO 丢最旧并给出 `dropped/truncated` 统计；
- **只读预演**：`POST /sessions/{id}/preview` 用给定路线与 `dt/max_v` 判定能否
  安全走完、返回失败顶点/原因与全程最近间隙；确定性可复现、不改会话、
  锁外执行不阻塞其它会话，大图毫秒级；
- 多会话并发隔离；同会话步进/重规划/查询串行化，无错序与状态撕裂；
- 重复请求幂等，超时请求返回可区分错误，失败后状态可继续执行；
- **向后兼容**：不传任何新参数时，`step` 行为与响应字段不变，
  `advance(duration=dt)` 与连续 `step` 逐点位姿一致。

详见 [`docs/navigation.md`](docs/navigation.md)。
