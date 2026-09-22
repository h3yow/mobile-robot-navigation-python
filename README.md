# 移动机器人导航仿真与轨迹跟踪（本地 HTTP 服务）

基于占用栅格与差分驱动运动学的本地导航仿真：A* 全局规划（按机器人半径精确膨胀）、
纯追踪局部跟踪（弧段密集碰撞复核，绝不穿障/超速/超曲率）、动态障碍触发自动重规划、
会话级状态机与并发隔离、请求幂等，以及对多种地图与退化输入的可区分错误判定。

## 快速开始

```bash
uv sync
uv run fastapi run main.py --port 8000      # 打开 http://127.0.0.1:8000/docs
uv run pytest                                # 37 个测试
```

## 文档

- [运动学与代价模型、规划与跟踪参数](docs/01-model-and-parameters.md)
- [会话状态机、并发与错误语义](docs/02-session-state-machine.md)
- [本地运行与验证方法（含 curl 示例）](docs/03-local-verification.md)
