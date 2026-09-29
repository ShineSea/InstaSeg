# InstaSeg gRPC service

一个使用 **SAM 2.1** 的 Python gRPC 单图实例分割服务。它直接使用仓库根目录的
`instance_segmentation.proto`：客户端发送 RGB8 原始像素和一个正向提示点，服务返回该点对应
实例的逐像素二值掩码。

## 目录

```text
.
├── instance_segmentation.proto       # API 的唯一源文件
├── pyproject.toml                    # uv / Python 项目与依赖配置
├── scripts/generate_proto.py         # 重新生成 gRPC 代码
├── src/instaseg/
│   ├── config.py                     # 环境变量配置
│   ├── predictor.py                  # SAM 2.1 推理与并发保护
│   ├── service.py                    # 参数校验及 gRPC 实现
│   ├── server.py                     # 服务启动、关闭与消息大小限制
│   └── generated/                    # 已生成的 protobuf 代码
└── tests/test_service.py             # 无需模型权重的接口测试
```

## 安装与启动

需要 Python 3.10+。NVIDIA GPU 环境应先按照 PyTorch 官方说明选择匹配 CUDA 的 wheel；下面的
`uv sync` 会创建 `.venv` 并安装其余依赖（包括 Meta 官方 SAM 2 仓库）：

```bash
uv sync --dev
uv run instaseg-server
```

默认使用 `facebook/sam2.1-hiera-small`，首次启动会从 Hugging Face 下载模型。可通过环境变量
调整，完整示例见 `.env.example`。PowerShell 示例：

```powershell
$env:INSTASEG_DEVICE = "cuda"
$env:INSTASEG_MODEL_ID = "facebook/sam2.1-hiera-small"
uv run instaseg-server
```

也可以完全使用本地权重；必须同时提供权重和官方配置名：

```powershell
$env:INSTASEG_CHECKPOINT = "checkpoints/sam2.1_hiera_small.pt"
$env:INSTASEG_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"
uv run instaseg-server
```

CPU 可运行但速度较慢，设置 `INSTASEG_DEVICE=cpu` 即可。服务默认监听 `0.0.0.0:50051`。

## 调用约定

- `rgb8` 必须是无行填充的、从左上角开始逐行排列的 RGB 字节，长度严格为
  `width * height * 3`。
- `(x, y)` 是实例内部的正向提示点，坐标必须位于图像内。
- `mask` 长度严格为 `width * height`，每字节只能是 `0` 或 `1`。
- SAM 返回多个候选时，服务选择模型置信分数最高的候选。
- 推理器会缓存当前图像状态，因此请求通过进程内锁串行进入模型；gRPC 校验仍可并发执行。
  如需更高吞吐，建议每张 GPU 启动一个服务进程并在其前面做负载均衡。

## 开发

修改 proto 后重新生成绑定并测试：

```bash
uv run python scripts/generate_proto.py
uv run pytest
uv run ruff check .
```

