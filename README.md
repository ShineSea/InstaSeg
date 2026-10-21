# InstaSeg HTTP / gRPC service

一个使用 **SAM 2.1** 的 Python gRPC 单图实例分割服务。它直接使用仓库根目录的
`instance_segmentation.proto`：客户端发送 RGB8 原始像素和一个正向提示点，服务返回该点对应
实例的逐像素二值掩码。

另外提供独立的 HTTP 服务和内置网页，支持上传图片、选点或画框分割，在网页中查看与下载结果。
HTTP 服务复用服务端配置的 SAM 2.1 模型，网页和请求均不提供模型选择。原有 proto、gRPC
启动命令与接口约定保持不变。

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
│   ├── http_server.py                # HTTP 接口与独立启动入口
│   ├── web/                          # 内置网页（无需前端构建或外部 CDN）
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

## HTTP 服务与网页测试

安装依赖后，使用独立命令启动 HTTP 服务：

```bash
uv run instaseg-http
```

模型加载完成后，在浏览器打开 <http://localhost:8000>，接口文档位于
<http://localhost:8000/docs>。首次启动可能需要下载权重。HTTP 默认监听 `0.0.0.0:8000`，
通过 `INSTASEG_HTTP_HOST`、`INSTASEG_HTTP_PORT` 修改；默认上传限制为 20 MiB，
可通过 `INSTASEG_HTTP_MAX_UPLOAD_BYTES` 修改。图片像素数受 `INSTASEG_MAX_IMAGE_PIXELS`
限制，默认 2500 万。`.env.example` 为配置参考，启动前需要将配置设置为环境变量。

1. 点击上传或拖入 JPEG、PNG、WebP、BMP 图片。
2. 选择「选点」并点击目标内部，或选择「画框」并拖动圈住完整目标。
3. 点击「开始分割」，查看叠加图、黑白掩码或透明背景抠图，并下载 PNG 结果。
4. 重新标记可再次分割；切换模式或更换图片会清除旧标记和结果。

HTTP 与 gRPC 是独立进程，各自加载模型；只做网页测试时仅启动 HTTP 即可。

### HTTP API

`GET /api/health` 返回就绪状态与图片大小限制。模型在启动期间加载，加载失败则服务启动失败。

`POST /api/segment` 使用 `multipart/form-data`，必填字段：

| 字段 | 含义 |
| --- | --- |
| `image` | 图片文件 |
| `mode` | `point`（选点）或 `box`（画框） |
| `prompt` | JSON 数组：选点 `[x,y]`；画框 `[x1,y1,x2,y2]` |

坐标以 EXIF 方向修正后的原图左上角为原点，不使用页面缩放后的显示尺寸。
点坐标范围为 `0 <= x < width`、`0 <= y < height`，小数向下取整。
框坐标必须满足 `0 <= x1 < x2 <= width`、`0 <= y1 < y2 <= height`。

```bash
curl http://localhost:8000/api/segment -F "image=@sample.jpg" -F "mode=point" -F "prompt=[120,80]"
curl http://localhost:8000/api/segment -F "image=@sample.jpg" -F "mode=box" -F "prompt=[20,30,240,200]"
```

返回 JSON 包含 `width`、`height`、`foreground_pixels`、`elapsed_ms` 和三个 PNG Data URL：
`mask` 为像素值 0/255 的单通道掩码，`overlay` 为绿色叠加图，`cutout` 为 RGBA 透明背景图。
图片与输出不持久化保存；上传使用框架临时文件，请求结束后关闭。
无效图片返回 400，大小或像素数超限返回 413，无效提示返回 422，推理失败返回 500。

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

