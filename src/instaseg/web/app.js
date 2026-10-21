"use strict";

const $ = (id) => document.getElementById(id);
const canvas = $("canvas");
const ctx = canvas.getContext("2d");
const state = {
  image: null,
  file: null,
  mode: "point",
  prompt: null,
  start: null,
  result: null,
  view: "overlay",
  busy: false,
  ready: false,
  loading: false,
  loadId: 0,
  maxBytes: 20 * 1024 * 1024,
  maxPixels: 25000000,
};

function status(message, error = false) {
  $("status").textContent = message;
  $("status").classList.toggle("error", error);
}

function updateControls() {
  $("run").disabled =
    !state.ready || !state.prompt || state.busy || state.loading;
  $("clear").disabled = !state.prompt || state.busy || state.loading;
  $("file").disabled = state.busy;
  document.querySelectorAll(".mode").forEach((button) => {
    button.disabled = state.busy;
  });
  $("run").innerHTML = state.busy ? "正在分割…" : "开始分割 <span>↗</span>";
  $("prompt-info").textContent = state.prompt
    ? (state.mode === "point" ? "坐标 " : "选框 ") + state.prompt.join(", ")
    : "尚未标记目标";
}

function clearResult() {
  state.result = null;
  $("result").hidden = true;
  $("result").removeAttribute("src");
  $("result-empty").hidden = false;
  $("stats").textContent = "结果将以原图分辨率输出";
  ["download-mask", "download-cutout"].forEach((id) => {
    $(id).hidden = true;
    $(id).removeAttribute("href");
  });
}

function draw() {
  if (!state.image) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(state.image, 0, 0);
  if (!state.prompt) return;
  const scale =
    canvas.width / Math.max(1, canvas.getBoundingClientRect().width);
  ctx.strokeStyle = "#22e6a4";
  ctx.fillStyle = "rgba(34, 230, 164, .14)";
  ctx.lineWidth = 2 * scale;
  if (state.mode === "point") {
    const [x, y] = state.prompt;
    ctx.beginPath();
    ctx.arc(x, y, 6 * scale, 0, Math.PI * 2);
    ctx.fillStyle = "#22e6a4";
    ctx.fill();
    ctx.strokeStyle = "#ffffff";
    ctx.stroke();
  } else {
    const [x1, y1, x2, y2] = state.prompt;
    ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
  }
}

function fitCanvas() {
  if (!state.image) return;
  const stage = $("input-stage");
  const scale = Math.min(
    (stage.clientWidth - 28) / canvas.width,
    (stage.clientHeight - 28) / canvas.height,
    1,
  );
  canvas.style.width = `${canvas.width * scale}px`;
  canvas.style.height = `${canvas.height * scale}px`;
  draw();
}
new ResizeObserver(fitCanvas).observe($("input-stage"));

async function loadFile(file) {
  if (!file || state.busy) return;
  const loadId = ++state.loadId;
  state.loading = true;
  updateControls();
  let url;
  try {
    if (file.size > state.maxBytes)
      throw new Error("图片文件超过上传大小限制。");
    if (!/\.(jpe?g|png|webp|bmp)$/i.test(file.name)) {
      throw new Error("请选择 JPEG、PNG、WebP 或 BMP 图片。");
    }
    url = URL.createObjectURL(file);
    const image = new Image();
    image.src = url;
    await image.decode();
    if (loadId !== state.loadId) return;
    if (image.naturalWidth * image.naturalHeight > state.maxPixels) {
      throw new Error("图片像素数超过服务限制，请缩小后重试。");
    }
    state.image = image;
    state.file = file;
    state.prompt = null;
    state.start = null;
    clearResult();
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;
    canvas.hidden = false;
    $("input-empty").hidden = true;
    $("dimensions").textContent = `${canvas.width} × ${canvas.height}`;
    $("file-info").textContent =
      `${file.name} · ${(file.size / 1024 / 1024).toFixed(2)} MB`;
    fitCanvas();
    status(
      state.mode === "point"
        ? "请点击目标内部的一个点。"
        : "请拖动画出包含目标的选框。",
    );
  } catch (error) {
    if (loadId === state.loadId) status(error.message, true);
  } finally {
    if (url) URL.revokeObjectURL(url);
    if (loadId === state.loadId) {
      state.loading = false;
      updateControls();
    }
    $("file").value = "";
  }
}

$("file").addEventListener("change", (event) =>
  loadFile(event.target.files[0]),
);
const dropzone = $("dropzone");
dropzone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    if (!state.busy) $("file").click();
  }
});
["dragenter", "dragover"].forEach((name) =>
  dropzone.addEventListener(name, (event) => {
    event.preventDefault();
    if (!state.busy) dropzone.classList.add("dragover");
  }),
);
["dragleave", "drop"].forEach((name) =>
  dropzone.addEventListener(name, (event) => {
    event.preventDefault();
    dropzone.classList.remove("dragover");
    if (name === "drop") loadFile(event.dataTransfer.files[0]);
  }),
);

document.querySelectorAll(".mode").forEach((button) =>
  button.addEventListener("click", () => {
    if (state.busy || state.mode === button.dataset.mode) return;
    state.mode = button.dataset.mode;
    state.prompt = null;
    state.start = null;
    clearResult();
    document.querySelectorAll(".mode").forEach((item) => {
      item.classList.toggle("active", item === button);
      item.setAttribute("aria-pressed", String(item === button));
    });
    $("hint").textContent =
      state.mode === "point"
        ? "在左侧图片中，点击目标内部的一个点。"
        : "在左侧图片中，拖动画框圈住完整目标。";
    status(state.image ? "请重新标记目标。" : "上传图片后即可开始。");
    draw();
    updateControls();
  }),
);

function position(event, edge = false) {
  const rect = canvas.getBoundingClientRect();
  return [
    Math.max(
      0,
      Math.min(
        canvas.width - (edge ? 0 : 1),
        Math.floor(((event.clientX - rect.left) * canvas.width) / rect.width),
      ),
    ),
    Math.max(
      0,
      Math.min(
        canvas.height - (edge ? 0 : 1),
        Math.floor(((event.clientY - rect.top) * canvas.height) / rect.height),
      ),
    ),
  ];
}
function boxFrom(start, end) {
  return [
    Math.min(start[0], end[0]),
    Math.min(start[1], end[1]),
    Math.max(start[0], end[0]),
    Math.max(start[1], end[1]),
  ];
}

canvas.addEventListener("pointerdown", (event) => {
  if (
    !state.image ||
    state.busy ||
    state.loading ||
    event.button !== 0 ||
    !event.isPrimary
  )
    return;
  canvas.setPointerCapture(event.pointerId);
  clearResult();
  const point = position(event, state.mode === "box");
  state.start = state.mode === "box" ? point : null;
  state.prompt = state.mode === "point" ? point : null;
  draw();
  updateControls();
  status(
    state.mode === "point"
      ? "目标已标记，可以开始分割。"
      : "拖动到目标的另一个角后松开。",
  );
});
canvas.addEventListener("pointermove", (event) => {
  if (!state.start || state.busy || !canvas.hasPointerCapture(event.pointerId))
    return;
  state.prompt = boxFrom(state.start, position(event, true));
  draw();
});
canvas.addEventListener("pointerup", (event) => {
  if (!canvas.hasPointerCapture(event.pointerId)) return;
  canvas.releasePointerCapture(event.pointerId);
  if (!state.start) return;
  state.prompt = boxFrom(state.start, position(event, true));
  state.start = null;
  const [x1, y1, x2, y2] = state.prompt;
  if (x1 === x2 || y1 === y2) {
    state.prompt = null;
    status("选框需要一定宽度和高度，请重新拖动画框。", true);
  } else status("选框已标记，可以开始分割。");
  draw();
  updateControls();
});
canvas.addEventListener("pointercancel", () => {
  state.start = null;
  state.prompt = null;
  draw();
  updateControls();
});
$("clear").addEventListener("click", () => {
  state.prompt = null;
  state.start = null;
  clearResult();
  draw();
  updateControls();
  status("标记已清除，请重新标记目标。");
});

document.querySelectorAll(".view").forEach((button) =>
  button.addEventListener("click", () => {
    state.view = button.dataset.view;
    document.querySelectorAll(".view").forEach((item) => {
      item.classList.toggle("active", item === button);
      item.setAttribute("aria-pressed", String(item === button));
    });
    if (state.result) $("result").src = state.result[state.view];
  }),
);

$("run").addEventListener("click", async () => {
  if (!state.prompt || !state.file || state.busy) return;
  state.busy = true;
  clearResult();
  updateControls();
  status("正在上传并分割图片，请稍候…");
  try {
    const form = new FormData();
    form.append("image", state.file);
    form.append("mode", state.mode);
    form.append("prompt", JSON.stringify(state.prompt));
    const response = await fetch("/api/segment", {
      method: "POST",
      body: form,
    });
    const data = await response.json();
    if (!response.ok)
      throw new Error(
        typeof data.detail === "string" ? data.detail : "请求参数无效。",
      );
    state.result = data;
    $("result").src = data[state.view];
    $("result").hidden = false;
    $("result-empty").hidden = true;
    const ratio = (
      (data.foreground_pixels / (data.width * data.height)) *
      100
    ).toFixed(1);
    $("stats").textContent = `${data.elapsed_ms} ms · 目标占比 ${ratio}%`;
    for (const key of ["mask", "cutout"]) {
      $(`download-${key}`).href = data[key];
      $(`download-${key}`).hidden = false;
    }
    status(
      data.foreground_pixels
        ? "分割完成，可切换视图或下载结果。"
        : "未识别到目标，请调整标记后重试。",
    );
  } catch (error) {
    status(`分割失败：${error.message}`, true);
  } finally {
    state.busy = false;
    updateControls();
  }
});

async function connect() {
  try {
    const response = await fetch("/api/health");
    if (!response.ok) throw new Error("unavailable");
    const health = await response.json();
    state.maxBytes = health.max_upload_bytes;
    state.maxPixels = health.max_image_pixels;
    state.ready = true;
    $("limits").textContent =
      `最大 ${(state.maxBytes / 1024 / 1024).toFixed(0)} MB · ${(state.maxPixels / 1000000).toFixed(1)} 百万像素`;
    $("connection").textContent = "服务已就绪";
    $("connection").classList.add("ready");
    updateControls();
  } catch {
    $("connection").textContent = "服务未连接";
    status("暂时无法连接服务，正在重试…", true);
    setTimeout(connect, 5000);
  }
}
connect();
