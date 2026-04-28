# Batch Upload API — 批量上传文件到 RAGFlow 知识库

## 概述

提供 `POST /ragflow/batch-upload` 端点，根据 JSON 清单（manifest）批量上传文件到 RAGFlow 知识库，支持为每个文件单独指定：
- **目标知识库** (knowledge_base)
- **元数据字段** (meta_fields，如 series / model / machine_type / customer / date)
- **解析方法** (chunk_method: naive / table / manual / qa / paper / book / …)

上传后自动触发解析并轮询等待完成。内置**递归文件搜索**、**上传/解析双重重试**、**全局并发控制**、**文件间冷却间隔**四层防护。

---

## 快速开始

### 1. 准备好文件

把 `manifest.json` 和数据文件放到服务器上的同一个文件夹（可嵌套子目录）：

```
/data/ragflow_files/
├── manifest.json                  # JSON 清单
├── 产品库/
│   ├── 产品信息表-260107WF.xlsx
│   └── EST产品尺寸、重量(2023.2.20).pdf
├── 经验库/
│   └── 品管事件处理表-251204.xlsx
└── 程序库/
    └── 程序库2025-12-18.xlsx
```

### 2. 准备 manifest.json

```json
[
  {
    "file_name": "液压卧式注塑机机器配置对照表-20240102.xlsx",
    "meta": {
      "series": null,
      "model": null,
      "machine_type": "卧式",
      "customer": null,
      "date": "20240102"
    },
    "knowledge_base": "产品库",
    "chunk_method": "table"
  },
  {
    "file_name": "EST产品尺寸、重量(2023.2.20).pdf",
    "meta": {
      "series": "ES",
      "model": "EST",
      "machine_type": null,
      "customer": null,
      "date": "2023.2.20"
    },
    "knowledge_base": "产品库",
    "chunk_method": "manual"
  },
  {
    "file_name": "经验库--品管事件处理表-251204.xlsx",
    "meta": {
      "series": null,
      "model": null,
      "machine_type": null,
      "customer": null,
      "date": "251204"
    },
    "knowledge_base": "经验库",
    "chunk_method": "table"
  },
  {
    "file_name": "程序库2025-12-18.xlsx",
    "meta": {
      "series": null,
      "model": null,
      "machine_type": null,
      "customer": null,
      "date": "2025-12-18"
    },
    "knowledge_base": "程序库",
    "chunk_method": "table"
  },
  {
    "file_name": "产品信息表-260107WF.xlsx",
    "meta": {
      "series": null,
      "model": "ES260卧式",
      "machine_type": null,
      "customer": null,
      "date": null
    },
    "knowledge_base": "产品库",
    "chunk_method": "table"
  },
  {
    "file_name": "震雄ES650卧式用户手册.pdf",
    "meta": {
      "series": "ES",
      "model": "ES650",
      "machine_type": "卧式",
      "customer": "震雄",
      "date": null
    },
    "knowledge_base": "产品库",
    "chunk_method": "manual"
  }
]
```

### 3. 调用接口

```bash
curl -X POST http://localhost:8123/ragflow/batch-upload \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "manifest_path": "/data/ragflow_files/manifest.json",
    "folder_path": "/data/ragflow_files"
  }'
```

文件会被**递归搜索**找到，不管放在 `folder_path` 下的哪个子目录。

---

## API 参考

**端点**：`POST /ragflow/batch-upload`
**认证**：Admin（`Authorization: Bearer <token>`）
**Content-Type**：`application/json`

### 请求参数

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `folder_path` | string | 是 | 服务器上文件所在根目录的绝对路径 |
| `manifest_path` | string | 二选一 | 服务器上 JSON 清单文件的绝对路径 |
| `manifest` | array | 二选一 | 内联 JSON 清单数组 |
| `skip_duplicate_check` | bool | 否 | 跳过重名检测，默认 `false` |
| `auto_parse` | bool | 否 | 自动触发解析并等待完成，默认 `true` |

**注意**：`manifest_path` 和 `manifest` 不能同时提供。

manifest 每项的字段：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `file_name` | string | 是 | 文件名（会在 `folder_path` 下递归搜索） |
| `knowledge_base` | string | 是 | 目标知识库：产品库/图片库/视频库/通用库/程序库/经验库 |
| `meta` | object | 否 | 元数据，`null` 值会被自动剔除 |
| `chunk_method` | string | 否 | 解析方法，默认 `"naive"` |

### 方式 A：用 JSON 文件（推荐）

```bash
curl -X POST http://localhost:8123/ragflow/batch-upload \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "manifest_path": "/data/ragflow_files/manifest.json",
    "folder_path": "/data/ragflow_files"
  }'
```

### 方式 B：内联 JSON 数组

```bash
curl -X POST http://localhost:8123/ragflow/batch-upload \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "manifest": [
      {"file_name": "产品信息表.xlsx", "knowledge_base": "产品库", "chunk_method": "table", "meta": {"model": "ES260"}}
    ],
    "folder_path": "/data/ragflow_files"
  }'
```

### 递归搜索说明

系统会从 `folder_path` 开始**深度优先遍历**所有子文件夹寻找 `file_name`：
- 先在根目录找 → 再逐层深入子文件夹
- 自动跳过隐藏文件夹（`.` 开头，如 `.git`）
- 返回找到的第一个匹配文件
- 如果 `folder_path` 下存在多个同名文件，只有第一个被使用

```
folder_path=/data/ragflow_files, file_name="产品信息表.xlsx"

查找顺序:
  1. /data/ragflow_files/产品信息表.xlsx          ← 先看根目录
  2. /data/ragflow_files/产品库/产品信息表.xlsx    ← 再遍历子目录
  3. /data/ragflow_files/归档/2024/产品信息表.xlsx ← 任意深度都能找到
```

---

## 响应格式

```json
{
  "total": 6,
  "success": 5,
  "skipped": 1,
  "not_found": 0,
  "error": 0,
  "results": [
    {
      "file_name": "液压卧式注塑机机器配置对照表-20240102.xlsx",
      "knowledge_base": "产品库",
      "status": "success",
      "doc_id": "abc123...",
      "dataset_id": "def456...",
      "dataset_name": "产品库",
      "parse_status": "DONE",
      "doc_url": "http://localhost:9000/kb_abc123/my_document.pdf?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=minioadmin%2F20260428%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20260428T120000Z&X-Amz-Expires=3600&X-Amz-SignedHeaders=host&X-Amz-Signature=...",
      "message": "Uploaded and parsed successfully"
    },
    {
      "file_name": "EST产品尺寸、重量(2023.2.20).pdf",
      "knowledge_base": "产品库",
      "status": "skipped",
      "doc_id": "existing-doc-id",
      "dataset_id": "def456...",
      "dataset_name": "产品库",
      "parse_status": null,
      "message": "Duplicate — already exists as doc_id=existing-doc-id"
    },
    {
      "file_name": "不存在的文件.pdf",
      "knowledge_base": "产品库",
      "status": "not_found",
      "doc_id": null,
      "dataset_id": null,
      "dataset_name": null,
      "parse_status": null,
      "message": "File not found recursively in: /data/ragflow_files"
    },
    {
      "file_name": "损坏的文件.xlsx",
      "knowledge_base": "产品库",
      "status": "error",
      "doc_id": "ghi789...",
      "dataset_id": "def456...",
      "dataset_name": "产品库",
      "parse_status": "FAIL",
      "doc_url": "http://localhost:9000/kb_ghi789/broken_file.xlsx?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=...",
      "message": "Uploaded but parsing FAILED: Parsing failed after 3 attempt(s)"
    }
  ]
}
```

| 字段 | 说明 |
|---|---|
| `total` | 清单文件总数 |
| `success` | 上传且解析成功 |
| `skipped` | 因重名跳过 |
| `not_found` | 文件未找到 |
| `error` | 上传或解析失败 |
| `results[].status` | `success` / `skipped` / `not_found` / `error` |
| `results[].parse_status` | `DONE` / `FAIL` / `CANCEL` / `TIMEOUT` / `SKIPPED` / `TRIGGER_FAILED` |

---

## 知识库类型映射

| manifest 中填写 | .env 配置 | 用途 |
|---|---|---|
| `"产品库"` | `PRODUCT_KB_NAME` | 用户手册、装配手册、规格书、配置表 |
| `"图片库"` | `IMAGE_KB_NAME` | 产品图片 |
| `"视频库"` | `VIDEO_KB_NAME` | 产品使用维护视频 |
| `"文件库"` | `FILE_KB_NAME` | 文件库 |
| `"通用库"` | `GENERAL_KB_NAME` | 国标/机械行业标准 |
| `"程序库"` | `PROGRAM_KB_NAME` | 程序功能说明 |
| `"经验库"` | `EXPERIENCE_KB_NAME` | 经验处理案例、故障排查 |

系统在 RAGFlow 中按关键词**模糊匹配**数据集名称，修改 `.env` 即可适配不同命名。

### chunk_method 推荐用法

| 文件类型 | chunk_method |
|---|---|
| PDF / Word 文档 | `manual` 或 `naive` |
| Excel 表格 | `table` |
| 图片 | `picture` |
| 知识图谱 | `knowledge-graph` |

完整可选值：`naive` | `manual` | `qa` | `table` | `paper` | `book` | `laws` | `presentation` | `picture` | `one` | `knowledge-graph` | `email`

---

## 文件下载 URL — MinIO 直链

### 背景

RAGFlow 内部使用 MinIO 存储所有上传的文档。上传文件后，系统自动生成 **MinIO presigned GET URL**，指向源文件本身（而非 RAGFlow 预览页面）。前端拿到这个 URL 后可以直接使用：

| 文件类型 | 前端用法 |
|---------|---------|
| 图片 (png/jpg/webp) | `<img src="{url}">` 直接渲染 |
| 视频 (mp4/mov/webm) | `<video src="{url}">` 直接播放 |
| 文档 (pdf/xlsx/docx) | `<a href="{url}">` 直接下载 |

### 原理

```
文件上传到 RAGFlow
       │
       ▼
RAGFlow 将文件存入 MinIO:
    bucket  = kb_id (dataset_id)
    object  = location (filename)
       │
       ▼
construct_doc_url() 从 RAGFlow 文档响应中拿到
kb_id + location，调用 MinIO SDK 生成
presigned GET URL（默认 1 小时有效）
       │
       ▼
URL 格式: http://minio:9000/{bucket}/{object}?
          X-Amz-Algorithm=AWS4-HMAC-SHA256&
          X-Amz-Credential=...&
          X-Amz-Signature=...
```

### 相比 RAGFlow 预览 URL 的优势

| | RAGFlow 预览 URL | MinIO 直链 |
|---|---|---|
| 实际内容 | RAGFlow 管理后台 HTML 页面 | 文件的原始二进制内容 |
| 图片渲染 | 无法直接嵌入 `<img>` | 可直接嵌入 |
| 视频播放 | 无法直接播放 | 可直接播放 |
| 文件下载 | 用户需要手动操作 | 浏览器自动下载 |
| 有效期 | 依赖 RAGFlow 会话 | 可配置（默认 1 小时） |
| 回退策略 | — | MinIO 不可用时自动回退到预览 URL |

### 配置方式

在 `.env` 中配置 RAGFlow 内置 MinIO 的凭据：

```bash
# RAGFlow 内置 MinIO 默认凭据
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_SECURE=false
MINIO_PRESIGNED_EXPIRY_SECONDS=3600
```

> **注意**：如果不配置 `MINIO_ACCESS_KEY`（留空），系统自动回退到 RAGFlow 预览 URL，不会影响现有功能。

### 获取 RAGFlow 的 MinIO 凭据

RAGFlow docker-compose 部署时自带 MinIO，凭据在其 `.env` 或 `docker-compose.yml` 中：

```yaml
# ragflow/docker/docker-compose.yml
minio:
  image: minio/minio:RELEASE.2024-12-18T13-15-44Z
  environment:
    MINIO_ROOT_USER: minioadmin      # → MINIO_ACCESS_KEY
    MINIO_ROOT_PASSWORD: minioadmin  # → MINIO_SECRET_KEY
```

## 环境变量完整列表

```bash
# RAGFlow 服务
RAGFLOW_BASE_URL=http://10.32.1.172:9222
RAGFLOW_API_KEY=ragflow-your-api-key-here

# 知识库数据集名称匹配关键词
PRODUCT_KB_NAME=产品库
IMAGE_KB_NAME=图片库
VIDEO_KB_NAME=视频库
FILE_KB_NAME=文件库
GENERAL_KB_NAME=通用库
PROGRAM_KB_NAME=程序库
EXPERIENCE_KB_NAME=经验库

# MinIO / S3 — RAGFlow 内置 MinIO（用于生成文件下载直链）
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_SECURE=false
MINIO_PRESIGNED_EXPIRY_SECONDS=3600

# 批量上传限流
BATCH_UPLOAD_MAX_CONCURRENT=1       # 同时允许的批量上传请求数（排队，建议 1-2）
BATCH_UPLOAD_COOLDOWN_SECONDS=2.0   # 文件间冷却间隔（秒，0 表示不等待）
```

---

## 资源保护与重试机制

### 四层防护

```
请求到达
  ↓
┌── 1. 全局 Semaphore ──────────────────────────────────┐
│   (BATCH_UPLOAD_MAX_CONCURRENT=1，多余请求 FIFO 排队)    │
│                                                        │
│  文件 N:                                               │
│    ┌── 2. 上传重试（指数退避: 2s → 4s，共 3 次）────┐   │
│    │  HTTP POST 文件到 RAGFlow                        │   │
│    └─────────────────────────────────────────────────┘   │
│      ↓                                                   │
│    设置 meta_fields / chunk_method / parser_config        │
│      ↓                                                   │
│    ┌── 3. 解析重试（轮询等待，失败重试 3 次）────────┐   │
│    │  触发解析 → 每 3s 轮询 → DONE / FAIL / TIMEOUT    │   │
│    │  最长等待 5 分钟                                    │   │
│    └─────────────────────────────────────────────────┘   │
│      ↓                                                   │
│    ┌── 4. 冷却间隔 ──────────────────────────────────┐   │
│    │  等待 N 秒（默认 2s）再处理下一个文件              │   │
│    └─────────────────────────────────────────────────┘   │
│      ↓                                                   │
│  文件 N+1 ...                                            │
└──────────────────────────────────────────────────────────┘
```

### 各层详解

| 层级 | 机制 | 配置 | 说明 |
|---|---|---|---|
| 全局并发控制 | `asyncio.Semaphore` | `BATCH_UPLOAD_MAX_CONCURRENT` | 限制同时处理的批量上传请求数，超出排队 |
| 上传重试 | 指数退避 | 内置（2s→4s，最多 3 次） | 网络波动时自动重试上传 |
| 解析重试 | 轮询 + 重触发 | 内置（每 3s 轮询，最多 3 次解析） | 解析失败自动重试，不重新上传文件 |
| 文件间冷却 | `asyncio.sleep` | `BATCH_UPLOAD_COOLDOWN_SECONDS` | 每个文件处理完等待 N 秒再处理下一个 |

---

## 完整处理流程

每个文件经历的完整生命周期：

```
1. 递归搜索文件（遍历 folder_path 所有子目录）
   ↓
2. 文件名查重（同一数据集内，可选跳过）
   ↓
3. 上传文件到 RAGFlow（失败自动重试 + 指数退避）
   ↓
4. 设置 meta_fields / chunk_method / parser_config
   ↓
5. 创建 {name, url} 索引 chunk
   ↓
6. 产品库额外 cross-reference 到文件库
   ↓
7. 写入本地 MediaIndex 表
   ↓
8. 触发 RAGFlow 异步解析
   ↓
9. 轮询等待解析完成（每 3s，最长 5min，失败重试 3 次）
   ↓
10. 返回结果 → cooldown → 下一个文件
```

---

## 注意事项

1. **文件夹路径**：`folder_path` 和 `manifest_path` 都是**服务器端路径**，不是客户端路径
2. **递归搜索**：文件可以放在 `folder_path` 的任意子目录中，系统会自动找到。隐藏文件夹（`.git` 等）会被跳过
3. **同名文件**：如果多个子目录中有同名文件，使用找到的第一个。建议尽量保持文件名唯一
4. **meta 字段 null**：值为 `null` 的字段会被自动剔除，不会写入 RAGFlow
5. **权限**：端点需要 admin token
6. **重复上传**：默认开启重名检测，传 `skip_duplicate_check=true` 可强制重传
7. **大文件**：扫描件 PDF 等大文件解析可能较慢，单文件最长等待 5 分钟。超时可适当调大文件间冷却间隔
8. **冷却调优**：RAGFlow 性能充足时可设 `BATCH_UPLOAD_COOLDOWN_SECONDS=0` 加速；资源紧张时可加大该值
