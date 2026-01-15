# TensorRT-LLM 部署问题报告

## 1. 项目背景

我们正在将 **MOSS-Speech**（复旦大学的端到端语音交互模型）部署到生产环境，目标是实现：
- **TTFA（首字延迟）**: < 300ms
- **RTF（实时率）**: < 1（实时语音）

当前使用 PyTorch FP16 推理：
- TTFA: 335ms ✅ 接近目标
- RTF: 4.25 ❌ 无法实时

根据 NVIDIA 官方文档和社区经验，TensorRT-LLM 可以将 RTF 降低 3-5 倍，因此我们尝试迁移到 TensorRT-LLM。

---

## 2. 当前环境

| 组件 | 版本 |
|------|------|
| **平台** | RunPod (云 GPU) |
| **GPU** | NVIDIA RTX 4090 (24GB VRAM) |
| **CUDA** | 12.4 |
| **Driver** | 570.195.03 |
| **Python** | 3.11 (系统) / 3.10 (虚拟环境) |
| **PyTorch** | 2.9.1+cu128 (系统) / 2.4.0+cu121 (虚拟环境) |
| **OS** | Ubuntu 22.04 (Docker 容器内) |

**重要限制**：
- RunPod 的 Pod 本身运行在 Docker 容器中
- 没有 `privileged` 权限，无法运行 Docker-in-Docker
- 本地磁盘配额约 50GB（`/` 分区）

---

## 3. 尝试的安装方案

### 方案 A: pip 安装 TensorRT-LLM 1.1.0

```bash
pip install tensorrt-llm --extra-index-url https://pypi.nvidia.com
```

**结果**: ❌ 失败

**错误**:
```
ImportError: libcublasLt.so.13: cannot open shared object file: No such file or directory
```

**原因分析**:
- TensorRT-LLM 1.1.0 的 wheel 是为 **CUDA 13** 编译的
- 我们的环境是 **CUDA 12.4**
- PyPI 上的 `nvidia-cublas-cu13` 构建失败

### 方案 B: Docker 官方镜像

```bash
docker pull nvcr.io/nvidia/tensorrt-llm:0.16.0
```

**结果**: ❌ 失败

**错误**:
```
failed to mount overlay: operation not permitted
iptables failed: Permission denied (you must be root)
```

**原因**: RunPod 容器没有 `privileged` 模式，无法运行 Docker-in-Docker

### 方案 C: 安装旧版本 (0.15.0 / 0.16.0)

**结果**: ❌ 未尝试

**原因**: 根据研究，这些版本需要 CUDA 12.6+，我们只有 12.4

---

## 4. 版本兼容性矩阵（已研究）

| TensorRT-LLM | CUDA 要求 | TensorRT | Python | 状态 |
|--------------|-----------|----------|--------|------|
| 1.1.0 | CUDA 13+ | 10.13 | 3.10 | ❌ 不兼容 |
| 0.21.0 | CUDA 12.6+ | 10.x | 3.10 | ❌ 不兼容 |
| 0.16.0 | CUDA 12.6.3 | 10.7 | 3.10 | ❌ 不兼容 |
| 0.15.0 | CUDA 12.4+ | 10.x | 3.10 | ⚠️ 可能兼容 |

---

## 5. 核心问题

### 问题 1: CUDA 版本不匹配
- PyPI 上的 TensorRT-LLM wheel 是为 CUDA 13 编译的
- 我们的环境是 CUDA 12.4
- 是否有为 CUDA 12.4 编译的 wheel？

### 问题 2: Docker 方案不可用
- RunPod 不提供 privileged 容器
- 无法使用官方 Docker 镜像
- 是否有其他方式绕过？

### 问题 3: 从源码编译的可行性
- 是否可以从源码编译 TensorRT-LLM 以支持 CUDA 12.4？
- 编译需要什么依赖？
- 预计编译时间？

---

## 6. 需要专家解答的问题

1. **CUDA 12.4 环境下，推荐安装哪个版本的 TensorRT-LLM？**

2. **是否有为 CUDA 12 编译的预构建 wheel？** 如果有，在哪里获取？

3. **如果必须从源码编译，流程是什么？**

4. **替代方案**：如果 TensorRT-LLM 确实不支持 CUDA 12.4，是否可以：
   - 单独使用标准 TensorRT 优化 MOSS-Speech 的 Flow Matching 解码器？
   - 使用其他推理加速方案（如 vLLM、SGLang）？

5. **升级 CUDA 的影响**：如果升级 RunPod 到 CUDA 12.6+，会影响现有的其他模型吗？

---

## 7. 附录：错误日志

### TensorRT-LLM 导入错误

```python
>>> import tensorrt_llm
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File ".../tensorrt_llm/__init__.py", line 70, in <module>
    import tensorrt_llm._torch.models as torch_models
  ...
  File ".../tensorrt_llm/disaggregated_params.py", line 11, in <module>
    from tensorrt_llm.bindings import executor as tllme
ImportError: libcublasLt.so.13: cannot open shared object file: No such file or directory
```

### Docker daemon 启动错误

```
failed to mount overlay: operation not permitted
failed to start daemon: Error initializing network controller: 
  iptables failed: iptables --wait -t nat -N DOCKER: 
  iptables v1.8.7 (nf_tables): Could not fetch rule set generation id: 
  Permission denied (you must be root)
```

---

## 8. 联系信息

如有进一步问题，请联系项目负责人。

---

*报告生成时间: 2026-01-14*



