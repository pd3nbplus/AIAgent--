明白了！你当前的系统架构是：

环境配置：当前环境是基于WSL2 + Ubuntu 24.04 + Docker Desktop构建的云原生开发平台，所有服务（MySQL、Redis、Qwen）均以独立容器形式运行并通过Docker Compose统一编排。如何配置请参考我的博客 [WSL2 + Ubuntu 24.04 + Docker Desktop 配置双内核环境](https://blog.csdn.net/weixin_52185313/article/details/158416250?spm=1011.2415.3001.5331)

- **目标**：在 WSL2 中配置 **Codex CLI 运行环境**
- **约束**：
  - 暂不修改 `~/.codex/config.toml`
  - 不使用本地 Qwen 模型（后续会用 OpenAI 或兼容 API）
  - 当前 Node.js 是 v18.19.1，但你需要 **Node.js v22**

---

## ✅ 第一步：彻底升级 Node.js 到 v22（推荐使用 `nvm`）

你当前用的是系统包管理器安装的 Node.js（v18），版本老旧且难以灵活切换。**最佳实践是使用 `nvm`（Node Version Manager）** 来管理多版本 Node。

### 1. 卸载旧版 Node.js（可选但推荐）

```bash
# 移除通过 apt 安装的 nodejs 和 npm
sudo apt remove --purge -y nodejs npm
```

> ⚠️ 注意：不要手动删 `/usr/bin/node`，用包管理器卸载更干净。

### 2. 安装 nvm（Node Version Manager）

```bash
# 下载并运行 nvm 安装脚本
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
```

### 3. 重载 shell 配置

```bash
# 如果你用的是 zsh（Ubuntu 24.04 默认）
source ~/.zshrc

# 如果你用的是 bash
source ~/.bashrc
```

验证 nvm 是否安装成功：

```bash
nvm --version
# 应输出 0.39.7 或类似
```

### 4. 安装并使用 Node.js v22

```bash
# 安装 LTS 版本（当前 v22 是 Active LTS）
nvm install 22

# 设置为默认版本
nvm use 22
nvm alias default 22
```

### 5. 验证

```bash
node -v  # 应输出 v22.x.x
npm -v   # 应 >= 10.x
```

> ✅ 此时你的 Node 环境已符合 Codex 官方推荐（Node 20+，v22 更佳）。

---

## ✅ 第二步：安装 Codex CLI（全局可用）

现在 Node.js 已就绪，安装 Codex：

```bash
npm i -g @openai/codex
```

> 💡 如果遇到权限问题（EACCES），**不要用 sudo**！因为 nvm 安装的 Node 不需要 root。如果报错，检查 `which node` 是否指向 `~/.nvm/...`。

验证安装：

```bash
codex --version
# 应输出类似 0.x.x
```

---

## ✅ 第三步：初始化 Codex（生成默认配置目录）

首次运行会创建 `~/.codex` 目录（但不会写入模型配置，满足你“暂不修改配置文件”的要求）：

```bash
codex
```

> 首次启动可能会提示登录 OpenAI 账号（如果你打算用官方 GPT-4o/Codex）。  
> **你可以直接 Ctrl+C 退出**，因为我们还没配置 API。

此时目录结构如下：e

```bash
~/.codex/
├── config.toml      # （空或默认，我们暂不编辑）
└── ...              # 其他运行时文件
```

---

## ✅ 第四步：准备后续模型接入（OpenAI / 兼容 API）

虽然你现在不配置模型，但可以提前做好准备：

### 方案 A：使用 OpenAI 官方（需 Plus 订阅）
- 获取 API Key：https://platform.openai.com/api-keys
- 后续在 `~/.codex/config.toml` 中配置 `model = "gpt-4o"` 和 `api_key`

### 方案 B：使用兼容 OpenAI API 的国产模型（免费/低成本）
例如：
- **DeepSeek**（deepseek-coder）
- **Moonshot**（moonshot-v1-8k）
- **OpenRouter**（聚合多个模型）

> 📌 这些都只需提供 `base_url` 和 `api_key`，格式与 OpenAI 一致。

---

## ✅ 最佳实践总结（针对你的架构）

| 项目 | 推荐做法 |
|------|--------|
| **Node.js 管理** | 使用 `nvm`，避免系统包冲突 |
| **Node 版本** | **v22 LTS**（Codex + MCP 工具链兼容性最佳） |
| **Codex 安装位置** | WSL2 Ubuntu 全局（`npm install -g codex`） |
| **模型选择** | 暂不绑定本地 Qwen；后续用 OpenAI 或兼容 API |
| **配置文件** | 暂不编辑 `config.toml`，保持干净 |
| **MCP 工具** | 后续通过 **MCP Router** 统一管理（避免直接写 `[mcp_servers]`） |

---

## 🔜 下一步建议

当你准备好 API 密钥后，可以：

1. 创建 `~/.codex/config.toml`（参考你提供的 CSDN 教程）
2. 设置 `OPENAI_API_KEY` 环境变量（**不要硬编码到文件中**）
3. 运行 `codex` 并测试 `/init` 生成 `AGENTS.md`

需要我帮你写一个安全的 `config.toml` 模板（带环境变量引用）吗？或者部署 MCP Router？随时告诉我！