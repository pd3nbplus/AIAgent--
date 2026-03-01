
WSL2 + Ubuntu 24.04 + Docker Desktop 配置双内核环境
---
@[toc]

# WSL2 + Ubuntu 24.04 + Docker Desktop 配置双内核环境
## 使用WSL自动安装

```bash
wsl --set-default-version 2
# 查看网络上有哪些 Ubuntu 版本可用
wsl --list --online
# 安装 Ubuntu 24.04
wsl --install Ubuntu-24.04
```

如果上面的方案不行，尝试手动安装。

## 手动安装
根据微软官方文档，可以通过以下方式获取Ubuntu 24.04安装包：

1. 访问微软官方下载链接：
   - [Ubuntu 24.04 LTS (x64, arm64)](https://wslstorestorage.blob.core.windows.net/wslblob/Ubuntu2404-240425.AppxBundle)

2. 在 PowerShell (管理员) 中运行：
```powershell
# 将路径替换为你实际下载的文件路径
Add-AppxPackage -Path "C:\Users\用户名\Downloads\CanonicalGroupLimited.Ubuntu24.04LTS_2404.1.26.0_neutral_~_79rhkp1fndgsc.AppxBundle"
```

安装完成后，在开始菜单搜索 "Ubuntu 24.04" 并点击运行。它会弹出一个小黑框进行最后的解压和初始化（这一步很快，不需要联网下载大文件）。
输入用户名和密码即可。

✅ 验证安装成功
```
wsl -l -v
```

输出
```text
  NAME            STATE           VERSION
* Ubuntu-24.04    Running         1
```

由于观察到 version 为1,运行以下命令将 Ubuntu-24.04 转换为 WSL 2：
```
wsl --set-version Ubuntu-24.04 2
```
注意：这个过程可能需要几分钟，因为它需要将文件系统从虚拟硬盘格式转换为 VHDX 格式。请耐心等待直到显示 "Conversion complete"。

## ⚠️ 关于内核版本过旧 (5.10.16) 的重要提示

你提到的 内核版本：5.10.16 确实非常旧（那是 2020 年的内核）。虽然 Mirror 模式主要依赖 Windows 宿主机的网络栈，但较新的内核对 Docker、GPU 直通和文件系统的性能有巨大提升。


1. 手动更新 WSL2 内核：
在 PowerShell (管理员) 中运行：
```powershell
wsl --update
```
如果这步依然报错或卡住，说明你的 Windows 更新策略限制了它。你可以尝试去微软官网下载最新的 WSL2 Linux Kernel Update Package (MSI) 手动安装。
+ 下载地址：https://learn.microsoft.com/en-us/windows/wsl/install-manual#step-4---download-the-linux-kernel-update-package
+ 下载后双击安装即可。

1. 安装完成后，

运行以下命令查看内核版本:
```bash
wsl --shutdown
# 重新进入 Ubuntu
wsl -d Ubuntu-24.04
# 检查内核版本
uname -r
```
输出
```text
6.6.87.2-microsoft-standard-WSL2
```

# 将ubuntu导出到D盘

WSL2 的发行版本质上是一个 .vhdx 虚拟磁盘文件。我们可以通过 "导出 (Export) -> 注销 (Unregister) -> 导入 (Import)" 的三步法，轻松把它"搬家"到 D 盘。

假设你想把 Ubuntu 移动到 D:\WSL\Ubuntu-24.04 目录下。

第一步：完全关闭 WSL
在操作前，必须确保 Ubuntu 已经停止运行，否则文件会被锁定。
打开 PowerShell (管理员)，执行：
```bash
wsl --shutdown
```

第二步：导出 Ubuntu 为备份文件
```bash
wsl --export Ubuntu-24.04 D:\ubuntu-backup.tar
```

第三步：注销（删除）原 C 盘的 Ubuntu

⚠️ 警告：这一步会彻底删除当前安装在 C 盘的 Ubuntu 及其所有数据。请确保第二步的 .tar 文件已经生成成功且大小正常（通常几 GB）。
```bash
# 语法：wsl --unregister <发行版名称>
wsl --unregister Ubuntu-24.04
```

执行完后，你在 wsl -l -v 中将看不到 Ubuntu-24.04 了。

第四步：导入到 D 盘
```bash
mkdir D:\WSL\Ubuntu-24.04
# 语法：wsl --import <新名称> <安装目录> <备份文件路径> --version 2
wsl --import Ubuntu-24.04 D:\WSL\Ubuntu-24.04 D:\ubuntu-backup.tar --version 2
```

第五步：恢复默认用户（重要！）
通过 --import 方式安装的 WSL，默认会以 root 用户登录，而不是你之前创建的普通用户。我们需要把它改回来。
```bash
# 启动 Ubuntu：
wsl -d Ubuntu-24.04

# 修改默认用户配置：
# # 创建/编辑 wsl.conf
nano /etc/wsl.conf
```

在文件中加入以下内容（把 myuser 换成你真实的用户名）：
```conf
[user]
default=myuser
```
按 Ctrl+O 保存，Enter 确认，Ctrl+X 退出。

重启 WSL 生效：
```bash
wsl --shutdown
wsl -d Ubuntu-24.04
```

第六步：清理临时文件
确认 D 盘的 Ubuntu 运行正常，数据都在，就可以删除那个巨大的临时备份包了：
```bash
del D:\ubuntu-backup.tar
```

最后依旧检查
```bash
wsl -l -v
```

# 安装Docker Desktop

 **推荐方案：安装 Docker Desktop**
虽然纯命令行党可能更喜欢直接装 docker.io，但在 WSL2 + GPU 这个特定场景下，Docker Desktop 是目前唯一能"开箱即用"且"省心"的方案。
为什么选 Docker Desktop？
1. GPU 直通自动化：
   + Docker Desktop：安装时勾选 "Use WSL 2 based engine"，它会自动配置好 NVIDIA Container Toolkit。你只需要在 docker-compose.yml 里写 deploy.resources...，它就能自动调用你的 4070 显卡跑 Qwen。
   + 手动安装 (docker.io)：你需要手动安装 NVIDIA Driver (WSL版)、手动安装 nvidia-container-toolkit、手动修改 /etc/docker/daemon.json、手动重启 docker 服务。一旦 WSL 内核更新，还可能失效，排查起来非常头疼。
2. WSL 集成管理：
   + Docker Desktop 有一个漂亮的 GUI，可以直观地看到哪个 WSL 发行版（Ubuntu-24.04）被激活了，资源占用多少，一键开关。
   + 它会自动处理 Windows 和 WSL2 之间的端口映射和网络桥接。
3. 上下文切换：
   + 作为全栈，你可能偶尔需要切换到其他容器环境，Docker Desktop 的 Context 管理非常方便。

什么时候才选手动安装 (docker.io)？
+ 你的电脑内存极小（<8G），无法承受 Docker Desktop 的后台进程（但你由 64G 内存，完全不用考虑这个）。
+ 你在构建极度精简的生产环境镜像，不允许任何 GUI 组件。
+ 你是 Linux 内核专家，享受手动配置每一个参数的过程。

## 备份原有mysql并停止

因为我是打算迁移到Docker Desktop，所以先备份mysql

```bash
# windows下
mysqldump -u root -p --all-databases --single-transaction --quick --lock-tables=false > D:\mysql_backup_all.sql
```

```bash
# 进入 WSL2 Ubuntu 终端
wsl -d Ubuntu-24.04

# 验证能否访问 Windows 文件
# WSL2 自动将 Windows 磁盘挂载在 /mnt/ 下。
ls -lh /mnt/d/mysql_backup_all.sql
```
如果能看到文件信息，说明通路正常。

停用 Windows MySQL
1. 按 Win + R，输入 services.msc，回车。
2. 找到 MySQL 服务：通常叫 MySQL、MySQL80 或 MariaDB（我的是mysql57）
3. 停止并禁用：
   1. 右键点击 -> 停止 (Stop)。
   2. 双击进入属性 -> 将"启动类型"改为 手动 (Manual) 或 禁用 (Disabled)。
   3. 建议选"手动"：万一哪天 Docker 挂了急需临时用原生库救急，还能手动开一下

验证端口释放：
在 PowerShell 运行：
```bash
netstat -ano | findstr :3306
```
如果没有输出，说明端口已释放，完美！(如果有，等一会再看一下)

## 🚀 执行步骤：安装 Docker Desktop
既然你已经在 D 盘安好了 Ubuntu，我们按以下步骤操作：
1. 下载与安装 (在 Windows 宿主机操作)
   1. 访问 Docker Desktop 官网。
   2. 下载 Windows 版 (它会同时支持 WSL2)。
   3. 运行安装程序。
   4. 关键配置：在安装向导中，确保勾选 "Use WSL 2 instead of Hyper-V" (默认通常就是)。
   5. 安装完成后，重启电脑（或者注销重新登录）。
2. 初始化配置 (在 Windows 操作)
   1. 启动 Docker Desktop。
   2. 首次启动会提示你同意协议。
   3. 进入 Settings (齿轮图标) -> Resources -> WSL Integration。
   4. 你会看到你的 Ubuntu-24.04 出现在列表中。
      + ✅ 开启 "Enable integration with my default WSL distro"。
      + ✅ 开启 "Ubuntu-24.04" 的开关。
   5. 点击 Apply & Restart。
3. 验证安装 (在 Ubuntu 终端操作)

打开你的 WSL2 Ubuntu 终端：
```bash
# 进入Ubuntu
wsl -d Ubuntu-24.04

# 检查 docker 命令是否可用
docker --version

# 检查是否能连接到守护进程
docker info
# 关键测试：检查 GPU 是否被识别
# 运行一个带有 nvidia-smi 的临时容器
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

如果成功：你会看到一个表格，显示你的 RTX 4070 显卡信息、显存占用等。

## docker配置

### 🚀 第一步：创建项目结构与配置文件
请在你的 WSL2 Ubuntu 终端 中执行以下命令：
```bash
# 1. 创建项目根目录
mkdir -p ~/ai-stack && cd ~/ai-stack

# 2. 创建数据挂载目录 (防止权限问题)
mkdir -p data/redis data/mysql models

# 3. 创建 docker-compose.yml 文件
nano docker-compose.yml
```

### 📝 第二步：填入配置内容
将以下内容完整复制并粘贴到 nano 编辑器中：
```yaml
services:
  redis:
    image: redis:8.0
    container_name: redis-local
    ports:
      - "6379:6379"
    volumes:
      - ./data/redis:/data
    command: redis-server --appendonly yes
    restart: unless-stopped
    networks:
      - ai-net

  mysql:
    image: mysql:8.0
    container_name: mysql-local
    environment:
      MYSQL_ROOT_PASSWORD: xxxxxxx # 你的mysql密码
      TZ: Asia/Shanghai
    ports:
      - "3306:3306"
    volumes:
      - ./data/mysql:/var/lib/mysql
    command: --character-set-server=utf8mb4 --collation-server=utf8mb4_unicode_ci
    restart: unless-stopped
    networks:
      - ai-net

  qwen:
    image: vllm/vllm-openai:latest
    container_name: qwen-local
    ports:
      - "7575:8000"
    volumes:
      - /mnt/d/Models/Qwen3-4B:/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    command: >
      /models
      --dtype half
      --gpu-memory-utilization 0.85
      --max-model-len 8192
      --served-model-name qwen-3-4b
    restart: unless-stopped
    networks:
      - ai-net
    environment:
      - HF_ENDPOINT=https://hf-mirror.com

networks:
  ai-net:
    driver: bridge
```

💡 配置亮点解析：
+ Redis: 开启了 --appendonly yes (AOF)，保证数据不丢失。
+ MySQL: 设置了 utf8mb4 字符集，避免中文乱码。增加了时区设置。
+ Qwen:
   + --gpu-memory-utilization 0.85: 预留 15% 显存给系统和其他任务，防止 OOM。
   + --served-model-name: 给模型起个别名，方便代码里调用。
   + networks: 所有容器加入同一个 ai-net 网络。以后你的 Agent 代码如果在另一个容器里，可以直接用 redis://redis:6379 连接，无需知道 IP。

### 🛠️ 第三步：启动服务
保存文件 (Ctrl+O, Enter) 并退出 (Ctrl+X)，然后执行：

```bash
# 1. 拉取镜像并启动所有服务 (-d 表示后台运行)
docker compose up -d

# 2. 实时查看日志 (观察 Qwen 加载模型的过程，这可能需要几分钟)
docker compose logs -f qwen

# 3. 将备份的sql数据恢复到mysql中 prootpassword 为你的mysql密码 直接管道导入（推荐） -f 强制导入 忽略权限错误
cat /mnt/d/mysql_backup_all.sql | docker exec -i mysql-local mysql -uroot -prootpassword -f

# ✅ 验证导入结果
# 进入容器内部：
docker exec -it mysql-local mysql -uroot -pxxx
show databases;
USE blog_project;
show tables;
EXIT;
```

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/61b5f4d828db4e2f9f9b1a025c7bb6bc.png#pic_center)



#### 网络设置

如果出现了网络连接超时或中断问题
```text
Error response from daemon: failed to resolve reference "docker.io/library/mysql:8.0": failed to do request: Head "https://registry-1.docker.io/v2/library/mysql/manifests/8.0": EOF
```

可以配置 Docker Desktop 使用proxy（最推荐，图形化）
1. 打开 Docker Desktop 界面。
2. 点击右上角的 齿轮图标 (Settings)。
3. 选择左侧的 Resources -> Proxies。
4. 选择 Manual proxy configuration。
5. 填写你的proxy信息：
   + HTTP Proxy: http://host.docker.internal:7897
     > 注意：在 Docker Desktop 设置里，通常用 host.docker.internal 代表宿主机 Windows。如果不行，试一下 http://192.168.x.x:7897 (你的 Windows 局域网 IP)。
   + HTTPS Proxy: http://host.docker.internal:7897
   + No Proxy: localhost,127.0.0.1,.local,.internal
6. 点击 Apply & Restart。

重启完成后，再次尝试运行 docker compose up -d。

+ 注意：记得在设置里开启"局域网连接"，并且检查proxy节点

#### 网络波动导致失败

下载大模型（尤其是像 Qwen 这种几十 GB 的模型）时，网络波动导致失败是非常令人崩溃的。因为 vLLM 或 HuggingFace 默认是流式下载，一旦中断，往往需要从头开始或者很难断点续传。

🏆 方案一：在 Windows 下用"下载器"下载，再映射进 WSL2（最稳、最快）

使用 Git LFS 或 huggingface-cli 下载 (推荐)：
在 Windows PowerShell 中运行（记得开启"允许局域网"且配置了环境变量）：
```bash
# 设置proxy (临时生效)
$env:HF_ENDPOINT="https://hf-mirror.com" # 使用国内镜像源加速
$env:HTTP_PROXY="http://127.0.0.1:7897"
$env:HTTPS_PROXY="http://127.0.0.1:7897"

# 安装 huggingface hub (如果没装)
pip install huggingface_hub

# 下载模型到 Windows 本地目录 (例如 D:\Models\Qwen3-4B)
huggingface-cli download --resume-download Qwen/Qwen3-4B --local-dir "D:\Models\Qwen3-4B"
```
`--resume-download`：关键参数！支持断点续传。如果断了，重新运行命令它会接着下。

再不行 就直接去huggingface.co下载 https://huggingface.co/Qwen/Qwen3-4B/tree/main?show_file_info=model-00002-of-00003.safetensors

映射到 WSL2：
   + 下载完成后，文件在 D:\Models\Qwen3-4B。
   + 在 WSL2 中，这个路径自动挂载为 /mnt/d/Models/Qwen3-4B。

修改 Docker Compose：
   + 编辑 ~/ai-stack/docker-compose.yml，将卷挂载路径指向这里：

```yaml
volumes:
  # 原来：- ./models:/models
  # 现在：直接挂载 Windows 下载好的目录 (只读模式 ro 可选，防止误删)
  - /mnt/d/Models/Qwen3-4B:/models:ro
```

启动服务：
```bash
docker compose up -d
```

## 现代云原生开发的核心理念

问题:这是专门为我的ai agent开发准备的环境吗？ 还是说这是统一配置的环境，我希望mysql和redis能作为独立服务进行部署
+ 简短的回答是：这是一个"统一配置的开发环境"，但其中的 MySQL 和 Redis 确实是作为"独立服务"运行的。

这种架构既不是"为了 AI 硬凑的临时方案"，也不是"把所有东西塞进一个黑盒"。它是目前业界（包括阿里、字节、Google 等）最标准的**微服务化开发模式**。

让我为你详细拆解一下这个设计的精妙之处：
1. 核心概念：容器化 = 独立服务
在 `docker-compose.yml` 中，`redis、mysql、qwen` 分别定义在三个独立的 services 块里。
   + 独立性：它们运行在三个完全隔离的容器中。
      + Redis 挂了，MySQL 和 Qwen 不受影响。
      + 你可以单独重启 MySQL (docker restart mysql-local)，而不需要动 Qwen。
      + 它们有独立的进程、独立的文件系统、独立的内存空间。
   + 服务化：它们通过网络端口进行通信，就像部署在三台不同的服务器上一样。
      + 你的 Python 代码连接 Redis 是通过 localhost:6379 (网络请求)。
      + 你的 Python 代码连接 MySQL 是通过 localhost:3306 (网络请求)。
      + 你的 Python 代码调用 Qwen 是通过 localhost:7575 (HTTP API 请求)。

**结论：是的，它们就是独立的标准化服务。你并没有把它们"写死"在 AI 代码里，而是让它们作为基础设施存在。**
2. 为什么要把它们放在一起 (docker-compose)？
既然是独立服务，为什么不分开启动（比如手动 docker run 三次）？
   + 一键编排 (Orchestration)：  `docker-compose up -d` 的作用是同时启动这一组相互依赖的服务，并自动构建它们之间的网络。
      + 如果你以后要把这个项目发给同事，他只需要拿到这个 docker-compose.yml 文件，运行一行命令，就能获得和你一模一样的环境（相同的版本、相同的配置、相同的端口）。
      + 这就是"环境即代码" (Infrastructure as Code)。
   + 资源管理：
      在这个文件中，我们可以统一规划资源。比如限制 Qwen 只能用 90% 显存，防止它把显卡占满导致其他任务无法运行。

3. 这是专门为 AI Agent 准备的吗？
**既是，也不是。**
   + "不是"的一面（通用性）：
      + Redis：这是通用的缓存和消息队列。无论你做 Web 后端、游戏服务器、还是高频交易，都会用到它。这里的配置（开启 AOF 持久化）是生产级标准配置。
      + MySQL：这是通用的关系型数据库。存储用户信息、订单、日志，什么都可以。这里的配置也是标准的。
      + 即使你不做 AI，这个 docker-compose.yml 去掉 Qwen 部分，就是一个完美的全栈 Web 后端基础环境。
   + "是"的一面（针对性）：
      + Qwen (vLLM)：这部分确实是专门为 AI Agent 准备的。它加载了大模型，提供了 LLM 能力。
      + 组合效应：AI Agent 的核心公式是 LLM + Memory (Redis/MySQL) + Tools。
         + Redis 在这里通常充当 短期记忆 (Short-term Memory) 或 消息总线（Agent 内部模块通信）。
         + MySQL 在这里通常充当 长期记忆 (Long-term Memory) 或 知识库元数据（存储向量索引、用户画像、历史对话记录）。
         + Qwen 是 大脑。
      + 这个配置文件将"大脑"和"记忆系统"预组装在了一起，让你能立刻开始编写 Agent 的业务逻辑，而不用花几天时间去调试数据库连接和模型部署。

4. 未来的扩展性（解耦）
如果你将来业务做大了，这个架构可以轻松拆分：
   1. 场景 A：数据库上云
      + 你觉得本地 MySQL 性能不够了，买了阿里云 RDS。
      + 操作：只需修改 docker-compose.yml，删掉 mysql 服务，然后把你的 Python 代码里的数据库连接地址改成阿里云的 IP。
      + 结果：Redis 和 Qwen 依然在本地跑，数据库在云端。完全解耦。
   2. 场景 B：多 Agent 集群
      + 你需要跑 10 个不同的 Agent。
      + 操作：保留 Redis 和 MySQL 作为共享服务。复制 qwen 服务块，启动 qwen-agent-1, qwen-agent-2... 或者把 Qwen 部署到专门的 GPU 服务器上。
      + 结果：基础设施（DB/Cache）与计算节点（Agent）分离。
   3. 场景 C：纯后端项目
      + 你突然想做一个不带 AI 的电商网站。
      + 操作：直接复用这个 docker-compose.yml，注释掉 qwen 部分。
      + 结果：你瞬间拥有了一个高可用的电商后端基础环境。

📝 总结
这个配置方案是"**以 AI Agent 开发为场景切入点**"构建的"**标准化云原生开发环境**"。
   + MySQL 和 Redis：是独立、通用、标准的基础设施服务。它们不依赖 AI 存在，AI 也不依赖它们的特定实现（只要协议对就行）。
   + Docker Compose：只是一个启动器，方便你一键拉起这套"全家桶"。
   + 你的收益：你不需要分别去安装 MySQL 安装包、配置 Redis conf 文件、折腾 vLLM 环境。你拥有了一套可移植、可复现、易维护的开发底座。

### 复用docker-compose.yml，还会再拉取一遍mysql镜像再启动一个实例吗？

答案是：不会重新下载镜像，也不会产生冲突，但会启动一个新的容器实例（除非你复用旧的数据卷）。

1. 关于"再拉取一遍镜像"
**结论：不会重新下载。**
   + 机制：Docker 镜像是全局共享的。
      + 当你第一次运行 docker compose up -d (AI 场景) 时，Docker 已经下载了 mysql:8.0 和 redis:8.0 镜像存在本地。
      + 当你做"场景 C"时，修改了 docker-compose.yml（注释掉 qwen），再次运行 docker compose up -d。
      + Docker 会检查本地是否有 mysql:8.0。发现有！它会直接复用本地镜像，跳过下载步骤，秒级启动。
   + 例外：除非你手动执行了 docker rmi mysql:8.0 删除了镜像，或者过了很久镜像版本更新了且你配置了自动拉取，否则绝不会重复下载。

<br>

2. 关于"再启动一个实例"
**结论：这取决于你如何操作目录和数据卷。**
这里有两种情况：
+ **情况 A：你在同一个目录 (~/ai-stack) 操作（推荐）**
   1. 如果你只是修改了当前的 docker-compose.yml，把 qwen 部分注释掉，然后运行 `docker compose up -d`：
   2. Qwen 容器：Docker 发现配置文件里没了 qwen 服务，它会停止并移除 qwen-local 容器（释放显存）。
   3. MySQL/Redis 容器：Docker 发现配置里还有它们，且镜像没变。
      + 如果容器已经在运行：它检测到配置无变化，什么都不做（保持运行，数据保留）。
      + 如果容器停止了：它会重启原有的容器。
      + 关键点：因为它复用了之前的 ./data/mysql 挂载卷，你的数据（用户表、订单表）都在！ 你不需要重新初始化数据库。
   4. 结果：你得到了一个干净的、去除了 AI 功能的、数据完整的电商后端环境。这是最丝滑的体验。
+ **情况 B：你复制了一份文件到新目录 (~/ecommerce-project)**
   1. 如果你把 docker-compose.yml 复制到了新文件夹，并注释掉 qwen，然后运行：
   2. 镜像：依然不会重新下载，复用本地缓存。
   3. 容器：Docker 会创建全新的 mysql 和 redis 容器（因为项目目录变了，默认的卷路径也变了，或者名字冲突处理机制不同）。
   4. 数据：新容器的数据库是空的（因为没有挂载到旧的 ./data/mysql）。
   5. 结果：你得到了一个全新的、空白的电商环境。你需要重新导入数据或运行迁移脚本。

#### 💡 最佳实践建议

为了实现你描述的"瞬间拥有高可用电商后端"，我建议采用 情况 A 的变体，或者使用 Profile 功能（Docker Compose 的高级特性）。

方法一：使用 Docker Compose Profiles（最优雅）
你不需要注释代码！我们可以给服务打标签。
修改你的 docker-compose.yml：
```yaml
services:
  redis:
    # ... 配置不变 ...
    profiles: ["common"] # 标记为通用服务

  mysql:
    # ... 配置不变 ...
    profiles: ["common"] # 标记为通用服务

  qwen:
    # ... 配置不变 ...
    profiles: ["ai"] # 标记为 AI 专属服务
```

使用方式：
开发 AI Agent 时：
```bash
docker compose --profile common --profile ai up -d
# 或者默认启动所有（如果不指定 profile）
docker compose up -d 
```

开发纯后端电商时：
```bash
docker compose --profile common up -d
# 这样 Qwen 根本不会启动，节省显存，但 MySQL/Redis 照常运行且数据保留
```

# 其他操作

## 停止容器

```bash
cd ~/ai-stack
docker compose down
```

## 清理mysql数据集

```bash
# 删除之前挂载的 MySQL 数据目录
sudo rm -rf ./data
# 重新创建空目录
mkdir -p ./data/redis ./data/mysql ./models
# 再次确认 mysql 目录是空的
ls -la ./data/mysql
```

## 修改端口映射

```yaml
nano docker-compose.yml
# 以修改mysql端口为例
- "3306:3306"
# 修改为 (宿主机 3307 -> 容器 3306)
- "3307:3306"
```

## 全面测试

🚀 第一步：检查容器状态与日志
```bash
docker ps
nano ~/ai-stack/docker-compose.yml
docker-compose up -d
docker-compose down
docker logs -f qwen-local
```

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b613802f3d584995967db4e144b5fd15.png#pic_center)


+ mysql测试

连上mysql
```sql
-- 用vs code插件连也行 
-- 进入mysql容器也行
-- docker exec -it mysql-local mysql -uroot -pxxx
show databases;
```

+ redis测试

连上redis
```bash
ping
```

+ qwen测试

```bash
# 查看启动日志
docker logs -f qwen-local
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c1a57820b51441649086f89a50725c62.png#pic_center)



直接在浏览器访问 http://localhost:7575/docs

qwen启动日志报错常见问题：
1. max-model-len 设置的太长了 模拟的时候显存预估不够
2. command 参数在新版 vLLM 推荐直接把模型路径作为第一个参数，而不是用 --model。

## 内存占用问题

现象解释：
你看到的"高内存占用"其实是 Docker + WSL2 + GPU 模型加载机制 的正常表现，但这其中可能包含了一些可以优化的浪费。
我们需要区分两种情况：
1. 显存 (VRAM) 占用高：这是正常且必须的。
2. 系统内存 (RAM/WSL2 内存) 占用高：这可能是配置不当导致的，可以优化。

如果你是指 Windows 的任务管理器中 "内存" (64GB RAM) 占用很高，或者 WSL2 进程 (vmmem 或 vmmemWSL) 占用了大量内存（比如 10GB+），即使服务空闲：
原因分析：
1. WSL2 的"只增不减"机制：WSL2 默认会动态使用 Windows 内存，但不会自动释放。一旦 Docker 或模型加载用掉了 10GB，即使你停止了容器，WSL2 进程往往还会抱着这 10GB 不放，直到你重启 WSL。
2. Docker 桌面后台进程：Docker Desktop 本身及其辅助进程也会占用几百 MB 到 1GB 的内存。
3. 模型权重加载策略：虽然 vLLM 主要把模型放在显存里，但在初始化阶段或部分算子计算时，可能会在系统内存中保留一份副本或临时缓冲区。

重启wsl
```bash
wsl --shutdown
wsl -d Ubuntu-24.04
cd ~/ai-stack
docker-compose up -d
```

设置开机自启动： 只自启动redis和mysql

🚀 方案：使用 Windows 任务计划程序
第一步：创建启动脚本
1. 在 Windows 上创建一个新文件，比如 D:\Scripts\start-dev-env.bat (如果没有 Scripts 文件夹就新建一个)。
2. 右键点击该文件 -> 编辑 (或用记事本打开)。
3. 粘贴以下代码：
```batch
@echo off
REM 等待网络完全就绪 (可选，防止网络驱动没加载完)
timeout /t 5 /nobreak >nul

REM 1. 启动 WSL2 (默认发行版)
REM 这步会唤醒 WSL，但不会自动启动任何 Docker 容器
wsl --exec echo "WSL2 Started"

REM 2. 进入你的项目目录并启动指定服务
REM 注意：这里我们显式指定只启动 redis 和 mysql，不启动 qwen
wsl -d Ubuntu-24.04 -e bash -c "cd ~/ai-stack && docker compose up -d redis mysql"

REM 结束
exit
```

4. 保存并关闭文件。
> 💡 原理：
> + wsl --exec ...：先轻轻唤醒 WSL 内核。
> + wsl -d Ubuntu-24.04 -e bash -c "..."：以非交互模式执行命令。
> + docker compose up -d redis mysql：关键点。Docker Compose 允许你指定只启动某些服务。因为 qwen 不在列表中，它绝不会启动，显存完全空闲。

第二步：设置开机自启 (任务计划程序)
1. 打开任务计划程序：
   + 按 Win + S，搜索 "任务计划程序" (Task Scheduler)，打开它。
2. 创建基本任务：
   + 点击右侧的 "创建基本任务..."。
   + 名称：输入 Start Dev Services (或者你喜欢的名字)。
   + 描述：可选，比如"开机自动启动 Redis 和 MySQL"。
   + 点击 下一步。
3. 设置触发器：
   + 选择 "计算机启动时" (When the computer starts)。
   + 点击 下一步。
4. 设置操作：
   + 选择 "启动程序" (Start a program)。
   + 点击 下一步。
   + 程序或脚本：浏览并选择你刚才创建的 D:\Scripts\start-dev-env.bat。
   + 起始于 (可选)：填写脚本所在的文件夹路径，例如 D:\Scripts (这很重要，防止相对路径错误)。
   + 点击 下一步。
5. 完成配置：
   + 勾选 "当单击'完成'时，打开此任务属性的对话框"。
   + 点击 完成。

第三步：关键权限设置 (必须做！)
为了让它在后台静默运行且不弹窗，需要修改刚才打开的属性窗口：
1. 常规 (General) 选项卡：
    + ✅ 勾选 "不管用户是否登录都要运行" (Run whether user is logged on or not)。
    + 注意：选了这项后，脚本会在后台运行，你看不到黑框框，这是正常的。
    + ✅ 勾选 "使用最高权限运行" (Run with highest privileges)。
    + 配置 (Configure for)：选择你的 Windows 版本 (如 Windows 10/11)。
2. 条件 (Conditions) 选项卡：
    + ❌ 取消勾选 "只有在计算机使用交流电源时才启动此任务" (如果你是用笔记本，且希望拔电源也能启动，就取消；如果是台式机，可以保留)。
    + ❌ 取消勾选 "只有在以下网络连接可用时才启动" (防止因网络判断延迟导致启动失败)。
3. 设置 (Settings) 选项卡：
    + ✅ 勾选 "如果任务运行时间超过... 停止任务" 可以取消勾选，或者设长一点 (虽然这个脚本很快就跑完了)。
4. 点击 确定。
    + 系统会提示你输入 Windows 登录密码 以保存凭据。输入后确认。


第四步：测试与验证
不要重启电脑，先手动测试一下：
1. 停止所有服务：
```bash
wsl -d Ubuntu-24.04 -e bash -c "cd ~/ai-stack && docker compose down"
```
2. 在任务计划程序中测试：
   + 找到刚才创建的 Start Dev Services 任务。
   + 右键点击 -> 运行 (Run)。
3. 验证结果：
   + 打开 WSL2 终端，运行 docker ps。
   + 你应该看到 redis-local 和 mysql-local 是 Up 状态。
   + Qwen (qwen-local) 应该不在列表中。
   + 打开任务管理器，确认显存 (GPU Memory) 没有被占用，但 Vmmem 内存有少量增加 (Redis/MySQL 的开销)。

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ac0b39be35ab4390b31325a2eb0480d1.png)

以后想启动 Qwen 怎么办？
```bash
wsl -d Ubuntu-24.04
cd ~/ai-stack
docker-compose up -d qwen
```

🚀 方案B：修改配置文件

1. 先设置docker desktop开机自启动
2. 然后修改qwen的配置

```bash
nano ~/ai-stack/docker-compose.yml
qwen:
    image: vllm/vllm-openai:latest
    container_name: qwen-local
    # ... 其他配置 ...
    restart: "no"  # <--- 修改这里！禁止自动重启
```


## 修改终端样式

🏆 方案一：直接使用 Windows Terminal (最推荐，微软官方出品)
这是目前 Windows 上最好的终端工具，它原生支持 WSL2。当你打开它时，你可以直接选择 "Ubuntu-24.04" 标签页，本质上你就是在运行 Ubuntu 的 Bash，所以它的样式、颜色、提示符和你之前在 WSL 里看到的一模一样！
1. 安装
   + 打开 Microsoft Store (微软应用商店)。
   + 搜索 "Windows Terminal" (或者直接搜 "Terminal")。
   + 点击 获取/安装。
     + (注：Windows 11 默认已预装，直接在开始菜单搜 "Terminal" 或 "wt" 即可)
2. 配置为默认终端 (关键步骤)
    为了让它接管所有的命令行操作：
    1. 打开 Windows Terminal。
    2. 点击标题栏的下拉箭头 -> 设置 (Settings) (或按 Ctrl + ,)。
    3. 在左侧选择 "启动" (Startup)。
    4. 默认配置文件 (Default profile)：选择 Ubuntu-24.04。
    5. 默认终端程序选择刚才下载的那个windows终端
    6. (可选) 勾选 "以管理员身份运行" (如果你经常需要 sudo)。
    7. 点击 保存。