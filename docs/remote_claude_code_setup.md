# 远程 GPU 服务器配置 Claude Code 完整指南

## 背景

本地机器能上外网，GPU 服务器（内网）不能直接访问 Anthropic API。通过 SSH 反向隧道 + nginx 反向代理，让远程 Claude Code 经由本地机器中转访问 API。

## 环境信息

| 角色 | IP | 系统 | 说明 |
|------|-----|------|------|
| 本地机器 | 10.169.12.153 | Ubuntu (Precision-3660) | 能访问外网和内网 API |
| GPU 服务器 | 172.16.0.110 | Ubuntu 22.04 (node110) | 不能访问外网，有 GPU |

API 端点：`http://model.mify.ai.srv/anthropic`（内网地址 10.16.73.7，仅本地可达）

## 架构图

```
远程 GPU 服务器 (172.16.0.110)                     本地机器 (10.169.12.153)
┌──────────────────────────────┐                   ┌─────────────────────────┐
│                              │                   │                         │
│  Claude Code                 │                   │                         │
│      │                       │                   │                         │
│      ▼                       │                   │                         │
│  nginx (:9081)               │    SSH 隧道       │                         │
│  添加 Host header            │ ◄════════════════►│                         │
│      │                       │  -R 9080:         │                         │
│      ▼                       │  model.mify.ai.   │    model.mify.ai.srv   │
│  SSH 隧道端口 (:9080)         │  srv:80           │ ──►  :80 (API)         │
│                              │                   │                         │
│  tinyproxy 隧道 (:8888)      │  -R 8888:         │    tinyproxy (:8888)   │
│  (npm 等通用网络)             │  127.0.0.1:8888   │ ──►  外网              │
└──────────────────────────────┘                   └─────────────────────────┘
```

## 配置步骤

### Step 1：本地安装 tinyproxy（通用代理，用于 npm 等）

```bash
sudo apt update && sudo apt install -y tinyproxy

# 确认配置：只监听 localhost，端口 8888
grep -E "^(Port|Listen|Allow)" /etc/tinyproxy/tinyproxy.conf
# 应该看到:
#   Port 8888
#   Allow 127.0.0.1
#   Allow ::1

sudo systemctl restart tinyproxy
sudo systemctl enable tinyproxy

# 验证
curl -x http://localhost:8888 https://api.anthropic.com/
# 返回 404 即正常（说明代理通了，只是没给认证参数）
```

### Step 2：SSH 免密登录

```bash
# 本地生成 key（如果没有）
ssh-keygen -t ed25519

# 复制到远程
ssh-copy-id root@172.16.0.110

# 验证
ssh root@172.16.0.110 "echo OK"
```

### Step 3：远程安装 Node.js + Claude Code

```bash
# 通过 SSH 反向隧道连接，让远程能上网
ssh -R 8888:127.0.0.1:8888 root@172.16.0.110

# 在远程执行（通过代理安装）
export http_proxy=http://127.0.0.1:8888
export https_proxy=http://127.0.0.1:8888

# 安装 Node.js 20
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
apt install -y nodejs

# 安装 Claude Code
npm install -g @anthropic-ai/claude-code

# 验证
node --version    # v20.x
claude --version  # 2.x
```

### Step 4：远程安装 nginx（解决 Host header 问题）

**为什么需要 nginx？**
SSH 隧道把远程 `127.0.0.1:9080` 映射到本地 `model.mify.ai.srv:80`，但 HTTP 请求的 Host header 是 `127.0.0.1`，API 网关（openresty）会因为 Host 不匹配返回 403。nginx 负责把 Host 改写为正确的 `model.mify.ai.srv`。

```bash
# 在远程安装
apt install -y nginx

# 创建配置
cat > /etc/nginx/sites-available/claude-api << 'EOF'
server {
    listen 9081;
    location / {
        proxy_pass http://127.0.0.1:9080;
        proxy_set_header Host model.mify.ai.srv;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 300s;
        proxy_connect_timeout 10s;
        proxy_buffering off;
    }
}
EOF

# 启用并重启
ln -sf /etc/nginx/sites-available/claude-api /etc/nginx/sites-enabled/claude-api
nginx -t && systemctl restart nginx
```

### Step 5：远程配置 Claude Code

**settings.json：**

```bash
mkdir -p ~/.claude
cat > ~/.claude/settings.json << 'EOF'
{
  "env": {
    "ENABLE_TOOL_SEARCH": "false"
  },
  "model": "ppio/pa/claude-opus-4-6",
  "baseUrl": "http://127.0.0.1:9081/anthropic",
  "apiKey": "<你的 API Key>",
  "defaultModel": "ppio/pa/claude-opus-4-6",
  "autoSwitchModel": false,
  "skipDangerousModePermissionPrompt": true
}
EOF
```

注意：`baseUrl` 指向 nginx 的 9081 端口，不是 SSH 隧道的 9080。

**环境变量（~/.bashrc）：**

```bash
cat >> ~/.bashrc << 'EOF'

# Claude Code proxy (through SSH reverse tunnel, for npm etc.)
export https_proxy=http://127.0.0.1:8888
export http_proxy=http://127.0.0.1:8888
export HTTPS_PROXY=http://127.0.0.1:8888
export HTTP_PROXY=http://127.0.0.1:8888

# Claude Code API config
export ANTHROPIC_API_KEY="<你的 API Key>"
export ANTHROPIC_BASE_URL="http://127.0.0.1:9081/anthropic"

# 关键：排除 localhost 不走代理，否则 Claude Code 访问 nginx 会被代理拦截
export no_proxy="127.0.0.1,localhost"
export NO_PROXY="127.0.0.1,localhost"
EOF

source ~/.bashrc
```

### Step 6：本地创建一键连接脚本

```bash
cat > ~/connect_gpu.sh << 'SCRIPT'
#!/bin/bash
# 一键连接 GPU 服务器 (带反向隧道)
#
# 隧道说明:
#   远程 9080 -> 本地 -> model.mify.ai.srv:80  (Claude API)
#   远程 8888 -> 本地 -> tinyproxy:8888         (通用代理, npm等)

GPU_HOST="root@172.16.0.110"

# 确保本地 tinyproxy 在跑
if ! pgrep -x tinyproxy > /dev/null; then
    echo "[*] Starting tinyproxy..."
    sudo systemctl start tinyproxy
    sleep 1
fi

if pgrep -x tinyproxy > /dev/null; then
    echo "[+] tinyproxy running on port 8888"
else
    echo "[!] tinyproxy not running (npm may not work, but Claude API will)"
fi

echo "[*] Connecting to GPU server with reverse tunnels..."
echo "    Port 9080: Claude API (model.mify.ai.srv:80)"
echo "    Port 8888: General proxy (tinyproxy)"
echo ""
echo "    After login, run:  tmux new -s claude"
echo "                       cd /home/chenshuai/Project/TactileACT-cs"
echo "                       claude"
echo ""

ssh -o ServerAliveInterval=60 -o ServerAliveCountMax=3 \
    -R 9080:model.mify.ai.srv:80 \
    -R 8888:127.0.0.1:8888 \
    ${GPU_HOST}
SCRIPT

chmod +x ~/connect_gpu.sh
```

## 日常使用

```bash
# 1. 本地执行
~/connect_gpu.sh

# 2. 远程执行
tmux new -s claude                              # 新建 tmux（首次）
# 或
tmux attach -t claude                           # 重连 tmux（断线恢复）

# 3. 进入项目，启动 Claude Code
cd /home/chenshuai/Project/TactileACT-cs
claude
```

## 踩坑记录

### 坑 1：API 返回 403 Forbidden (openresty)

**现象**：Claude Code 启动后无法对话，API 返回 403。

**原因**：SSH 隧道转发的请求 Host header 是 `127.0.0.1:9080`，API 网关根据 Host 做路由，不认识这个 Host。

**解法**：在远程用 nginx 做反向代理，`proxy_set_header Host model.mify.ai.srv` 改写 Host。

### 坑 2：UND_ERR_ABORTED 连接中断

**现象**：Claude Code 报 `Unable to connect to API (UND_ERR_ABORTED)`。

**原因**：`.bashrc` 中设置了 `http_proxy=127.0.0.1:8888`，Claude Code (Node.js undici) 访问 `127.0.0.1:9081` (nginx) 时被代理拦截，走了 tinyproxy 而不是直连 nginx。

**解法**：添加 `no_proxy=127.0.0.1,localhost`，让 localhost 请求不走代理。

### 坑 3：SSH 端口冲突 "remote port forwarding failed"

**现象**：新建 SSH 连接时提示 `Warning: remote port forwarding failed for listen port 9080`。

**原因**：上一次 SSH 连接没正常关闭，旧的 sshd 进程还占着端口。

**解法**：在远程手动清理 `fuser -k 9080/tcp && fuser -k 8888/tcp`，或等旧连接超时自动释放。

## 端口说明

| 端口 | 位置 | 服务 | 说明 |
|------|------|------|------|
| 8888 | 本地 | tinyproxy | HTTP 代理，远程 npm/apt 通过此上网 |
| 8888 | 远程 | SSH 隧道 | 映射到本地 tinyproxy |
| 9080 | 远程 | SSH 隧道 | 映射到本地 model.mify.ai.srv:80 |
| 9081 | 远程 | nginx | 反向代理，加 Host header 后转发到 9080 |
