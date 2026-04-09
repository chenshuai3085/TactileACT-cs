#!/bin/bash
# 创建 4 窗格 tmux 布局 (2行×2列)

SESSION="claude4"

# 如果会话已存在，直接 attach
tmux has-session -t $SESSION 2>/dev/null
if [ $? -eq 0 ]; then
    echo "会话 $SESSION 已存在，正在连接..."
    tmux attach -t $SESSION
    exit 0
fi

# 创建新会话
tmux new-session -d -s $SESSION

# 水平分割 -> 上下两行
tmux split-window -v -t $SESSION

# 上面一行垂直分割
tmux select-pane -t $SESSION:0.0
tmux split-window -h -t $SESSION

# 下面一行垂直分割
tmux select-pane -t $SESSION:0.2
tmux split-window -h -t $SESSION

# 自动平铺布局
tmux select-layout -t $SESSION tiled

# 选中第一个窗格
tmux select-pane -t $SESSION:0.0

# 连接到会话
tmux attach -t $SESSION
