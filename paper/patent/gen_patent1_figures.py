#!/usr/bin/env python3
"""
专利一：触觉前瞻预测与接触质量评估 — 高质量技术配图
使用 matplotlib 生成黑白风格技术图，适合专利交底书
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

# 全局设置
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 9,
    'axes.linewidth': 0.8,
    'figure.dpi': 200,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
})

OUTPUT_DIR = '/home/chenshuai/Project/TactileACT-cs/paper/patent/figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def add_rounded_box(ax, xy, width, height, text, fontsize=8, bold=False,
                    facecolor='white', edgecolor='black', lw=1.0,
                    text_color='black', padding=0.02, zorder=2):
    """添加圆角矩形框+文字"""
    x, y = xy
    box = FancyBboxPatch((x, y), width, height,
                         boxstyle=f"round,pad={padding}",
                         facecolor=facecolor, edgecolor=edgecolor,
                         linewidth=lw, zorder=zorder)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x + width/2, y + height/2, text,
            ha='center', va='center', fontsize=fontsize,
            fontweight=weight, color=text_color, zorder=zorder+1)
    return box


def add_arrow(ax, start, end, color='black', lw=1.0, style='->', zorder=1):
    """添加箭头"""
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw),
                zorder=zorder)


def add_text(ax, x, y, text, fontsize=8, ha='center', va='center',
             style='normal', color='black', bold=False):
    """添加无框文字"""
    weight = 'bold' if bold else 'normal'
    ax.text(x, y, text, ha=ha, va=va, fontsize=fontsize,
            fontstyle=style, color=color, fontweight=weight)


# ==================== 图1: 系统硬件布局 ====================
def fig1_system_layout():
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    ax.set_title('Fig.1  System Hardware Layout', fontsize=12, fontweight='bold', pad=10)

    # 机械臂 (中心上方) - 用简化的机器人臂表示
    # 基座
    add_rounded_box(ax, (4.2, 4.5), 1.6, 0.6, 'Robot Arm (7-DOF)\nJoint Encoder: 7D', fontsize=7, bold=True)
    # 夹爪
    add_rounded_box(ax, (4.0, 3.5), 2.0, 0.7, 'Gripper + GelSight\n9x9 Marker Displacement, 30Hz', fontsize=7)

    # 连接线
    add_arrow(ax, (5.0, 4.5), (5.0, 4.2))

    # 全局相机 (左侧)
    add_rounded_box(ax, (0.5, 3.8), 2.2, 0.7, 'Global Camera\n200x266 RGB', fontsize=7)
    add_arrow(ax, (2.7, 4.15), (4.0, 4.15))

    # 腕部相机 (右侧)
    add_rounded_box(ax, (7.3, 3.8), 2.2, 0.7, 'Wrist Camera\n200x266 RGB', fontsize=7)
    add_arrow(ax, (7.3, 4.15), (6.0, 4.15))

    # 工件 (左下)
    add_rounded_box(ax, (2.5, 1.5), 2.0, 0.8, 'Target Part\n(USB/Pin/FPC)', fontsize=7)

    # 目标插槽 (右下)
    add_rounded_box(ax, (5.5, 1.5), 2.0, 0.8, 'Target Slot\n(Fixed on Workstation)', fontsize=7)

    # 连接
    add_arrow(ax, (5.0, 3.5), (3.5, 2.3))
    add_arrow(ax, (4.5, 1.9), (5.5, 1.9))

    # 工作台
    ax.plot([1.5, 8.5], [1.2, 1.2], 'k-', lw=2)
    add_text(ax, 5.0, 0.8, 'Workstation', fontsize=8)

    # 数据流标注
    add_text(ax, 5.0, 0.3, 'Output: Visual Images + Tactile Marker Displacement (9x9x2/frame) + Joint Angles (7D)  |  Control: 5~10 Hz',
             fontsize=7, style='italic')

    fig.savefig(os.path.join(OUTPUT_DIR, 'patent1_fig1_system_layout.png'))
    plt.close()
    print('  Fig.1 done: system_layout')


# ==================== 图2: 整体流程 ====================
def fig2_pipeline():
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis('off')
    ax.set_title('Fig.2  Overall Pipeline: "Compress -> Generate -> Predict -> Score -> Execute"',
                 fontsize=11, fontweight='bold', pad=10)

    # 输入
    add_rounded_box(ax, (0.2, 1.8), 1.8, 1.5, 'Sensor Input\n\nGlobal Camera\nWrist Camera\nTactile (8 frames)\nJoint Angles (7D)',
                    fontsize=6.5, facecolor='#f0f0f0')

    # 步骤一: TactileVAE
    add_rounded_box(ax, (2.5, 2.0), 1.8, 1.1, 'Step 1:\nTactileVAE\n162D -> 144D\nCausal Spatio-\nTemporal Encoding',
                    fontsize=6.5, bold=True)
    add_arrow(ax, (2.0, 2.55), (2.5, 2.55))

    # 步骤三: Diffusion Policy
    add_rounded_box(ax, (4.8, 2.0), 1.8, 1.1, 'Step 3:\nDiffusion Policy\nDDPM 100 steps\n-> K=16 Candidates',
                    fontsize=6.5, bold=True)
    add_arrow(ax, (4.3, 2.55), (4.8, 2.55))

    # 步骤二+四: Foresight + CQF
    add_rounded_box(ax, (7.1, 2.0), 2.0, 1.1, 'Step 2+4:\nForesight Transformer\n-> Predict Future Tac.\nCQF Scorer\n-> Rank Candidates',
                    fontsize=6.5, bold=True)
    add_arrow(ax, (6.6, 2.55), (7.1, 2.55))

    # 输出
    add_rounded_box(ax, (9.6, 2.0), 1.8, 1.1, 'Execute\n\nBest Candidate\nFirst 2 Steps\n-> Loop',
                    fontsize=6.5, bold=True, facecolor='#e8e8e8')
    add_arrow(ax, (9.1, 2.55), (9.6, 2.55))

    # 循环箭头
    ax.annotate('', xy=(10.5, 1.7), xytext=(10.5, 0.5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.2,
                                connectionstyle='arc3,rad=0'))
    ax.annotate('', xy=(1.1, 0.5), xytext=(10.5, 0.5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.2))
    ax.annotate('', xy=(1.1, 1.8), xytext=(1.1, 0.5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.2))
    add_text(ax, 5.8, 0.3, 'Get new observation & repeat', fontsize=7, style='italic', color='gray')

    # 离线/在线标注
    add_text(ax, 3.4, 4.2, 'Offline Training', fontsize=8, bold='bold')
    add_text(ax, 3.4, 3.9, 'Step 1 -> Step 2 -> Step 3: trained sequentially', fontsize=7, style='italic')

    add_text(ax, 8.1, 4.2, 'Online Inference', fontsize=8, bold='bold')
    add_text(ax, 8.1, 3.9, 'Input -> Compress -> K candidates -> Predict -> Score -> Execute', fontsize=7, style='italic')
    add_text(ax, 8.1, 3.6, 'Latency ~200ms (single GPU, K=16)', fontsize=7, style='italic')

    fig.savefig(os.path.join(OUTPUT_DIR, 'patent1_fig2_pipeline.png'))
    plt.close()
    print('  Fig.2 done: pipeline')


# ==================== 图3: TactileVAE 结构 ====================
def fig3_tactile_vae():
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')
    ax.set_title('Fig.3  TactileVAE: Causal Spatio-Temporal Encoder-Decoder', fontsize=11, fontweight='bold', pad=10)

    # === 编码器 (上半部分) ===
    add_text(ax, 0.5, 5.5, 'Encoder (Causal: no future frames)', fontsize=9, ha='left', bold='bold')

    enc_y = 4.2
    enc_h = 0.9
    enc_boxes = [
        ('Input\n(B,8,9,9,2)\nMarker Seq.', 0.3, 1.5),
        ('Causal 3D Conv\n2->32 ch\nKernel 3x3x3', 2.1, 1.6),
        ('ResBlock+Down\n32->64 ch\n9x9->5x5', 4.0, 1.6),
        ('ResBlock+Down\n64->128 ch\n5x5->3x3\n8->4 frames', 5.9, 1.6),
        ('Latent Proj.\nmu, logsigma2\n(16,4,3,3)', 7.8, 1.5),
        ('Attn Pool\n4 frames->1\nz:(16,3,3)=144D', 9.6, 1.5),
    ]

    for text, x, w in enc_boxes:
        add_rounded_box(ax, (x, enc_y), w, enc_h, text, fontsize=6.5)

    # 编码器箭头
    for i in range(len(enc_boxes) - 1):
        x1 = enc_boxes[i][1] + enc_boxes[i][2]
        x2 = enc_boxes[i+1][1]
        add_arrow(ax, (x1, enc_y + enc_h/2), (x2, enc_y + enc_h/2))

    # === 解码器 (下半部分) ===
    add_text(ax, 0.5, 2.8, 'Decoder (Cross-Attention)', fontsize=9, ha='left', bold='bold')

    dec_y = 1.5
    dec_h = 0.9

    add_rounded_box(ax, (1.0, dec_y), 2.2, dec_h, 'Latent z\n(B,16,3,3)\n9 spatial pos as KV', fontsize=7)
    add_rounded_box(ax, (4.0, dec_y), 3.2, dec_h, 'Cross-Attention x2\n81 Queries (9x9 grid)\n+ 2D Sine Pos. Enc.\nEach output sees all 9 KV', fontsize=7)
    add_rounded_box(ax, (8.0, dec_y), 2.2, dec_h, 'Output\n(B,9,9,2)\nReconstructed\nMarker Displacement', fontsize=7)

    add_arrow(ax, (3.2, dec_y + dec_h/2), (4.0, dec_y + dec_h/2))
    add_arrow(ax, (7.2, dec_y + dec_h/2), (8.0, dec_y + dec_h/2))

    # z连接线 (从编码器到解码器)
    ax.annotate('', xy=(2.1, dec_y + dec_h), xytext=(8.55, enc_y),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.0,
                                connectionstyle='arc3,rad=-0.3', linestyle='dashed'))

    # 力度-模式解耦标注
    add_text(ax, 6.0, 0.7, 'Key Design: Force-Pattern Decoupling', fontsize=8, bold='bold')
    add_text(ax, 6.0, 0.4, 'z[0] = force channel (supervised by ||marker||_2)  |  z[1:15] = pattern channel',
             fontsize=7, style='italic')
    add_text(ax, 6.0, 0.1, 'Loss = MSE + 0.2*Cosine + 1e-6*KL + 0.1*Force Supervision + 0.05*Ranking',
             fontsize=7, style='italic')

    fig.savefig(os.path.join(OUTPUT_DIR, 'patent1_fig3_tactile_vae.png'))
    plt.close()
    print('  Fig.3 done: tactile_vae')


# ==================== 图4: 前瞻变换器 ====================
def fig4_foresight_transformer():
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis('off')
    ax.set_title('Fig.4  Foresight Transformer: Factorized Spatio-Temporal Attention (x3 Layers)',
                 fontsize=11, fontweight='bold', pad=10)

    # === 左侧: 输入 ===
    add_text(ax, 1.0, 6.5, 'Inputs', fontsize=9, bold='bold')

    input_x = 0.3
    input_w = 2.5
    input_h = 0.8

    inputs = [
        ('Visual Tokens (63)\nResNet18 -> 512D\n+ Camera Embedding', 5.5),
        ('Tactile Tokens (9)\nVAE Latent -> 512D\n+ Spatial Pos. Emb.', 4.4),
        ('Action Tokens\nCandidate 10 steps x 7D\n-> Linear Projection', 3.3),
    ]

    for text, y in inputs:
        add_rounded_box(ax, (input_x, y), input_w, input_h, text, fontsize=6.5)

    # === 中间: 三种注意力 ===
    add_text(ax, 5.0, 6.5, 'Factorized Attention (per layer)', fontsize=9, bold='bold')

    attn_x = 3.5
    attn_w = 3.5
    attn_h = 0.9

    attns = [
        ('(a) Spatial Self-Attention\nAll V+T tokens interact\nwithin same timestep', 5.3),
        ('(b) Temporal Self-Attention\nSame spatial position\nacross k history frames', 4.1),
        ('(c) Action Cross-Attention\nQ = V/T tokens\nK/V = Action tokens', 2.9),
    ]

    for text, y in attns:
        add_rounded_box(ax, (attn_x, y), attn_w, attn_h, text, fontsize=6.5, bold=True)

    # 输入到注意力的箭头
    add_arrow(ax, (input_x + input_w, 5.5 + input_h/2), (attn_x, 5.3 + attn_h/2))
    add_arrow(ax, (input_x + input_w, 4.4 + input_h/2), (attn_x, 4.1 + attn_h/2))
    add_arrow(ax, (input_x + input_w, 3.3 + input_h/2), (attn_x, 2.9 + attn_h/2))

    # 注意力之间的向下箭头
    add_arrow(ax, (attn_x + attn_w/2, 5.3), (attn_x + attn_w/2, 4.1 + attn_h))
    add_arrow(ax, (attn_x + attn_w/2, 4.1), (attn_x + attn_w/2, 2.9 + attn_h))

    # === 右侧: 输出 ===
    out_x = 7.8
    out_w = 2.0

    add_rounded_box(ax, (out_x, 5.3), out_w, 0.8, 'FFN + Residual\n+ LayerNorm', fontsize=7)
    add_rounded_box(ax, (out_x, 4.1), out_w, 0.8, 'Repeat x3 Layers', fontsize=7, bold=True)
    add_rounded_box(ax, (out_x, 2.6), out_w, 1.0, 'Output:\nTactile Tokens\n-> Linear\n-> z_pred (144D)', fontsize=7, bold=True)

    add_arrow(ax, (attn_x + attn_w, 5.3 + attn_h/2), (out_x, 5.3 + 0.4))
    add_arrow(ax, (out_x + out_w/2, 5.3), (out_x + out_w/2, 4.1 + 0.8))
    add_arrow(ax, (out_x + out_w/2, 4.1), (out_x + out_w/2, 2.6 + 1.0))

    # 底部训练说明
    add_text(ax, 5.0, 1.5, 'Training: Freeze ResNet18 + TactileVAE', fontsize=8, bold='bold')
    add_text(ax, 5.0, 1.1, 'Condition: GT action trajectory', fontsize=7, style='italic')
    add_text(ax, 5.0, 0.8, 'Target: z_{t+H} (future tactile latent)', fontsize=7, style='italic')
    add_text(ax, 5.0, 0.5, 'Loss: L1 in latent space', fontsize=7, style='italic')

    fig.savefig(os.path.join(OUTPUT_DIR, 'patent1_fig4_foresight_transformer.png'))
    plt.close()
    print('  Fig.4 done: foresight_transformer')


# ==================== 图5: CQF评分器 ====================
def fig5_cqf_scorer():
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis('off')
    ax.set_title('Fig.5  CQF Contact Quality Scorer: Three-Branch Architecture',
                 fontsize=11, fontweight='bold', pad=10)

    # === 输入行 ===
    add_text(ax, 2.0, 6.5, 'Inputs', fontsize=9, bold='bold')

    inputs = [
        ('[z_cur, z_pred, delta_z]\n432D (144x3)', 0.3, 5.5),
        ('Candidate Action\n140D (20 steps x 7D)', 3.5, 5.5),
        ('Current Joint Angle\n7D', 7.0, 5.5),
    ]

    for text, x, y in inputs:
        add_rounded_box(ax, (x, y), 2.5, 0.9, text, fontsize=7)

    # === 分支行 ===
    branches = [
        ('Tactile Branch (Core)\n432->256->256\nLayerNorm + ReLU', 0.3, 3.8),
        ('Action Branch\n140->256->256\n50% Dropout (train)', 3.5, 3.8),
        ('State Branch\n7->256->256\nLayerNorm + ReLU', 7.0, 3.8),
    ]

    for text, x, y in branches:
        add_rounded_box(ax, (x, y), 2.5, 0.9, text, fontsize=7)
        add_arrow(ax, (x + 1.25, 5.5), (x + 1.25, 3.8 + 0.9))

    # === 融合 ===
    add_rounded_box(ax, (2.0, 2.2), 6.0, 0.9,
                    'Concat [h_tac, h_act, h_state] = 768D  ->  Fusion MLP: 768->256->128->1',
                    fontsize=7.5, bold=True)

    for x in [0.3, 3.5, 7.0]:
        add_arrow(ax, (x + 1.25, 3.8), (5.0, 2.2 + 0.9))

    # === 输出 ===
    add_rounded_box(ax, (3.5, 0.8), 3.0, 0.8,
                    'Output: Scalar Score S\n(Higher = Better Contact Quality)',
                    fontsize=7.5, bold=True, facecolor='#e8e8e8')
    add_arrow(ax, (5.0, 2.2), (5.0, 0.8 + 0.8))

    # 训练说明
    add_text(ax, 5.0, 0.2, 'Training: Adaptive-margin pairwise ranking loss | Action branch 50% zero-out to prevent shortcut',
             fontsize=7, style='italic')

    fig.savefig(os.path.join(OUTPUT_DIR, 'patent1_fig5_cqf_scorer.png'))
    plt.close()
    print('  Fig.5 done: cqf_scorer')


if __name__ == '__main__':
    print('Generating Patent 1 figures...')
    fig1_system_layout()
    fig2_pipeline()
    fig3_tactile_vae()
    fig4_foresight_transformer()
    fig5_cqf_scorer()
    print(f'All Patent 1 figures saved to {OUTPUT_DIR}/')
