#!/usr/bin/env python3
"""
专利二：多模态异常诊断与修正 — 高质量技术配图
包含人形机器人示意图
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Arc
import numpy as np
import os

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
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw),
                zorder=zorder)


def add_text(ax, x, y, text, fontsize=8, ha='center', va='center',
             bold=False, style='normal', color='black'):
    weight = 'bold' if bold else 'normal'
    ax.text(x, y, text, ha=ha, va=va, fontsize=fontsize,
            fontweight=weight, fontstyle=style, color=color)


# ==================== 图1: 人形机器人应用场景 ====================
def fig1_humanoid_scenario():
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis('off')
    ax.set_title('Fig.1  Humanoid Robot Bolt Assembly Workstation', fontsize=12, fontweight='bold', pad=10)

    # === 人形机器人 (简化示意图) ===
    cx, cy = 5.0, 4.5  # 机器人中心

    # 头部 (圆形)
    head = Circle((cx, cy + 1.2), 0.35, facecolor='#e0e0e0', edgecolor='black', lw=1.5)
    ax.add_patch(head)
    # 眼睛 (双目相机)
    ax.plot(cx - 0.12, cy + 1.25, 'ko', markersize=3)
    ax.plot(cx + 0.12, cy + 1.25, 'ko', markersize=3)
    add_text(ax, cx, cy + 1.6, 'Head Camera\n(Stereo)', fontsize=6, style='italic')

    # 躯干
    torso = FancyBboxPatch((cx - 0.5, cy - 0.2), 1.0, 1.2,
                           boxstyle="round,pad=0.05",
                           facecolor='#d0d0d0', edgecolor='black', lw=1.5)
    ax.add_patch(torso)

    # 左臂 (抓螺栓)
    arm_l_x = [cx - 0.5, cx - 1.2, cx - 1.5]
    arm_l_y = [cy + 0.6, cy + 0.3, cy - 0.3]
    ax.plot(arm_l_x, arm_l_y, 'k-', lw=2.5, solid_capstyle='round')
    # 左手 (灵巧手 + 触觉)
    add_rounded_box(ax, (cx - 1.8, cy - 0.6), 0.6, 0.4, 'Tactile\nFingertip', fontsize=5.5, facecolor='#f8f8f8')
    # 螺栓
    add_rounded_box(ax, (cx - 1.7, cy - 1.1), 0.4, 0.35, 'Bolt', fontsize=6, bold=True, facecolor='#e8e8e8')
    add_arrow(ax, (cx - 1.5, cy - 0.6), (cx - 1.5, cy - 0.75))

    # 右臂 (持工具)
    arm_r_x = [cx + 0.5, cx + 1.2, cx + 1.4]
    arm_r_y = [cy + 0.6, cy + 0.2, cy - 0.4]
    ax.plot(arm_r_x, arm_r_y, 'k-', lw=2.5, solid_capstyle='round')
    # 右手 + F/T sensor
    add_rounded_box(ax, (cx + 1.1, cy - 0.7), 0.6, 0.4, 'F/T Sensor\n(6-axis)', fontsize=5.5, facecolor='#f8f8f8')

    # 左腿
    ax.plot([cx - 0.3, cx - 0.4, cx - 0.35], [cy - 0.2, cy - 1.2, cy - 1.8], 'k-', lw=2.5, solid_capstyle='round')
    # 右腿
    ax.plot([cx + 0.3, cx + 0.4, cx + 0.35], [cy - 0.2, cy - 1.2, cy - 1.8], 'k-', lw=2.5, solid_capstyle='round')
    # 脚
    ax.plot([cx - 0.6, cx - 0.15], [cy - 1.8, cy - 1.8], 'k-', lw=3)
    ax.plot([cx + 0.15, cx + 0.6], [cy - 1.8, cy - 1.8], 'k-', lw=3)

    # === 螺纹孔 (工件) ===
    add_rounded_box(ax, (7.5, 2.0), 1.5, 0.8, 'Threaded Hole\n(Chassis Panel)', fontsize=7, facecolor='#f0f0f0')
    ax.plot([7.5, 9.0, 9.0, 7.5, 7.5], [2.0, 2.0, 1.5, 1.5, 2.0], 'k-', lw=1.5)

    # === 边缘计算模块 ===
    add_rounded_box(ax, (0.3, 0.5), 2.5, 1.0,
                    'Edge Computing Module\n(Jetson AGX Orin)\n\nAbnormality Detection\nLLM Diagnosis\nVisual Measurement',
                    fontsize=6.5, facecolor='#f0f0f0')

    # === 远程监控 ===
    add_rounded_box(ax, (7.5, 0.5), 2.2, 1.0,
                    'Remote Monitor\nDiagnosis Report\nOperator Confirm/Override',
                    fontsize=6.5, facecolor='#f0f0f0')
    add_arrow(ax, (2.8, 1.0), (7.5, 1.0), style='->', color='gray')
    add_text(ax, 5.15, 1.15, 'Network', fontsize=7, style='italic', color='gray')

    # 传感器标注线
    add_text(ax, cx - 2.5, cy - 0.3, 'GelSight tactile\n(9x9 marker, 30Hz)', fontsize=6, ha='right', style='italic')
    add_text(ax, cx + 2.5, cy - 0.4, '6-axis F/T sensor\n(500Hz)', fontsize=6, ha='left', style='italic')

    fig.savefig(os.path.join(OUTPUT_DIR, 'patent2_fig1_humanoid_scenario.png'))
    plt.close()
    print('  Fig.1 done: humanoid_scenario')


# ==================== 图2: 系统架构 ====================
def fig2_system_architecture():
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')
    ax.set_title('Fig.2  System Architecture: Perceive -> Diagnose -> Correct', fontsize=11, fontweight='bold', pad=10)

    # 感知层
    add_rounded_box(ax, (0.3, 3.5), 2.2, 1.8,
                    'Perception Layer\n\nF/T Sensor (500Hz)\nGelSight Tactile\nHead Stereo Camera\nWrist Camera\nJoint Encoders',
                    fontsize=7, facecolor='#f0f0f0')

    # 信号处理层
    add_rounded_box(ax, (3.0, 3.5), 2.2, 1.8,
                    'Signal Processing\n\nAnomaly Detection (LSTM)\nSignal -> Text Translation\nFeature Extraction\nForce-Tactile Tokenizer',
                    fontsize=7)
    add_arrow(ax, (2.5, 4.4), (3.0, 4.4))

    # 大模型推理层
    add_rounded_box(ax, (5.7, 3.5), 2.8, 1.8,
                    'Diagnosis Reasoning\n\nMultimodal LLM (7B)\n+ RAG Case Retrieval\n+ Chain-of-Thought\n-> Diagnosis + Plan',
                    fontsize=7, bold=True)
    add_arrow(ax, (5.2, 4.4), (5.7, 4.4))

    # 执行层
    add_rounded_box(ax, (9.0, 3.5), 2.2, 1.8,
                    'Execution Layer\n\nCorrection Planning\nSafety Constraint Check\nRobot Control\nForce-Protected Motion',
                    fontsize=7)
    add_arrow(ax, (8.5, 4.4), (9.0, 4.4))

    # 经验积累层 (底部)
    add_rounded_box(ax, (3.0, 1.2), 6.0, 1.2,
                    'Experience Accumulation Loop\n\nCase Library Expansion -> RAG Retrieval Improve -> Periodic LoRA Fine-tune -> Better Diagnosis',
                    fontsize=7, facecolor='#f5f5f5')
    add_arrow(ax, (7.1, 3.5), (7.1, 2.4))

    # 人机界面
    add_rounded_box(ax, (9.5, 1.2), 2.0, 1.2,
                    'Human-Machine\nInterface\n\nDiagnosis Report\nOperator Confirm',
                    fontsize=7, facecolor='#f0f0f0')
    add_arrow(ax, (10.1, 3.5), (10.1, 2.4))

    # 底部核心特征
    add_text(ax, 6.0, 0.5,
             'Core: Language as intermediate representation between perception and action | LLM causal reasoning | Case library for domain experience',
             fontsize=7, style='italic')

    fig.savefig(os.path.join(OUTPUT_DIR, 'patent2_fig2_system_architecture.png'))
    plt.close()
    print('  Fig.2 done: system_architecture')


# ==================== 图3: 诊断流程 ====================
def fig3_diagnosis_workflow():
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis('off')
    ax.set_title('Fig.3  "Perceive -> Diagnose -> Plan -> Correct" Closed-Loop Workflow',
                 fontsize=11, fontweight='bold', pad=10)

    steps = [
        ('S1: Anomaly\nPerception\n\nMulti-modal\nSignal Capture\nAnomaly Prob > 0.7\nTrigger Diagnosis', 0.2),
        ('S2: Signal\nTokenization\n\nForce Curve -> Tokens\nTactile -> Tokens\nVision -> Token\n~5ms', 2.5),
        ('S3: LLM\nDiagnosis\n\nChain-of-Thought\nRAG Case Match\nOutput: Cause + Plan\n~1.5-2s', 4.8),
        ('S4: Correction\nPlanning\n\nParam. Regression\nSafety Check\nGraded Control\n~2ms', 7.1),
        ('S5: Execute\n& Verify\n\nForce-Protected\nMotion Execution\nResult Verify\n5-15s', 9.4),
    ]

    for text, x in steps:
        add_rounded_box(ax, (x, 1.5), 2.1, 2.5, text, fontsize=6.5, bold=True)

    # 箭头
    for i in range(len(steps) - 1):
        x1 = steps[i][1] + 2.1
        x2 = steps[i+1][1]
        add_arrow(ax, (x1, 2.75), (x2, 2.75))

    # 反馈循环
    ax.annotate('', xy=(steps[0][1] + 1.05, 1.5), xytext=(steps[-1][1] + 1.05, 0.3),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.2,
                                connectionstyle='arc3,rad=0'))
    ax.annotate('', xy=(steps[0][1] + 1.05, 0.3), xytext=(steps[-1][1] + 1.05, 0.3),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.2))
    add_text(ax, 5.8, 0.15, 'Fail -> Retry (max 3 times) or escalate to human operator', fontsize=7, style='italic', color='gray')

    # 时序标注
    add_text(ax, 6.0, 4.5, 'S1(real-time) | S2(~5ms) | S3(1.5-2s) | S4(~2ms) | S5(5-15s)', fontsize=7, style='italic')
    add_text(ax, 6.0, 4.2, 'Total decision latency: ~1.6-2.1s (from anomaly trigger to correction start)', fontsize=7, style='italic', bold='bold')

    fig.savefig(os.path.join(OUTPUT_DIR, 'patent2_fig3_diagnosis_workflow.png'))
    plt.close()
    print('  Fig.3 done: diagnosis_workflow')


# ==================== 图4: 信号分词器 ====================
def fig4_signal_tokenizer():
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')
    ax.set_title('Fig.4  Force-Tactile Signal Tokenizer Architecture', fontsize=11, fontweight='bold', pad=10)

    # === 力矩分支 (上方) ===
    add_text(ax, 0.5, 5.5, 'Force/Torque Branch', fontsize=9, bold='bold', ha='left')

    force_boxes = [
        ('F/T Input\n(500x6)\n1s sequence', 0.3, 4.2, 1.8),
        ('1D Causal Conv\n3 layers\n6->64->128->256\n500->62 steps', 2.4, 4.2, 2.0),
        ('Bi-GRU\nhidden=256\nGlobal temporal', 4.7, 4.2, 1.8),
        ('VQ Layer\nCodebook=512\n-> 62 tokens', 6.8, 4.2, 1.8),
    ]

    for text, x, y, w in force_boxes:
        add_rounded_box(ax, (x, y), w, 0.9, text, fontsize=6.5)

    for i in range(len(force_boxes) - 1):
        x1 = force_boxes[i][1] + force_boxes[i][3]
        x2 = force_boxes[i+1][1]
        add_arrow(ax, (x1, 4.65), (x2, 4.65))

    # === 触觉分支 (中间) ===
    add_text(ax, 0.5, 3.7, 'Tactile Branch', fontsize=9, bold='bold', ha='left')

    tac_boxes = [
        ('Tactile Input\n(30x9x9x2)\n1s sequence', 0.3, 2.4, 1.8),
        ('2D Spatial Conv\n+ 1D Causal Conv\n2->32->64 ch\n30->8 frames', 2.4, 2.4, 2.0),
        ('Spatial Pool\n3x3 -> 1x1\n-> (8x64)', 4.7, 2.4, 1.8),
        ('VQ Layer\nCodebook=256\n-> 8 tokens', 6.8, 2.4, 1.8),
    ]

    for text, x, y, w in tac_boxes:
        add_rounded_box(ax, (x, y), w, 0.9, text, fontsize=6.5)

    for i in range(len(tac_boxes) - 1):
        x1 = tac_boxes[i][1] + tac_boxes[i][3]
        x2 = tac_boxes[i+1][1]
        add_arrow(ax, (x1, 2.85), (x2, 2.85))

    # === 视觉分支 (下方) ===
    add_text(ax, 0.5, 1.9, 'Vision Branch', fontsize=9, bold='bold', ha='left')

    add_rounded_box(ax, (0.3, 0.8), 1.8, 0.7, 'Wrist Image\n(1 frame)', fontsize=7)
    add_rounded_box(ax, (2.4, 0.8), 2.0, 0.7, 'ViT-B/14\n-> CLS token\n768D -> 4096D', fontsize=7)
    add_rounded_box(ax, (4.7, 0.8), 1.8, 0.7, '1 Vision Token\n(continuous\nembedding)', fontsize=7)

    add_arrow(ax, (2.1, 1.15), (2.4, 1.15))
    add_arrow(ax, (4.4, 1.15), (4.7, 1.15))

    # === 汇总输出 (右侧) ===
    add_rounded_box(ax, (9.0, 1.5), 2.5, 3.0,
                    'Output to LLM\n\n62 Force tokens\n+ 8 Tactile tokens\n+ 1 Vision token\n= 71 Sensor tokens\n\nCombined with\ntext tokens\nfor LLM input',
                    fontsize=7, bold=True, facecolor='#f0f0f0')

    add_arrow(ax, (8.6, 4.65), (9.0, 3.5))
    add_arrow(ax, (8.6, 2.85), (9.0, 2.8))
    add_arrow(ax, (6.5, 1.15), (9.0, 2.0))

    # 两阶段训练说明
    add_text(ax, 6.0, 0.2, 'Stage 1: Self-supervised reconstruction pre-training (VQ-VAE style, ~100K curves) | Stage 2: Diagnostic alignment fine-tuning (LoRA, ~500 labeled)',
             fontsize=6.5, style='italic')

    fig.savefig(os.path.join(OUTPUT_DIR, 'patent2_fig4_signal_tokenizer.png'))
    plt.close()
    print('  Fig.4 done: signal_tokenizer')


if __name__ == '__main__':
    print('Generating Patent 2 figures...')
    fig1_humanoid_scenario()
    fig2_system_architecture()
    fig3_diagnosis_workflow()
    fig4_signal_tokenizer()
    print(f'All Patent 2 figures saved to {OUTPUT_DIR}/')
