#!/usr/bin/env python3
"""专利二：完整智能装配系统 — 新增配图（完整闭环工作流程图）"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 9,
    'figure.dpi': 200,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
})

OUTPUT_DIR = '/home/chenshuai/Project/TactileACT-cs/paper/patent/figures'


def add_box(ax, xy, w, h, text, fs=7.5, bold=False, fc='white', ec='black'):
    x, y = xy
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03",
                         facecolor=fc, edgecolor=ec, linewidth=1.2, zorder=2)
    ax.add_patch(box)
    wt = 'bold' if bold else 'normal'
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fs, fontweight=wt, zorder=3)


def add_arrow(ax, start, end, color='black', lw=1.0, style='->'):
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw), zorder=1)


def add_text(ax, x, y, text, fs=8, ha='center', va='center',
             bold=False, style='normal', color='black'):
    wt = 'bold' if bold else 'normal'
    ax.text(x, y, text, ha=ha, va=va, fontsize=fs,
            fontweight=wt, fontstyle=style, color=color)


# ==================== 完整闭环工作流程图 ====================
def fig_complete_workflow():
    fig, ax = plt.subplots(1, 1, figsize=(14, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis('off')
    ax.set_title('Fig.4  Complete Intelligent Assembly Closed-Loop Workflow',
                 fontsize=12, fontweight='bold', pad=10)

    # === Phase 1: Normal Assembly (top row) ===
    add_text(ax, 3.5, 6.6, 'Phase 1: Normal Assembly Execution', fs=10, bold=True)

    # Sensor Input
    add_box(ax, (0.2, 5.0), 2.0, 1.2,
            'Multi-Modal\nPerception\n\nVision + Tactile\n+F/T + Proprio', fs=7, fc='#f0f0f0')

    # Perception Encoder
    add_box(ax, (2.8, 5.0), 2.0, 1.2,
            'Perception\nEncoder\n\nResNet18+PointNet\n1D Conv+Joint Norm', fs=7)
    add_arrow(ax, (2.2, 5.6), (2.8, 5.6))

    # Diffusion Policy
    add_box(ax, (5.4, 5.0), 2.0, 1.2,
            'Diffusion Policy\nAction Generator\n\nDDPM Denoising\n20-step Action Chunk', fs=7, bold=True)
    add_arrow(ax, (4.8, 5.6), (5.4, 5.6))

    # Execute
    add_box(ax, (8.0, 5.0), 1.8, 1.2,
            'Execute\nJoint Control\n\n7-DOF Robot\nArm Movement', fs=7)
    add_arrow(ax, (7.4, 5.6), (8.0, 5.6))

    # Loop back from Execute to Perception
    ax.annotate('', xy=(1.2, 5.0), xytext=(8.9, 5.0),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.0,
                                connectionstyle='arc3,rad=0.5', linestyle='dashed'))

    # === Phase 2: Real-time Anomaly Detection (middle row) ===
    add_text(ax, 3.5, 4.5, 'Phase 2: Real-time Anomaly Monitoring (Parallel)', fs=10, bold=True)

    # LSTM Anomaly Detector
    add_box(ax, (0.2, 3.0), 2.5, 1.2,
            'LSTM Anomaly\nDetector\n\nF/T Pattern Analysis\np_anomaly > 0.7?', fs=7, fc='#fff3e0')

    # Connection from Perception to LSTM
    add_arrow(ax, (3.8, 5.0), (1.45, 4.2), color='gray', lw=0.8)
    add_text(ax, 3.0, 4.4, 'shared encoding', fs=6, style='italic', color='gray')

    # Normal path
    add_box(ax, (3.5, 3.0), 1.5, 1.2,
            'Normal\nContinue\nAssembly', fs=7.5, fc='#e8f5e9')
    add_arrow(ax, (2.7, 3.6), (3.5, 3.6))

    # Anomaly trigger
    add_box(ax, (5.5, 3.0), 1.5, 1.2,
            'Anomaly\nDetected!\nTrigger LLM', fs=7.5, fc='#ffebee', ec='red')
    add_arrow(ax, (2.7, 3.3), (5.5, 3.3), color='red', lw=1.5)

    # === Phase 3: LLM Diagnosis (bottom row) ===
    add_text(ax, 9.5, 4.5, 'Phase 3: LLM Diagnosis & Correction', fs=10, bold=True)

    # Signal Tokenizer
    add_box(ax, (7.5, 3.0), 1.8, 1.2,
            'Signal\nTokenizer\n\nF/T -> 62 tokens\nTac -> 8 tokens\nVis -> 1 token', fs=6.5)
    add_arrow(ax, (7.0, 3.6), (7.5, 3.6), color='red', lw=1.5)

    # RAG Retrieval
    add_box(ax, (7.5, 1.5), 1.8, 1.0,
            'RAG Case\nRetrieval\nFAISS Top-3', fs=7, fc='#f0f0f0')

    # LLM
    add_box(ax, (9.8, 2.2), 2.0, 2.0,
            'Multimodal\nLLM (7B)\n\nCoT Reasoning\nRAG Context\nSensor Tokens\n\n-> JSON Diagnosis', fs=7, bold=True)
    add_arrow(ax, (9.3, 3.6), (9.8, 3.6))
    add_arrow(ax, (9.3, 2.0), (9.8, 2.8))

    # Correction Head
    add_box(ax, (12.2, 3.0), 1.5, 1.2,
            'Correction\nHead (MLP)\n\n8-dim Params\nGeometric\nConstraints', fs=6.5, fc='#e3f2fd')
    add_arrow(ax, (11.8, 3.6), (12.2, 3.6))

    # Safety Grading
    add_box(ax, (12.2, 1.5), 1.5, 1.0,
            'Safety\nGrading\n\nLow/Med/High\nRisk Levels', fs=7, fc='#fff3e0')
    add_arrow(ax, (12.95, 3.0), (12.95, 2.5))

    # Execute Correction
    add_box(ax, (9.8, 0.5), 2.0, 1.0,
            'Execute\nCorrection\n\nForce-Protected\nRetry (max 3)', fs=7, fc='#e8f5e9')
    add_arrow(ax, (12.95, 1.5), (10.8, 1.5))

    # Experience accumulation
    add_box(ax, (7.5, 0.5), 1.8, 1.0,
            'Experience\nAccumulation\n\nCase Library\nPeriodic Fine-tune', fs=6.5, fc='#f3e5f5')
    add_arrow(ax, (8.4, 2.5), (8.4, 1.5))

    # Feedback loop: correction back to normal assembly
    ax.annotate('', xy=(8.9, 5.0), xytext=(10.8, 1.5),
                arrowprops=dict(arrowstyle='->', color='green', lw=1.5,
                                connectionstyle='arc3,rad=-0.3'))
    add_text(ax, 11.5, 3.0, 'Resume\nAssembly', fs=7, style='italic', color='green')

    # Bottom note
    add_text(ax, 7.0, 0.1,
             'Complete Loop: Perceive -> Generate -> Monitor -> Diagnose -> Correct -> Verify -> Learn',
             fs=8, style='italic')

    fig.savefig(os.path.join(OUTPUT_DIR, 'patent2_fig5_complete_workflow.png'))
    plt.close()
    print('  Fig.5 done: complete_workflow')


if __name__ == '__main__':
    print('Generating Patent 2 additional figure...')
    fig_complete_workflow()
    print('Done!')
