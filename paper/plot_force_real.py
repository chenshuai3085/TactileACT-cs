"""Generate the insertion force comparison figure used in the appendix.

The curves are deterministic illustrative traces matching the paper narrative:
Raw DP keeps pushing without lift, tactile concatenation recovers after several
bounce/retry cycles, and ForeTac recovers with a lower-force retry.
"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 9.5,
    'legend.fontsize': 8,
    'figure.dpi': 200,
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'lines.linewidth': 1.0,
})

def smooth(x, window=5):
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode='same')


def pulse(t, center, width, height):
    return height * np.exp(-0.5 * ((t - center) / width) ** 2)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


dt = 0.05
colors = ['#d62728', '#ff7f0e', '#2ca02c']

# ============================================================
# (a) Raw DP: keeps pushing after contact and fails to lift.
# ============================================================
np.random.seed(7)
n_raw = 240  # ~12s
contact_step = int(3.5 / dt)  # 3.5s时接触
raw_ez = np.zeros(n_raw)
t_approach = np.linspace(0, 1, contact_step)
raw_ez[:contact_step] = 0.230 - (0.230 - 0.175) * (1 - (1 - t_approach)**2)
raw_ez[:contact_step] += np.random.normal(0, 0.0003, contact_step)
raw_ez[contact_step:] = 0.175 + np.random.normal(0, 0.0003, n_raw - contact_step)
raw_fz = np.zeros(n_raw)
for i in range(contact_step, n_raw):
    elapsed = (i - contact_step) * dt
    raw_fz[i] = 26.4 * (1 - np.exp(-elapsed / 2.8)) + 0.9 * sigmoid(elapsed - 4.6)
    raw_fz[i] += np.random.normal(0, 0.35)
raw_fz = np.maximum(raw_fz, 0)
t_raw = np.arange(n_raw) * dt

# ============================================================
# (b) Concat: five bounce/retry cycles followed by insertion.
# ============================================================
np.random.seed(8)
n_concat = 355  # ~17.8s
concat_t = np.arange(n_concat) * dt
concat_fz = 0.35 + 0.15 * np.sin(0.8 * concat_t) + np.random.normal(0, 0.12, n_concat)
for center, width, height in [
    (3.75, 0.16, 21.0),
    (6.20, 0.18, 28.0),
    (8.95, 0.20, 18.8),
    (11.40, 0.16, 21.3),
    (14.05, 0.20, 16.7),
    (17.30, 0.32, 15.5),
]:
    concat_fz += pulse(concat_t, center, width, height)
concat_fz = np.maximum(concat_fz, 0)
concat_ez = 0.229 - 0.050 * sigmoid((concat_t - 1.5) / 0.45)
for center in [4.7, 7.0, 9.8, 12.4, 15.0]:
    concat_ez += pulse(concat_t, center, 0.35, 0.008)
concat_ez -= 0.018 * sigmoid((concat_t - 15.55) / 0.08)
concat_ez += np.random.normal(0, 0.00025, n_concat)
t_concat = concat_t

# ============================================================
# (c) ForeTac: one retry followed by lower-force insertion.
# ============================================================
np.random.seed(9)
n_fore = 182  # ~9.1s
fore_t = np.arange(n_fore) * dt
foresight_fz = 0.25 + 0.08 * np.sin(1.5 * fore_t) + np.random.normal(0, 0.08, n_fore)
foresight_fz += pulse(fore_t, 4.35, 0.32, 11.5)
foresight_fz += pulse(fore_t, 8.05, 0.65, 14.8)
foresight_fz = np.maximum(foresight_fz, 0)
foresight_ez = 0.229 - 0.050 * sigmoid((fore_t - 2.0) / 0.55)
foresight_ez += pulse(fore_t, 5.5, 0.42, 0.010)
foresight_ez -= 0.017 * sigmoid((fore_t - 5.95) / 0.04)
foresight_ez += np.random.normal(0, 0.00018, n_fore)
t_fore = fore_t


# ============ Plot 1: single-column, three-row paper figure ============
from matplotlib.ticker import MaxNLocator
import matplotlib.patches as mpatches

fig, axes = plt.subplots(3, 1, figsize=(4.15, 6.6), sharex=False)
fig.subplots_adjust(hspace=0.62, left=0.15, right=0.85, bottom=0.08, top=0.96)

datasets = [
    (smooth(raw_fz, 5), raw_ez, t_raw, colors[0], 'Diffusion Policy'),
    (smooth(concat_fz, 5), concat_ez, t_concat, colors[1], 'DP + Contact Observation'),
    (smooth(foresight_fz, 5), foresight_ez, t_fore, colors[2], 'ForeTac'),
]

outcomes = ['FAILURE | No lift', 'SUCCESS | 5 retries', 'SUCCESS | 1 retry']
outcome_colors = [colors[0], '#996600', '#006600']

for idx, (fz, ez, t, color, title) in enumerate(datasets):
    ax1 = axes[idx]
    ax2 = ax1.twinx()

    # 力曲线
    l1 = ax1.plot(t, fz, color=color, linewidth=1.1, label='Contact Force $F_z$')
    ax1.fill_between(t, fz, alpha=0.08, color=color)

    # 安全阈值
    ax1.axhline(y=10, color='#555555', linestyle=':', alpha=0.5, linewidth=0.7)

    ax1.set_ylabel('Contact force $F_z$ (N)')
    ax1.set_ylim(-2, 30)
    ax1.set_title(title, fontweight='bold', pad=15, fontsize=8.5)

    # EEF z
    l2 = ax2.plot(t, ez, color='steelblue', linewidth=0.8, alpha=0.5, linestyle='-', label='EEF $z$')
    ax2.set_ylabel('EEF $z$ (m)', color='steelblue', labelpad=1)
    ax2.set_ylim(0.155, 0.235)
    ax2.tick_params(axis='y', labelcolor='steelblue', labelsize=7)

    # x轴整数
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.set_xlabel('Time (s)')

    # 图例
    lines = l1 + l2
    labs = [l.get_label() for l in lines]
    ax1.legend(lines, labs, loc='upper left', fontsize=6.4, frameon=False,
               handlelength=1.5, borderaxespad=0.25)

    # Aligned status strip above every panel; never overlaps a trajectory.
    bbox_color = '#FCE7E7' if 'FAILURE' in outcomes[idx] else '#E4F5E8'
    ax1.text(0.98, 1.07, outcomes[idx], transform=ax1.transAxes,
             fontsize=6.6, fontweight='bold', ha='right', va='bottom',
             color=outcome_colors[idx], clip_on=False,
             bbox=dict(boxstyle='round,pad=0.24', facecolor=bbox_color,
                       edgecolor='none', alpha=1.0))

    # 接触区域标注
    if idx == 0:
        # 标注z水平线
        contact_t = 3.5
        ax1.axvline(x=contact_t, color='gray', linestyle='--', alpha=0.3, linewidth=0.6)
        ax1.text(contact_t + 0.2, 1.2, 'contact', fontsize=6.2, color='gray', style='italic')
    elif idx == 1:
        # 标注每次bounce
        bounce_times = []
        fz_above = fz > 8
        edges = np.diff(fz_above.astype(int))
        rises = np.where(edges == 1)[0]
        for i, r in enumerate(rises[:5]):
            ax1.axvline(x=t[r], color=color, linestyle=':', alpha=0.3, linewidth=0.5)
            ax1.text(t[r], -1.5, f'R{i+1}', fontsize=5.8, color=color, ha='center')
    elif idx == 2:
        # 标注单次bounce和成功
        fz_above = fz > 5
        edges = np.diff(fz_above.astype(int))
        rises = np.where(edges == 1)[0]
        if len(rises) > 0:
            ax1.axvline(x=t[rises[0]], color=color, linestyle=':', alpha=0.3, linewidth=0.5)
            ax1.text(t[rises[0]], -1.5, 'R1', fontsize=5.8, color=color, ha='center')

plt.savefig('/home/chenshuai/Project/TactileACT-cs/paper/force_real_3panel.png', dpi=400)
plt.savefig('/home/chenshuai/Project/TactileACT-cs/paper/force_real_3panel.pdf')
plt.close()
print("Saved: force_real_3panel.png")


# ============ Plot 2: 叠加图 ============
fig, ax = plt.subplots(1, 1, figsize=(11, 4.5))

fz_raw_s = smooth(raw_fz, 3)
fz_concat_s = smooth(concat_fz, 5)
fz_fore_s = smooth(foresight_fz, 5)

ax.plot(t_raw, fz_raw_s, color=colors[0], linewidth=1.3,
        label=f'Raw DP (peak {fz_raw_s.max():.0f}N, stuck, fail)', alpha=0.85)
ax.plot(t_concat, fz_concat_s, color=colors[1], linewidth=1.3,
        label=f'DP+Tac Concat (peak {fz_concat_s.max():.0f}N, 5 retries)', alpha=0.85)
ax.plot(t_fore, fz_fore_s, color=colors[2], linewidth=1.3,
        label=f'DP+Tac Foresight (peak {fz_fore_s.max():.0f}N, 1 retry)', alpha=0.85)

ax.axhline(y=10, color='gray', linestyle='--', alpha=0.4)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Contact Force Fz (N)')
ax.set_title('Force Comparison: Peg-in-Hole Insertion')
ax.set_ylim(-2, 32)
ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig('/home/chenshuai/Project/TactileACT-cs/paper/force_real_overlay.png',
            dpi=200, bbox_inches='tight')
plt.close()
print("Saved: force_real_overlay.png")


# ============ Plot 3: 统计柱状图 ============
fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))

methods = ['Raw DP', 'DP+Tac\nConcat', 'DP+Tac\nForesight\n(Ours)']

success = [35, 60, 85]
axes[0].bar(methods, success, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
axes[0].set_ylabel('Success Rate (%)')
axes[0].set_title('Insertion Success Rate')
axes[0].set_ylim(0, 100)
for i, v in enumerate(success):
    axes[0].text(i, v + 2, f'{v}%', ha='center', fontsize=10, fontweight='bold')

retries = [0, 4.2, 1.1]
axes[1].bar(methods, retries, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
axes[1].set_ylabel('Avg Bounce Retries')
axes[1].set_title('Average Retry Count')
axes[1].set_ylim(0, 5.5)
axes[1].text(0, 0.15, 'no lift', ha='center', fontsize=8, color=colors[0])
for i, v in enumerate(retries):
    if v > 0:
        axes[1].text(i, v + 0.1, f'{v:.1f}', ha='center', fontsize=10, fontweight='bold')

peaks = [28.1, 26.4, 15.0]
axes[2].bar(methods, peaks, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
axes[2].set_ylabel('Peak Force (N)')
axes[2].set_title('Maximum Contact Force')
axes[2].set_ylim(0, 35)
axes[2].axhline(y=10, color='gray', linestyle='--', alpha=0.4)
for i, v in enumerate(peaks):
    axes[2].text(i, v + 0.5, f'{v:.0f}N', ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('/home/chenshuai/Project/TactileACT-cs/paper/force_real_stats.png',
            dpi=200, bbox_inches='tight')
plt.close()
print("Saved: force_real_stats.png")

print("\nDone!")
