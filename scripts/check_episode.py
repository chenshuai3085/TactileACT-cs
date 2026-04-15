"""
数据质量检查脚本：检查所有模态的恒定值片段
用法: python check_data_quality.py --dataset_dir /path/to/dataset --threshold 5
"""
import h5py
import numpy as np
import argparse
import os
import json

# 要检查的模态，格式: (hdf5路径, 描述, 是否图像模态)
MODALITIES = [
    ('actions/joint_abs',                    'action',       False),
    ('observations/proprio_joint',           'proprio',      False),
    ('observations/tac/left/marker_offset',  'tac_marker',   False),
    ('ft',                                   'force_torque', False),
    ('joint_current',                        'joint_cur',    False),
    ('observations/images/global',           'img_global',   True),
    ('observations/images/wrist',            'img_wrist',    True),
    ('observations/tac/left/img',            'tac_img',      True),
]


def check_consecutive_stuck(arr, threshold, is_image=False):
    """
    检查 arr (T, ...) 中连续恒定不变的片段。
    图像模态用帧均值代替逐像素比较（快）。
    返回: list of (start, end, length) 的坏片段，超过 threshold 的
    """
    if is_image:
        # 用每帧的均值作为代表值，检查均值是否不变
        frame_repr = arr.reshape(len(arr), -1).mean(axis=1)   # (T,)
        diff = np.abs(np.diff(frame_repr))                     # (T-1,)
        zero = diff < 1e-4
    else:
        diff = np.abs(np.diff(arr, axis=0))                    # (T-1, ...)
        diff = diff.reshape(len(diff), -1).max(axis=1)         # (T-1,)
        zero = diff == 0

    segments = []
    cur_start, cur_len = None, 0
    for t, z in enumerate(zero):
        if z:
            if cur_start is None:
                cur_start = t
            cur_len += 1
        else:
            if cur_len >= threshold:
                segments.append((cur_start, cur_start + cur_len, cur_len))
            cur_start, cur_len = None, 0
    if cur_len >= threshold:
        segments.append((cur_start, cur_start + cur_len, cur_len))

    return segments


def check_episode(path, threshold):
    """检查单个 episode 的所有模态，返回问题报告"""
    results = {}
    try:
        with h5py.File(path, 'r') as f:
            T = f['actions/joint_abs'].shape[0]
            for hdf5_key, name, is_image in MODALITIES:
                if hdf5_key not in f:
                    continue
                arr = f[hdf5_key][()]
                segs = check_consecutive_stuck(arr, threshold, is_image)
                if segs:
                    results[name] = segs
        return T, results, None
    except Exception as e:
        return None, {}, str(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str,
                        default='/sharedata/chenshuai/data/dataset/0401020307')
    parser.add_argument('--threshold', type=int, default=5,
                        help='连续不变帧数阈值')
    parser.add_argument('--output', type=str, default=None,
                        help='结果保存路径 (json)')
    args = parser.parse_args()

    base = args.dataset_dir
    threshold = args.threshold

    # 自动检测 episode 数量
    n_episodes = len([f for f in os.listdir(base)
                      if f.startswith('episode_') and f.endswith('.hdf5')])
    print(f'共 {n_episodes} 个 episode，阈值: 连续 >= {threshold} 帧不变\n')

    bad_episodes = {}
    corrupt_episodes = {}

    for i in range(n_episodes):
        path = os.path.join(base, f'episode_{i}.hdf5')
        T, results, err = check_episode(path, threshold)

        if err:
            corrupt_episodes[i] = err
            print(f'ep{i:3d}: [损坏] {err}')
            continue

        if results:
            bad_episodes[i] = {'T': T, 'modalities': {}}
            parts = []
            for name, segs in results.items():
                seg_str = ','.join(f'[{s}-{e}]({l}帧)' for s, e, l in segs)
                bad_episodes[i]['modalities'][name] = [[s, e, l] for s, e, l in segs]
                parts.append(f'{name}:{seg_str}')
            print(f'ep{i:3d} (T={T:3d}): ' + ' | '.join(parts))

    print(f'\n=== 汇总 ===')
    print(f'损坏文件: {len(corrupt_episodes)} 个')
    print(f'有恒定片段: {len(bad_episodes)} 个 / {n_episodes}')
    print(f'clean episode: {n_episodes - len(bad_episodes) - len(corrupt_episodes)} 个')

    print('\n各模态问题数:')
    modality_counts = {}
    for ep_data in bad_episodes.values():
        for name in ep_data['modalities']:
            modality_counts[name] = modality_counts.get(name, 0) + 1
    for name, cnt in sorted(modality_counts.items(), key=lambda x: -x[1]):
        print(f'  {name}: {cnt} 个 episode')

    if args.output:
        out = {
            'threshold': threshold,
            'n_episodes': n_episodes,
            'corrupt': {str(k): v for k, v in corrupt_episodes.items()},
            'bad_episodes': {str(k): v for k, v in bad_episodes.items()},
        }
        with open(args.output, 'w') as f:
            json.dump(out, f, indent=2)
        print(f'\n结果已保存到 {args.output}')


if __name__ == '__main__':
    main()
