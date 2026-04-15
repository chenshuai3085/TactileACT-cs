"""
数据集完整性检查脚本
检查内容：
1. 文件完整性：是否能打开、通道是否齐全、帧数是否对齐
2. 数据质量：全零帧、NaN帧、连续冻结帧
3. 唯一帧比例：每个通道有多少帧是唯一的（非重复）
4. proprio跳变检测：关节角度突变
5. episode去重：通过图像hash检测重复episode

用法：
  python scripts/check_dataset.py --data_dir /path/to/dataset
  python scripts/check_dataset.py --data_dir /path/to/dataset --episodes 13 46 61
  python scripts/check_dataset.py --data_dir /path/to/dataset --start 0 --end 360
"""

import h5py
import os
import numpy as np
import argparse


EXPECTED_CHANNELS = [
    'actions/eef_abs',
    'actions/joint_abs',
    'ft',
    'joint_current',
    'observations/images/global',
    'observations/images/wrist',
    'observations/proprio_eef',
    'observations/proprio_joint',
    'observations/tac/left/depth',
    'observations/tac/left/force6d',
    'observations/tac/left/img',
    'observations/tac/left/marker_offset',
    'observations/tac/right/depth',
    'observations/tac/right/force6d',
    'observations/tac/right/img',
    'observations/tac/right/marker_offset',
]

CONTENT_CHECK_CHANNELS = [
    'observations/images/global',
    'observations/images/wrist',
    'observations/tac/left/img',
    'observations/tac/left/depth',
    'observations/tac/right/img',
    'observations/tac/right/depth',
    'observations/tac/left/marker_offset',
    'observations/tac/right/marker_offset',
    'observations/proprio_joint',
    'observations/proprio_eef',
    'actions/joint_abs',
    'actions/eef_abs',
    'ft',
    'observations/tac/left/force6d',
    'observations/tac/right/force6d',
    'joint_current',
]


def check_episode(fpath, ep_id, verbose=True):
    """检查单个episode的完整性和数据质量"""
    result = {
        'ep_id': ep_id,
        'status': 'OK',
        'total_frames': 0,
        'missing_channels': [],
        'misaligned_channels': {},
        'issues': [],
    }

    try:
        with h5py.File(fpath, 'r') as f:
            # --- 1. 通道完整性 ---
            present = []
            def visitor(name, obj):
                if isinstance(obj, h5py.Dataset):
                    present.append(name)
            f.visititems(visitor)

            for ch in EXPECTED_CHANNELS:
                if ch not in f:
                    result['missing_channels'].append(ch)

            if result['missing_channels']:
                result['status'] = 'MISSING_CHANNELS'

            # --- 2. 帧数对齐 ---
            frame_counts = {}
            for ch in CONTENT_CHECK_CHANNELS:
                if ch in f:
                    frame_counts[ch] = f[ch].shape[0]

            unique_counts = set(frame_counts.values())
            if len(unique_counts) > 1:
                main_count = max(set(frame_counts.values()),
                                 key=list(frame_counts.values()).count)
                result['misaligned_channels'] = {
                    ch: cnt for ch, cnt in frame_counts.items() if cnt != main_count
                }
                result['status'] = 'MISALIGNED'
                result['total_frames'] = main_count
            elif unique_counts:
                result['total_frames'] = list(unique_counts)[0]

            total = result['total_frames']

            # --- 3. 逐通道内容检查 ---
            if verbose:
                print('=' * 70)
                print('episode_%d  frames=%d' % (ep_id, total))
                print('-' * 70)

            for ch in CONTENT_CHECK_CHANNELS:
                if ch not in f:
                    if verbose:
                        print('  %-45s MISSING' % ch)
                    continue

                data = f[ch][:]
                n = data.shape[0]
                flat = data.reshape(n, -1)

                # 全零帧
                zero_mask = np.all(flat == 0, axis=1)
                n_zero = int(zero_mask.sum())

                # NaN帧
                nan_mask = np.any(np.isnan(flat), axis=1)
                n_nan = int(nan_mask.sum())

                # 连续冻结帧（>=3帧连续相同）
                frozen_ranges = []
                start = -1
                for i in range(1, n):
                    if np.array_equal(flat[i], flat[i - 1]):
                        if start == -1:
                            start = i - 1
                    else:
                        if start != -1 and (i - start) >= 3:
                            frozen_ranges.append((start, i - 1))
                        start = -1
                if start != -1 and (n - start) >= 3:
                    frozen_ranges.append((start, n - 1))

                # 唯一帧数
                unique = 1
                for i in range(1, n):
                    if not np.array_equal(flat[i], flat[i - 1]):
                        unique += 1

                # 汇总
                parts = ['shape=%s' % str(data.shape)]
                parts.append('unique=%d/%d(%.0f%%)' % (unique, n, 100.0 * unique / n))
                if n_zero > 0:
                    zero_idx = np.where(zero_mask)[0]
                    parts.append('ZERO(%d): %s' % (n_zero, zero_idx[:10].tolist()))
                if n_nan > 0:
                    nan_idx = np.where(nan_mask)[0]
                    parts.append('NAN(%d): %s' % (n_nan, nan_idx[:10].tolist()))
                if frozen_ranges:
                    fr_str = ', '.join(['%d-%d(%df)' % (s, e, e - s + 1)
                                        for s, e in frozen_ranges])
                    parts.append('FROZEN: %s' % fr_str)

                has_issue = n_zero > 0 or n_nan > 0 or len(frozen_ranges) > 0 or unique < n * 0.9
                if has_issue:
                    result['issues'].append(ch)

                if verbose:
                    status_mark = '!!' if has_issue else '  '
                    print('%s%-43s %s' % (status_mark, ch, '  '.join(parts)))

            # --- 4. proprio跳变 ---
            if 'observations/proprio_joint' in f:
                proprio = f['observations/proprio_joint'][:]
                diffs = np.abs(np.diff(proprio, axis=0))
                max_diff = diffs.max(axis=1)
                big_jumps = np.where(max_diff > 0.5)[0]
                if verbose and len(big_jumps) > 0:
                    print('  proprio jumps(>0.5rad): frames %s' % big_jumps[:20].tolist())

            # --- 5. 图像hash（用于去重）---
            if 'observations/images/global' in f:
                img_first = f['observations/images/global'][0]
                result['img_hash'] = hash(img_first.tobytes()[:2000]) % 1000000

            if verbose:
                print()

    except Exception as e:
        result['status'] = 'CORRUPT'
        result['issues'].append(str(e))
        if verbose:
            print('episode_%d  CORRUPT: %s' % (ep_id, e))

    return result


def main():
    parser = argparse.ArgumentParser(description='数据集完整性检查')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--episodes', type=int, nargs='+', default=None,
                        help='指定检查的episode编号')
    parser.add_argument('--start', type=int, default=None)
    parser.add_argument('--end', type=int, default=None)
    parser.add_argument('--quiet', action='store_true',
                        help='只输出有问题的episode')
    args = parser.parse_args()

    # 收集所有episode文件
    all_files = sorted(
        [f for f in os.listdir(args.data_dir) if f.endswith('.hdf5')],
        key=lambda x: int(x.split('_')[1].split('.')[0])
    )
    all_ids = [int(f.split('_')[1].split('.')[0]) for f in all_files]

    # 确定检查范围
    if args.episodes:
        check_ids = args.episodes
    elif args.start is not None or args.end is not None:
        s = args.start if args.start is not None else min(all_ids)
        e = args.end if args.end is not None else max(all_ids) + 1
        check_ids = [i for i in all_ids if s <= i < e]
    else:
        check_ids = all_ids

    print('Data dir: %s' % args.data_dir)
    print('Total files: %d, checking: %d' % (len(all_files), len(check_ids)))
    print()

    # 检查编号连续性
    expected_ids = set(range(len(all_files)))
    actual_ids = set(all_ids)
    missing_ids = expected_ids - actual_ids
    if missing_ids:
        print('WARNING: Missing episode IDs: %s' % sorted(missing_ids))

    results = []
    img_hashes = {}
    n_issues = 0

    for i, ep_id in enumerate(check_ids):
        fpath = os.path.join(args.data_dir, 'episode_%d.hdf5' % ep_id)
        if not os.path.exists(fpath):
            print('episode_%d: FILE NOT FOUND' % ep_id)
            continue

        verbose = not args.quiet
        r = check_episode(fpath, ep_id, verbose=verbose)
        results.append(r)

        if r['issues']:
            n_issues += 1
            if args.quiet:
                print('episode_%d(%df): %s' % (
                    ep_id, r['total_frames'], ', '.join(r['issues'])))

        # 去重检测
        if 'img_hash' in r:
            h = r['img_hash']
            if h in img_hashes:
                print('WARNING: episode_%d may be duplicate of episode_%d (same img hash %d)' % (
                    ep_id, img_hashes[h], h))
            img_hashes[h] = ep_id

        if (i + 1) % 100 == 0:
            print('--- checked %d/%d ---' % (i + 1, len(check_ids)))

    # 汇总
    print('=' * 70)
    print('SUMMARY')
    print('  Checked: %d episodes' % len(results))
    print('  Issues: %d episodes' % n_issues)
    print('  Corrupt: %d' % sum(1 for r in results if r['status'] == 'CORRUPT'))
    print('  Missing channels: %d' % sum(1 for r in results if r['missing_channels']))
    print('  Misaligned: %d' % sum(1 for r in results if r['misaligned_channels']))

    frames = [r['total_frames'] for r in results if r['total_frames'] > 0]
    if frames:
        print('  Frame range: %d ~ %d, mean=%.1f, std=%.1f' % (
            min(frames), max(frames), np.mean(frames), np.std(frames)))
    print('=' * 70)


if __name__ == '__main__':
    main()
