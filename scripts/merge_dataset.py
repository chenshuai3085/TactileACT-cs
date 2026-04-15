"""
合并数据集脚本：将 ms_tmp 中三个目录的 hdf5 文件合并到 0401020307，重新编号。

顺序：260401_0402(0~359) → 260403/bounce(360~449) → 260403/success(450~539)
     → 260407/bounce(540~629) → 260407/success(630~719)

安全措施：
1. 先清空目标目录
2. 用 Python int() 排序，不用 shell sort
3. 每个文件复制后校验文件大小
4. 最后做全量校验：逐一对比源和目标的文件大小
"""

import os
import shutil
import sys


def get_sorted_hdf5(directory):
    """获取目录下所有 hdf5 文件，按 episode 编号数字排序"""
    files = [f for f in os.listdir(directory) if f.endswith('.hdf5')]
    files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    return files


def main():
    TMP = "/sharedata/chenshuai/data/dataset/ms_tmp"
    TARGET = "/sharedata/chenshuai/data/dataset/0401020307"

    # 定义源目录和顺序
    sources = [
        ("260401_0402", os.path.join(TMP, "260401_0402")),
        ("260403/bounce", os.path.join(TMP, "260403/bounce")),
        ("260403/success", os.path.join(TMP, "260403/success")),
        ("260407/bounce", os.path.join(TMP, "260407/bounce")),
        ("260407/success", os.path.join(TMP, "260407/success")),
    ]

    # Step 0: 统计源文件
    all_source_files = []  # [(src_path, source_name, original_name)]
    for name, src_dir in sources:
        if not os.path.isdir(src_dir):
            print("ERROR: source dir not found: %s" % src_dir)
            sys.exit(1)
        files = get_sorted_hdf5(src_dir)
        print("%s: %d files (ep_%s ~ ep_%s)" % (
            name, len(files),
            files[0].split('_')[1].split('.')[0] if files else '?',
            files[-1].split('_')[1].split('.')[0] if files else '?'))
        for f in files:
            all_source_files.append((os.path.join(src_dir, f), name, f))

    total = len(all_source_files)
    print("\nTotal source files: %d" % total)
    print("Target dir: %s" % TARGET)
    print()

    # Step 1: 清空目标目录
    os.makedirs(TARGET, exist_ok=True)
    old_files = [f for f in os.listdir(TARGET) if f.endswith('.hdf5')]
    print("Removing %d old files from target..." % len(old_files))
    for f in old_files:
        os.remove(os.path.join(TARGET, f))
    print("Target cleared.")

    # Step 2: 逐个复制并校验
    copy_log = []  # [(new_idx, src_path, dst_path, src_size, dst_size)]
    errors = []

    for idx, (src_path, src_name, orig_name) in enumerate(all_source_files):
        dst_name = "episode_%d.hdf5" % idx
        dst_path = os.path.join(TARGET, dst_name)

        src_size = os.path.getsize(src_path)
        shutil.copy2(src_path, dst_path)
        dst_size = os.path.getsize(dst_path)

        if src_size != dst_size:
            errors.append("SIZE MISMATCH: %s -> %s (src=%d dst=%d)" % (
                src_path, dst_path, src_size, dst_size))

        copy_log.append((idx, src_path, dst_path, src_size, dst_size))

        if (idx + 1) % 100 == 0:
            print("  copied %d/%d ..." % (idx + 1, total))

    print("\nCopy done: %d files" % len(copy_log))

    # Step 3: 全量校验
    print("\n=== VERIFICATION ===")
    dst_files = [f for f in os.listdir(TARGET) if f.endswith('.hdf5')]
    print("Files in target: %d (expected %d)" % (len(dst_files), total))

    # 检查编号连续性
    dst_ids = sorted([int(f.split('_')[1].split('.')[0]) for f in dst_files])
    expected_ids = list(range(total))
    if dst_ids == expected_ids:
        print("Episode IDs: 0-%d CONTINUOUS OK" % (total - 1))
    else:
        missing = set(expected_ids) - set(dst_ids)
        extra = set(dst_ids) - set(expected_ids)
        if missing:
            print("MISSING IDs: %s" % sorted(missing))
        if extra:
            print("EXTRA IDs: %s" % sorted(extra))

    # 逐一对比源和目标文件大小
    size_mismatches = 0
    for idx, src_path, dst_path, src_size, dst_size in copy_log:
        actual_dst_size = os.path.getsize(dst_path)
        if src_size != actual_dst_size:
            print("  MISMATCH ep_%d: src=%d dst=%d" % (idx, src_size, actual_dst_size))
            size_mismatches += 1

    if size_mismatches == 0:
        print("File sizes: ALL MATCH")
    else:
        print("File sizes: %d MISMATCHES!" % size_mismatches)

    if errors:
        print("\nERRORS during copy:")
        for e in errors:
            print("  " + e)

    # 打印映射表（前几个和分界点）
    print("\n=== MAPPING (showing boundaries) ===")
    boundaries = [0, 1, 2, 359, 360, 449, 450, 539, 540, 629, 630, 719]
    for idx, src_path, dst_path, src_size, dst_size in copy_log:
        if idx in boundaries:
            print("  episode_%d <- %s (%d bytes)" % (idx, src_path, src_size))

    print("\nDONE")


if __name__ == '__main__':
    main()
