# 【NICOLE2026】
"""从完整验证集中按退化等级分层抽样，构建小验证子集（默认 200 张）。

动机：完整 val（1000 张）每次验证开销大；直接取前 N 张会使
mild/medium/severe_controlled 的比例失真（severe 仅约 15%，小样本下波动被放大）。
本脚本读取 degradation_meta.jsonl 中逐图的 level 记录，按完整验证集的
经验比例做最大余数法分层抽样，并把 HQ/LQ/face_weight 三份文件同步到新目录。

服务器用法：
    cd /root/autodl-tmp/MambaIR
    python tools/oldphoto/make_val_subset.py \
        --meta datasets/FFHQ_pair/degradation_meta.jsonl \
        --val_root datasets/FFHQ_pair/val \
        --out_root datasets/FFHQ_pair/val_200 \
        --num 200 --seed 42

运行后训练配置的 val 路径指向 out_root（v2/A0/A1 已改好），先跑本脚本再开训。
"""
import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

SUBDIRS = ('HQ', 'LQ', 'face_weight')


def load_val_records(meta_path):
    """读取 degradation_meta.jsonl 中 split == 'val' 的记录。"""
    records = []
    with open(meta_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get('split') == 'val':
                records.append(rec)
    if not records:
        raise ValueError(f'meta 中没有 split=="val" 的记录: {meta_path}')
    return records


def allocate_per_level(level_counts, num):
    """最大余数法按经验比例分配各 level 的抽样数，总和恰为 num 且不超过可用数。"""
    total = sum(level_counts.values())
    if num > total:
        raise ValueError(f'请求 {num} 张，但验证集只有 {total} 张。')
    raw = {lv: num * c / total for lv, c in level_counts.items()}
    alloc = {lv: min(int(v), level_counts[lv]) for lv, v in raw.items()}
    remainder = num - sum(alloc.values())
    # 按小数部分从大到小补齐余数；层内已满则跳过
    order = sorted(raw, key=lambda lv: raw[lv] - int(raw[lv]), reverse=True)
    while remainder > 0:
        progressed = False
        for lv in order:
            if remainder <= 0:
                break
            if alloc[lv] < level_counts[lv]:
                alloc[lv] += 1
                remainder -= 1
                progressed = True
        if not progressed:
            raise RuntimeError('分配失败：各层均已用尽。')
    return alloc


def main():
    parser = argparse.ArgumentParser(description='Build stratified validation subset.')
    parser.add_argument('--meta', default='datasets/FFHQ_pair/degradation_meta.jsonl',
                        help='degradation_meta.jsonl 路径（build_dataset 的输出）。')
    parser.add_argument('--val_root', default='datasets/FFHQ_pair/val',
                        help='完整验证集根目录（含 HQ/LQ/face_weight 子目录）。')
    parser.add_argument('--out_root', default='datasets/FFHQ_pair/val_200',
                        help='子集输出根目录。')
    parser.add_argument('--num', type=int, default=200, help='子集大小。')
    parser.add_argument('--seed', type=int, default=42, help='抽样随机种子。')
    args = parser.parse_args()

    val_root = Path(args.val_root)
    out_root = Path(args.out_root)

    records = load_val_records(args.meta)
    by_level = defaultdict(list)
    for rec in records:
        by_level[rec['level']].append(rec['filename'])

    level_counts = {lv: len(names) for lv, names in by_level.items()}
    alloc = allocate_per_level(level_counts, args.num)

    rng = random.Random(args.seed)
    chosen = []
    for lv in sorted(by_level):
        # sorted 保证文件顺序与生成顺序无关，同 seed 结果可复现
        chosen.extend((lv, name) for name in rng.sample(sorted(by_level[lv]), alloc[lv]))

    # 复制前先校验三份文件齐全，避免生成半套数据
    missing = []
    for _, name in chosen:
        for sub in SUBDIRS:
            src = val_root / sub / name
            if not src.exists():
                missing.append(str(src))
    if missing:
        preview = '\n  '.join(missing[:10])
        raise FileNotFoundError(f'{len(missing)} 个源文件缺失，例如:\n  {preview}')

    for sub in SUBDIRS:
        (out_root / sub).mkdir(parents=True, exist_ok=True)
    for _, name in chosen:
        for sub in SUBDIRS:
            shutil.copy2(val_root / sub / name, out_root / sub / name)

    manifest = {
        'source_val_root': str(val_root),
        'num': args.num,
        'seed': args.seed,
        'source_level_counts': level_counts,
        'subset_level_counts': alloc,
        'filenames': sorted(name for _, name in chosen),
    }
    manifest_path = out_root / 'subset_manifest.json'
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f'完成：{args.num} 张已复制到 {out_root}')
    print(f'{"level":<20}{"full val":>10}{"subset":>8}')
    for lv in sorted(level_counts):
        print(f'{lv:<20}{level_counts[lv]:>10}{alloc[lv]:>8}')
    print(f'清单已写入: {manifest_path}')


if __name__ == '__main__':
    main()
# 【NICOLE2026】
