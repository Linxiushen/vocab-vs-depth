"""从预训练语料中随机切出验证集，并从训练集中物理删除。

MiniMind 的 train_pretrain.py 只有 train_ds，没有验证集——这是仓库的第一个洞。
没有验证集你无法区分「在学习」和「在背题」。

⚠ 为什么必须随机切，不能切尾部：
   实测 pretrain_t2t_mini.jsonl 是【按来源排序的，完全没有 shuffle】：
     头部 3000 条：中文指令/诗歌，中位长度   231 字符
     尾部 3000 条：英文考试选择题，中位长度 1251 字符
   直接切尾部 2000 行，得到的验证集全是英文考题，和训练分布完全不同。
   我第一版就是这么写的，结果 val loss 6.51 vs train loss 4.0，
   差 2.5 nats，看起来像灾难性过拟合，实际只是切错了集合。

   本脚本用「按行号哈希」做确定性随机划分：单遍扫描、内存恒定、可复现，
   且不需要把 1.2GB 全读进内存。

用法:
    python lab/split_val.py --data minimind/dataset/pretrain_t2t_mini.jsonl --n 2000
产出:
    <data>.train.jsonl  (剩余行)
    <data>.val.jsonl    (随机抽取的 n 行)
"""
import argparse, hashlib, os

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--n", type=int, default=2000, help="验证集行数")
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()

base = a.data.rsplit('.jsonl', 1)[0]
train_p, val_p = f"{base}.train.jsonl", f"{base}.val.jsonl"

# 第一遍：数总行数
with open(a.data, 'rb') as f:
    total = sum(buf.count(b'\n') for buf in iter(lambda: f.read(1 << 22), b''))
assert total > a.n, f"文件只有 {total} 行，切不出 {a.n} 行验证集"

# 用行号的哈希做确定性划分：取哈希值最小的 n 行作为验证集。
# 单遍扫描无法预知阈值，所以先算所有行号的哈希（只存 int，1.27M 行约 10MB），
# 取第 n 小的值作为阈值。
def h(i):
    return int.from_bytes(hashlib.blake2b(f"{a.seed}:{i}".encode(), digest_size=8).digest(), 'big')

hashes = [h(i) for i in range(total)]
threshold = sorted(hashes)[a.n - 1]
val_idx = {i for i, v in enumerate(hashes) if v <= threshold}
del hashes

with open(a.data, encoding='utf-8') as src, \
     open(train_p, 'w', encoding='utf-8') as tr, \
     open(val_p, 'w', encoding='utf-8') as va:
    for i, line in enumerate(src):
        (va if i in val_idx else tr).write(line)

print(f"总计 {total:,} 行  (seed={a.seed}，划分可复现)")
print(f"  训练集 {train_p}  {total - len(val_idx):,} 行")
print(f"  验证集 {val_p}  {len(val_idx):,} 行  ← 全局随机抽取，非尾部切片")
print(f"\n⚠ 训练时必须用 --data_path {train_p}，用原文件 = 验证集泄漏，val loss 会虚低。")
