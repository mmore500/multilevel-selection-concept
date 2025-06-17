import glob
import sys
import uuid
import polars as pl


pattern = sys.argv[1]

dirs = glob.glob(pattern)


translation_dict_test = {
    "a": "nop-A",         # a
    "b": "nop-B",         # b
    "c": "nop-C",         # c
    "d": "if-n-equ",      # d
    "e": "if-less",       # e
    "f": "if-label",      # f
    "g": "mov-head",      # g
    "h": "jmp-head",      # h
    "i": "get-head",      # i
    "j": "set-flow",      # j
    "k": "shift-r",       # k
    "l": "shift-l",       # l
    "m": "inc",           # m
    "n": "dec",           # n
    "o": "push",          # o
    "p": "pop",           # p
    "q": "swap-stk",      # q
    "r": "swap",          # r 
    "s": "add",           # s
    "t": "sub",           # t
    "u": "nand",          # u
    "v": "h-copy",        # v
    "w": "h-alloc",       # w
    "x": "h-divide",      # x
    "y": "IO",            # y
    "z": "h-search",      # z
    "A": "rotate-l",
    "B": "read-faced-cell-org-id",
    "C": "get-cell-x",
    "D": "get-cell-y",
    "E": "get-cell-xy",
    "F": "get-north-offset"
}

seq_dict = {translation_dict_test[k] : k for k in translation_dict_test}


dfs = []

for dir in dirs:
    print(dir)
    rep_uuid = str(uuid.uuid4())
    replicate_num = dir.split("/")[-1].split("-")[-1]
    trt_name = dir.split("/")[-2]

    ancestral_seq = []
    with open(f"{dir}/default-heads.org", "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                ancestral_seq.append(seq_dict[line])
    ancestral_seq = "".join(ancestral_seq)

    def compute_diff(seq):
        return "{" + ", ".join(
            f"{i}: {char}" for i, char in enumerate(seq) if i < len(ancestral_seq) and char != ancestral_seq[i]
        ) + "}"

    for phylo in glob.glob(f"{dir}/phylogeny-snapshot*.csv"):
        lf = pl.read_csv(phylo, infer_schema_length=0).lazy()

        lf = lf.with_columns([
            pl.lit(rep_uuid).alias("replicate_uuid"),
            pl.lit(replicate_num).alias("replicate_num"),
            pl.lit(trt_name).alias("trt_name"),
            pl.lit(ancestral_seq).alias("ancestral_sequence"),
            pl.col("origin_time").cast(pl.Int64),
            pl.col("deme").cast(pl.Int64).alias("mlsgroup_id"),
            pl.col("ancestor_list").str.strip_chars("[]").alias("ancestor_id"),
        ])

        lf = lf.with_columns([
            pl.col("sequence").map_elements(lambda seq: compute_diff(seq), return_dtype=pl.String).alias("sequence_diff")
        ])

        lf = lf.group_by("id").agg([
            pl.first("replicate_uuid"),
            pl.first("replicate_num"),
            pl.first("trt_name"),
            pl.first("ancestral_sequence"),
            pl.first("origin_time"),
            pl.first("mlsgroup_id"),
            pl.first("ancestor_id"),
            pl.first("sequence_diff")
        ])

        dfs.append(lf)

print("Concatenating dataframes...")
full_df = pl.concat(dfs)

print("Finalizing dataframe...")
final_df = (
    full_df
    .collect()
    .with_columns([
        pl.col("replicate_uuid").cast(pl.Categorical),
        pl.col("ancestral_sequence").cast(pl.Categorical)
    ])
)

print("Writing final dataframe to parquet...")
final_df.write_parquet("2025-05-15-avida.parquet")
