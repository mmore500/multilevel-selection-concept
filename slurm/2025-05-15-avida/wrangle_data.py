import glob
import os
import sys
import uuid
import polars as pl
pl.enable_string_cache()
pl.Config.set_streaming_chunk_size(500)

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

j = 200

for dir in dirs[j:]:
    if os.path.exists(f"intermediate_{j}.parquet"):
        j+=1
        continue

    print(dir, flush=True)
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

    for k, phylo in enumerate(glob.glob(f"{dir}/phylogeny-snapshot*.csv")):
        # sequence_diff_expr = (
        #     # pl.col("sequence").str.split("").
        #     ("{" +
        #         # pl.concat_list([
        #         #     pl.col("sequence").str.split(""),
        #         #     pl.col("ancestral_sequence").str.split("")
        #         # ])
        #         pl.col("sequence").str.split("").eval(
        #             pl.when(pl.element() != a.list.get(pl.int_range(pl.len())))
        #             .then(
        #                 pl.format(
        #                     "{}: {}",
        #                     pl.int_range(pl.len()),
        #                     pl.element()
        #                 )
        #             )
        #             .otherwise(None)
        #         )
        #         .list.drop_nulls()
        #         .list.join(", ").cast(pl.String)
        #         + '}').alias("sequence_diff")
        # )

        lf = pl.scan_csv(phylo, infer_schema_length=0).select([
            pl.lit(rep_uuid).alias("replicate_uuid").cast(pl.Categorical),
            pl.lit(replicate_num).alias("replicate_num"),
            pl.lit(trt_name).alias("trt_name"),
            pl.lit(ancestral_seq).alias("ancestral_sequence").cast(pl.Categorical),
            pl.col("origin_time").cast(pl.Int64),
            pl.col("destruction_time").cast(pl.Int64),
            pl.col("id").cast(pl.Int64),
            pl.col("deme").cast(pl.Int64).alias("mlsgroup_id"),
            pl.col("sequence").map_elements(compute_diff, return_dtype=pl.String).alias("sequence_diff"),
            pl.col("ancestor_list").str.strip_chars("[]").alias("ancestor_id")
        ])
        lf.sink_parquet(f"very_intermediate_{j}_{k}.parquet", engine="streaming", maintain_order=False)

    intermediate_parts = pl.scan_parquet(f"very_intermediate_{j}_*.parquet").unique()
    intermediate_parts.sink_parquet(f"intermediate_{j}.parquet", engine="streaming", maintain_order=False)
    j += 1


# print(full_df.explain(engine="streaming"))

all_parts = pl.scan_parquet("intermediate_*.parquet")
all_parts.sink_parquet(sys.argv[2], engine="streaming", maintain_order=False)

# print("Finalizing dataframe...")
# final_df = (
#     full_df
#     .collect(streaming=True)
# )

# print("Writing final dataframe to parquet...")
# final_df.write_parquet("2025-05-15-avida.parquet")
