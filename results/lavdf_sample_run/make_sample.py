import pandas as pd, numpy as np
m = pd.read_csv("metadata/lavdf_metadata.csv")
print("original column examples:", m.loc[m.label == 1, "original"].head(3).tolist())
parts = []
for split, n in (("train", 3000), ("dev", 600), ("test", 1000)):
    parts.append(m[m.split == split].sample(n, random_state=42))
sample = pd.concat(parts)
sample.to_csv("results/lavdf_sample_run/sample.csv", index=False)
print(pd.crosstab(sample.split, [sample.modify_video, sample.modify_audio]))
test = sample[sample.split == "test"]
fakes = test[test.label == 1]
picked = fakes.groupby(["modify_video", "modify_audio"], group_keys=False).apply(lambda g: g.sample(min(150, len(g)), random_state=42))
originals = m[m.file.isin(picked.original.dropna().unique())]
extra = originals[~originals.file.isin(sample.file)]
extra.to_csv("results/lavdf_sample_run/cache_extra.csv", index=False)
print("swap candidates:", len(picked), "| extra source clips to cache:", len(extra), "| splits of sources:", originals.split.value_counts().to_dict())
