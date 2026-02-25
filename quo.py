import numpy as np

m = np.load("emb_mean.npy")
s = np.load("emb_std.npy")

print("std min:", s.min())
print("std max:", s.max())
print("std mean:", s.mean())