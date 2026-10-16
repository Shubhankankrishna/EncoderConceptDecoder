"""Change the cosine length stored in runs/main/ckpt_last.pt (D-033). os.replace breaks the hard link, so the
ckpt_stepNNNNNN.pt file stays untouched as the fallback."""
import os, sys, torch
p, n = sys.argv[1], int(sys.argv[2])
ck = torch.load(p, map_location="cpu", weights_only=False)
print("before:", ck["step"], ck["total_steps"], ck["args"]["total_steps"], ck["lr_scale"], ck["micro_bs"], ck["accum"], ck["cursor"])
ck["total_steps"] = n; ck["args"]["total_steps"] = n
torch.save(ck, p + ".tmp"); os.replace(p + ".tmp", p)
ck = torch.load(p, map_location="cpu", weights_only=False)
print("after:", ck["step"], ck["total_steps"], ck["args"]["total_steps"])
