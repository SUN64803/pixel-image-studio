import sys, torch
import torch.nn.functional as F
from safetensors import safe_open

cos, rel = [], []
with safe_open(sys.argv[1], "pt") as fa, safe_open(sys.argv[2], "pt") as fb:
    kb = set(fb.keys())
    for k in fa.keys():
        if k.endswith("lora_A.weight") and k in kb:
            kB = k.replace("lora_A", "lora_B")
            x = (fa.get_tensor(kB).float() @ fa.get_tensor(k).float()).flatten()
            y = (fb.get_tensor(kB).float() @ fb.get_tensor(k).float()).flatten()
            cos.append(F.cosine_similarity(x, y, dim=0).item())
            rel.append(((y - x).norm() / (y.norm() + 1e-8)).item())
print("โมดูลที่เทียบ:", len(cos))
print("cosine เฉลี่ย: %.3f | ต่ำสุด: %.3f" % (sum(cos) / len(cos), min(cos)))
print("การเปลี่ยนแปลงสัมพัทธ์ เฉลี่ย: %.3f" % (sum(rel) / len(rel)))
