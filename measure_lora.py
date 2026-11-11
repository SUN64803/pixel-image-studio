import sys
from safetensors import safe_open

for path in sys.argv[1:]:
    with safe_open(path, "pt") as f:
        norms = []
        for k in f.keys():
            if k.endswith("lora_A.weight"):
                A = f.get_tensor(k).float()
                B = f.get_tensor(k.replace("lora_A", "lora_B")).float()
                norms.append((B @ A).norm().item())
    print(path)
    print("  โมดูล:", len(norms),
          "| ‖BA‖ เฉลี่ย: %.3f | สูงสุด: %.3f" % (sum(norms) / len(norms), max(norms)))
