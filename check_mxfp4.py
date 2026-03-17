import struct, json, numpy as np

# Load first shard
path = "models/gpt-oss-20b/model-00000-of-00002.safetensors"
with open(path, 'rb') as f:
    n = struct.unpack('<Q', f.read(8))[0]
    meta = json.loads(f.read(n))

header_size = 8 + n

# Look at shapes for layer 0 experts
for k in sorted(meta.keys()):
    if k == '__metadata__':
        continue
    if 'layers.0.mlp.experts' in k:
        v = meta[k]
        print(f"{k}: dtype={v['dtype']}, shape={v['shape']}, offsets={v['data_offsets']}")

# Also check the blocks tensor format - the shape is [32, 5760, 90, 16]
# The 4th dimension (16) is 16 bytes = 32 nibbles = 32 FP4 values = 1 block
# The 3rd dimension (90) is 90 blocks per row
# So the layout is: [expert, row, block_idx, 16_packed_bytes]
# This means within each block of 32 values, they're stored as 16 bytes
# and the scale for that block is at scales[expert, row, block_idx]

# Let's read a few bytes from the blocks and scales to verify
key = "model.layers.0.mlp.experts.gate_up_proj_blocks"
info = meta[key]
start, end = info['data_offsets']
with open(path, 'rb') as f:
    f.seek(header_size + start)
    data = f.read(min(end - start, 1024))

# Print first few bytes of expert 0, row 0
print(f"\nFirst 32 bytes of blocks (expert 0, row 0, blocks 0-1):")
print(' '.join(f'{b:02x}' for b in data[:32]))

# Read scales
key = "model.layers.0.mlp.experts.gate_up_proj_scales"
info = meta[key]
start, end = info['data_offsets']
with open(path, 'rb') as f:
    f.seek(header_size + start)
    sdata = f.read(min(end - start, 256))

print(f"\nFirst 16 scale bytes (expert 0, row 0, first 16 scale blocks):")
print(' '.join(f'{b:02x}' for b in sdata[:16]))
print("As E8M0 values:", [2.0**((b-127)) if b > 0 and b < 255 else (0.0 if b == 0 else float('nan')) for b in sdata[:16]])
