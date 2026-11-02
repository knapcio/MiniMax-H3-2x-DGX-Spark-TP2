import sys
y = [float(v) for v in sys.stdin.read().split()]
st = sorted(y[k] - y[k - 1] for k in range(17, len(y), 17))
print(f"frames {len(y)} boundary step median {st[len(st)//2]:.2f} min {st[0]:.2f}")
for a, b in ((17, 277), (289, 532), (544, len(y))):
    s = sorted(y[k] - y[k - 1] for k in range(a, b, 17))
    if s:
        print(f"  frames {a}-{b}: median {s[len(s)//2]:.2f} min {s[0]:.2f}")
