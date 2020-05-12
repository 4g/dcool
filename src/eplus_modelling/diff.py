import sys
def get_pues(fname):
    y = [x.strip().split(",")[68] for x in open(fname)][1:]
    y = [float(x) for x in y]
    return y

avg = lambda x : sum(x) / len(x)


pue1 = get_pues(sys.argv[1])
pue2 = get_pues(sys.argv[2])

diffs = zip(pue1, pue2)
diffs = map(lambda x:x[1] - x[0], diffs)
diffs = list(diffs)
print(sum(diffs))
# print(diffs)
print(avg(pue1), avg(pue2))
