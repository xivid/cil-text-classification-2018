files = [
        "kaggle_20180628202627_accu0.882000.csv",
        "kaggle_20180628202815_accu0.883800.csv",
        "kaggle_20180628220118_accu0.884100.csv",
        "kaggle_20180628220450_accu0.884600.csv"
        ]

scores = [0] * 10001
best = [0] * 10001
for f in files:
    with open(f) as fin:
        for line in fin:
            try:
                idx, val = (lambda x: (int(x[0]), int(x[1])))(line.split(','))
            except:
                print("ignoring line `{}` from {}".format(line.strip(), f))
                continue
            scores[idx] += val
            if f == files[-1]:
                best[idx] = val

count = 0
for i in range(1, 10001):
    if scores[i] > 0:
        scores[i] = 1
    elif scores[i] < 0:
        scores[i] = -1
    else:
        scores[i] = best[i]

with open("final.csv", "w+") as out:
    out.write("Id,Prediction\n")
    for i in range(1, 10001):
        out.write("{},{}\n".format(i, scores[i]))

