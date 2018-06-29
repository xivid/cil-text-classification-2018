files = [
        "kaggle_20180628155839_accu0.881500.csv",
        "kaggle_20180628160635_accu0.882400.csv",
        "kaggle_20180628161539_accu0.882800.csv",
        "kaggle_20180628161907_accu0.883300.csv",
        "kaggle_20180628162812_accu0.886100.csv",
        "kaggle_20180628164723_accu0.887600.csv"
        ]

scores = [0] * 10001
best = [0] * 10001
for f in files:
    with open(f) as fin:
        for line in fin:
            idx, val = (lambda x: (int(x[0]), int(x[1])))(line.split(','))
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

