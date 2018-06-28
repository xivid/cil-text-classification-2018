files = [
        "kaggle_20180626142259_accu0.876900.csv",
        "kaggle_20180626142516_accu0.877500.csv",
        "kaggle_20180626144416_accu0.879700.csv",
        "kaggle_20180626144852_accu0.880700.csv",
        "kaggle_20180626145840_accu0.880700.csv",
        "kaggle_20180626151509_accu0.881300.csv"
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

