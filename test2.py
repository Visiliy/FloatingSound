for i in range(1, 4956):
    with open(f"audio1/name{i}.txt", "r", encoding="utf-8") as f:
        name = f.read().strip("\n").split(",")
    with open(f"audio1/name{i}.txt", "w", encoding="utf-8") as f2:
        f2.write(name[0])
print("Ok")