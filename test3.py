s = "йцукенгшщзхъфывапролджэячсмитьбю"
ar = {}
with open("songs.txt", "r", encoding="utf-8") as f:
    text = f.readlines()
    for i in text:
        new_i = i.split("|")[-1]
        cnt = 0
        for j in range(100):
            if new_i[j] not in s:
                cnt += 1
        if cnt < 30:
            with open("songs2.txt", "a", encoding="utf-8") as f2:
                f2.write(i)
print("Ok")