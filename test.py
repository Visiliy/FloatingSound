from bs4 import BeautifulSoup
import requests

url = 'https://text-pesni.ru/pop/page/'
cnt = 1
for i in range(1, 159) 
    response = requests.get(url + str(i) + "/")
    soup = BeautifulSoup(response.text, 'html.parser')
    div = soup.find_all("div", class_="track-item")
    for j in div:
        name = j.find("span").text 
        if name is not None and name != "":
            with open(f"audio1/name{cnt}.txt", "w", encoding="utf-8") as f1:
                f1.write(name.split("|")[0])
            href = j.find("a", class_="track-desc").get("href")
            res = requests.get(href)
            s = BeautifulSoup(res.text, 'html.parser')
            text = s.find("div", class_="sect-content").text
            with open(f"audio2/text{cnt}.txt", "w", encoding="utf-8") as f2:
                f2.write(text)

            cnt += 1
print("Ok")