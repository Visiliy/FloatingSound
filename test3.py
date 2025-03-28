import sqlite3


connection = sqlite3.connect('music.db')
cursor = connection.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS Music (
id INTEGER PRIMARY KEY,
name TEXT,
text TEXT,
notes TEXT
)
''')

with open("songs2.txt", "r", encoding="utf-8") as file:
    text = file.readlines()
    for i in text:
        i = i.strip("\n")
        if len(i.split("|")) == 2:
            name, music = i.split("|")
            cursor.execute('INSERT INTO Music (name, text, notes) VALUES (?, ?, ?)', (name, music, ""))


connection.commit()
connection.close()
print("Ok")