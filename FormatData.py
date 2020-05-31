import json

jsonFile = open("data/data.json", 'r', encoding="utf8")

fakeNews = {}
data = []
for line in jsonFile:
    data.append(line)

fakeNews["fakeNews"] = data

with open('data/formattedData.json', 'w') as outfile:
    json.dump(fakeNews, outfile)
