from bs4 import BeautifulSoup as soup
import requests
res = requests.get('https://en.wikipedia.org/wiki/Deepika_Padukone')
wiki = soup(res.text, 'html.parser')
f = open('DP.txt', 'w+')
for i in wiki.select('p'):
    f.write(i.getText())
f.close()
