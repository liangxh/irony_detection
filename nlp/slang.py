# -*- coding: utf-8 -*-
import re
import requests
from bs4 import BeautifulSoup


class Google(object):
    @classmethod
    def search(cls, token):
        url = 'https://www.google.com/search?q=twitter+{}'.format(token)
        res = requests.get(url)
        soup = BeautifulSoup(res.text, 'html.parser')
        res = soup.find_all('a', attrs={'href': lambda _v: _v.startswith('/search')})

        suggested = None
        for r_ in res:
            if r_.text.startswith('twitter') and r_.find('em') is not None:
                suggested = re.sub('^twitter ', '', r_.text).strip()
                break
        return suggested


if __name__ == '__main__':
    pass
