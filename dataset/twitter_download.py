# -*- coding: utf-8 -*-
from __future__ import print_function
import commandr
import itertools
import json
import os
import re
import requests
import socket
import time
import threading
from bs4 import BeautifulSoup
from Queue import Queue, Empty
from requests import RequestException

socket.setdefaulttimeout(10)


def extract_tweet(tweet_id):
    url = 'http://twitter.com/intent/retweet?tweet_id={}'.format(tweet_id)
    try:
        response = requests.get(url)
    except RequestException:
        raise

    if response.status_code != 200:
        return 'NOT AVAILABLE'

    html = response.text.replace('</html>', '') + '</html>'
    soup = BeautifulSoup(html, features='html5lib')
    jstt = soup.find_all("div", "tweet-text")
    tweets = list(set([_x.get_text() for _x in jstt]))

    if len(tweets) == 0:
        return 'NOT AVAILABLE'

    if len(tweets) > 1:
        raise Exception(tweet_id)

    text = re.sub(r'\s+', ' ', tweets[0]).strip()
    return text


@commandr.command
def test(tweet_id):
    print(extract_tweet(tweet_id))


@commandr.command
def main(n_thread, input_filename, output_filename):
    n_thread = int(n_thread)

    class SharedData(object):
        def __init__(self):
            self.exit_signal = False
            self.queue = Queue()
            self.total_count = itertools.count()
            self.fail_count = itertools.count()
            self.data = dict()

    shared = SharedData()

    if os.path.exists(output_filename):
        shared.data = json.loads(open(output_filename))

    def run(number):
        while not (shared.exit_signal and shared.queue.empty()):
            try:
                tweet_id = shared.queue.get_nowait()
            except Empty:
                time.sleep(1)
                continue

            if tweet_id in shared.data:
                continue

            tweet = extract_tweet(tweet_id)
            if tweet is not None:
                shared.data[tweet_id] = tweet

            last_total_count = shared.total_count.next()
            print('{} downloaded'.format(last_total_count))
        print('thread exit:{}'.format(number))

    # 初始化各个线程
    thread_list = []
    for i in range(n_thread):
        thread = threading.Thread(target=run, args=(i,))
        thread_list.append(thread)
        thread.setDaemon(True)
        thread.start()

    with open(input_filename, 'r') as file_obj:
        for line in file_obj:
            line = line.strip()
            if line == '':
                continue
            parts = line.split(',')
            shared.queue.put(parts[0].strip())

    shared.exit_signal = True  # 内容已全部加入队列中

    for i in range(n_thread):
        thread_list[i].join()

    json.dump(shared.data, open(output_filename + '.new', 'w'))


if __name__ == '__main__':
    commandr.Run()
