import requests
import time
import os

API_KEY = 'adasC3U9HmNXsTWSNWHBBDYEVb90yg1XBIbhLcWBQdNLx03BKc8Mbhy6'

headers = {
    'Authorization': API_KEY
}

params = {
    'query': 'cat',
    'per_page': 20,
}

# 指定图片保存路径
img_dir = './cats_and_dogs_train/cats'

if not os.path.exists(img_dir):
    os.makedirs(img_dir)

for page in range(1, 10):  # 获取前三页
    params['page'] = page
    response = requests.get('https://api.pexels.com/v1/search', headers=headers, params=params)

    data = response.json()
    for i, photo in enumerate(data['photos']):
        img_url = photo['src']['original']
        img_data = requests.get(img_url).content  # 获取图片数据
        img_name = f"{params['query']}_{page}_{i}.jpg"  # 图片文件名使用关键词、页数和图片序号
        with open(os.path.join(img_dir, img_name), 'wb') as f:  # 以二进制格式写入文件
            f.write(img_data)

    time.sleep(1)  # 在每次请求之间暂停1秒
