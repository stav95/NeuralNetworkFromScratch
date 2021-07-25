import os
import urllib.request
from zipfile import ZipFile

URL = 'https://nnfs.io/datasets/fashion_mnist_images.zip'
FILE = 'fashion_mnist_images.zip'
FOLDER = 'fashion_mnist_images'

if not os.path.isfile(FILE):
    print(f'Downloading {URL} and saving as {FILE}...')
    urllib.request.urlretrieve(URL, FILE)

print('Unzipping images...')
with ZipFile(FILE) as zip_images:
    zip_images.extractall(FOLDER)

# region fashion_mnist_images
# Label	Description
# 0	    T-shirt/top
# 1   	Trouser
# 2	    Pullover
# 3   	Dress
# 4	    Coat
# 5   	Sandal
# 6	    Shirt
# 7   	Sneaker
# 8   	Bag
# 9	    Ankle boot
# endregion
