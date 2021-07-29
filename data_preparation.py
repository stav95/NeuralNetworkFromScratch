from zipfile import ZipFile

FILE = 'fashion_mnist_images.zip'
FOLDER = 'fashion_mnist_images'

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
