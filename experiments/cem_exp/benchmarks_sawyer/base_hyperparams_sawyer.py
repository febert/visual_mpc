""" Base Hyperparameters for benchmarks"""

config = {
}

policy= {
}

IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
IMAGE_CHANNELS = 6 # 2x RGB

agent = {
    'image_height' : IMAGE_HEIGHT,
    'image_width' : IMAGE_WIDTH,
    'image_channels' : IMAGE_CHANNELS,
    'num_objects': 1,
}

