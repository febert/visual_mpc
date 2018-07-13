import numpy as np


def draw_text_image(text, background_color=(255,255,255), image_size=(30, 64), dtype=np.float32):

    from PIL import Image, ImageDraw
    text_image = Image.new('RGB', image_size[::-1], background_color)
    draw = ImageDraw.Draw(text_image)
    if text:
        draw.text((4, 0), text, fill=(0, 0, 0))
    if dtype == np.float32:
        return np.array(text_image).astype(np.float32)/255.
    else:
        return np.array(text_image)


def draw_text_onimage(text, image, color=(255, 0, 0)):
    if image.dtype == np.float32:
        image = (image*255.).astype(np.uint8)
    assert image.dtype == np.uint8
    from PIL import Image, ImageDraw
    text_image = Image.fromarray(image)
    draw = ImageDraw.Draw(text_image)
    draw.text((4, 0), text, fill=color)
    return np.array(text_image).astype(np.float32)/255.
