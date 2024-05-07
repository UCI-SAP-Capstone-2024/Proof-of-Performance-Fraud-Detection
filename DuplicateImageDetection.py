import redis
import imagehash
from PIL import Image

def check_image_exists(image, redis_host = 'localhost', redis_port = 6379, redis_db = 0):
    r = redis.Redis(host=redis_host, port=redis_port, db=redis_db)
    hex_value = image_to_hex(image)
    if r.exists(hex_value):
        return True
    else:
        r.set(hex_value, "image")
        pass

def image_to_hex(image):
    # img = Image.open(image)
    return str(imagehash.average_hash(image))
