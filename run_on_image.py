import captcha_model
import argparse
from pathlib import Path
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("model_path", type=Path)
parser.add_argument("image_path", type=Path)
args = parser.parse_args()

net = captcha_model.load_net(args.model_path)
image = Image.open(args.image_path)
result = captcha_model.solve_image(net, image)
print(result)
