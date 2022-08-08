# Frontex Captcha Solver

This repository contains the code for training a neural network for solving captchas.
It was developed to solve captchas created with [Jeff Atwoods ASP.NET Captcha Generation Library](https://www.codeproject.com/Articles/8751/A-CAPTCHA-Server-Control-for-ASP-NET), which Frontex uses on their [PAD portal](https://pad.frontex.europa.eu/Token/Create).

While it was developed for a specific captcha, it should work well for other captchas too.

## Installation

This project uses Pillow for image processing and pytorch for it's machine learning internals.
You can install all dependencies using

```
pip install -r requirements.txt
```

## Training

### Preparation

For training, you need to solve a few captchas by hand.
We had acceptable results with ~100 and good with ~400 manually solved captchas.
Store them all in one directory (we assume it is named `input/` in the following text) with the captcha text as their filename stem.
If for example the text in the image is `ABC123` and it is a JPEG file, store it as `input/ABC123.jpg` 

#### Using prodigy

If you have a [prodigy](https://prodi.gy/) license and a collection of downloaded captcha images, you can use the recipes in `prodigy_recipes.py`:

```
prodigy image-caption -F prodigy_recipes.py DATASET_NAME DIRECTORY_WITH_CAPTCHAS_TO_CAPTION
# Then after tagging
prodigy write-images -F prodigy_recipes.py DATASET_NAME input/
```

### Model Training

Now that you have your input ready, you can start the training.
First, check that the settings in the model.py are correct:
`CLASSES` should be all possible characters in the captchas.
`LETTER_COUNT` should be the number of characters per captcha.

If those setting are correct, simply run

```
python train.py input/ output/
```

This will train the model on the input data.
After a few minutes it will stop and write the trained model to `output/model.pth`.

It will also save the internal state of the optimizer to `output/optimizer.pth`.
This can be used to resume training the model.
To do that add the `--resume-training` flag to the command line above.

### Evaluating the model

During training the script will continously output the loss and accuracy of the model.

To try it on a single image, you can use the `run_on_image.py`-script:

```
python run_on_image.py output/model.pth PATH_TO_IMAGE
```

## Usage

To use the model, you can either shell-out to `run_on_image.py` or use the helper functions in captcha.py.

You need to load the model using `load_net` and then run `solve_image` on it:

```python
>>> import captcha_model
>>> from PIL import Image
>>> net = captcha_model.load_net("output/model.pth")
>>> image = Image.open("PATH_TO_IMAGE")
>>> result = captcha_model.solve_image(net, image)
```
