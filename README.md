# GPS-Denied UAV Localization using Pre-existing Satellite Imagery

This is a repo based on the code for the paper GPS-Denied UAV Localization using Pre-existing Satellite Imagery.

## Setup

Pre-requisites: 
1. Docker has to be installed in the machine.
2. Docker has to be configured to allow containers the use of enough RAM for training to work (16 GB min maybe?).

Building and Setup:
1. Run `bash build.sh` to create the container image.
2. Create a subfolder in the repo called `data` .
2. Download dataset folders from [this Google Drive](https://drive.google.com/drive/folders/1sscpYCZXCRUWKl9eUDQGz-DZQLo3HeDe?usp=sharing) and copy folders to the `data` subfolder in this repo.

## Train
1. Execute the following command (see evaluate.py for example parameters):
	- `bash run_container.sh run_local_evaluate.sh train SAT_PATH MODEL_PATH VGG_MODEL_PATH`

Example: fine-tune VGG16 conv3 block with New Jersey dataset ('woodbridge'):
	- `bash run_container.sh run_local_evaluate.sh ./data/sat_data/woodbridge trained_model_output.pth ./data/models/vgg16_model.pth`

## Test
1. Execute the following command (see evaluate.py for example parameters):
	- `bash run_container.sh run_local_evaluate.sh test SAT_PATH MODEL_PATH VGG_MODEL_PATH --TEST_DATA_SAVE_PATH`

Example: fine-tune VGG16 conv3 block with New Jersey dataset ('woodbridge'):
	- `bash run_container.sh run_local_evaluate.sh test ./data/sat_data/woodbridge ./data/models/conv_03_13_18_1850.pth ./data/models/vgg16_model.pth -t test_params.txt`

See `argparse` help for argument documentation.

## Cropping

Note that GeoTIFF image cropping preserving coordinates can be done with:

 - gdal_translate -srcwin x0 y0 x1 y1 source.tif cropped.tif
