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

## Train/Test
1. Execute the following command (see evaluate.py for example parameters):
	- `bash run.sh MODE FOLDER_NAME DATAPATH MODEL_PATH VGG_MODEL_PATH --TEST_DATA_SAVE_PATH`

Example: fine-tune VGG16 conv3 block with New Jersey dataset ('woodbridge'):
	- `bash run.sh train woodbridge ./data/sat_data/ trained_model_output.pth ./data/models/vgg16_model.pth`

See `argparse` help for argument documentation.
