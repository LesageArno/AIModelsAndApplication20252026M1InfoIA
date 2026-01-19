# AIMA-2025-Minh_Quan_HOANG_Arno_LESAGE
### Authors: Arno LESAGE (no. 22202985)

```
/!\ Important /!\
Upon the request of the team member **Minh Quan HOANG**, his name was removed from the authors of the project for personal reason.

For more information, please contact :
> minh-quan[dot]hoang[at]etu[dot]univ-cotedazur[dot]fr
> arno[dot]lesage[at]etu[dot]univ-cotedazur[dot]fr
```

## Install and run the project
### Installation
To install the project, select the repertory `src` as the working directory and ensure a Python 3.12 virtual environment with pip enabled is active.

When both the virtual environnement and the working directory are set up, you must first execute this command in order to download every needed Python libraries (and there dependencies) :

```shell
pip install -r requirements.txt
```
### Run
#### Step 1: preprocessing and face extraction
To run the project from the formerly `src` working directory and set up virtual environnement, you can run the script `RetinaExtractor.py` which will run the *Batch-Face model* to extract the faces from the `src_img` directory to the `working` and `working_coloured` directory.

#### Step 2: Train and test the models
Then, you can run the Jupyter Notebook `FinalStage.ipynb` cell-by-cell to add further processing steps, model definitions, training and tests.

#### Step 3: Load the model
To load the model without retraining them, you will find in the `src/models` directory some models and saved informations :

- `accuracyPerEpoch`: to load the evaluation and training accuracies of the Simple Grey-CNN model from 10 epochs to 210 (the first ten was not registered),
- `cnnEpochXXX+10.keras`: to load the Simple Grey-CNN after XXX+10 epochs,
- `rcmalli_vggface_labels_v1.npy`: the labels of the VGG Face Descriptor model.

**Note:** you will also need to download and put there the `rcmalli_vggface_tf_vgg16.h5` file, which contains the weights of the VGG Face Descriptor model. These weights are available <a href="https://github.com/rcmalli/keras-vggface/releases/">here</a>.

### And what about the other directories and files?
In the `rep` directory, you will find the LaTeX report of this project and its compilation as well as the `img` folder, used for containing figures of the report.

In the `others` directory, you will find two miscellaneous files corresponding to intermediary working files.

Last, but not least, the `src/First Attempt` is the deprecated first attempt to handle the project by trying to follow as strictly as possible the project instructions. It contains a `working` directory as well, but contrary to the preceding one, it should contains the results of extracted person after the usage of the `YOLO11n.pt`. 

You will also find two Python scripts corresponding to the preprocessing steps and a Jupyter Notebook for the last step. 


