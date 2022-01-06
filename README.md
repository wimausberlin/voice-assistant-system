# _"Hey Alfred"_ : Wake Word Detection
![Versions : 3.6 ,3.7,3.8,3.9](https://camo.githubusercontent.com/a7b5b417de938c1faf3602c7f48f26fde8761a977be85390fd6c0d191e210ba8/68747470733a2f2f696d672e736869656c64732e696f2f707970692f707976657273696f6e732f74656e736f72666c6f772e7376673f7374796c653d706c6173746963)

## Table of content
* [Wakeword detection](#wakeword-detection-based-on-a-CNN)
* [Demo](#demo)
* [Custom wakeword](#generating-custom-wakewords)

## Wakeword detection based on a CNN
Intelligent voice assistant systems, such as smartphone assistants (e.g., Siri, Cortana, Google Now), Amazon Echo, and Google Home are becoming pervasive in our daily life. These human–machine communication systems are still emerging, mainly due to large researches in Deep Learning. Creating a personal voice assistant system improves the interaction with ALFRED, the xArm robot. This entire project consists of implementing all the voice assistant from the Automatic Speech Recognition (ASR) to Text-to-Speech (TTS) through Wake Word Detection. In this paper, we are focusing on the Wake Word Detection and consists of classifing audio files in a binary way to detect if a specific word is characterized. With the help of Deep Learning (CNN), we are building a Binary Classifier by taking an audio as input and expecting a Boolean as output. 

More details can be found on the [report](https://github.com/wimausberlin/voice-assistant-system/blob/main/docs/report.pdf).

## Installation
This project works on python version: `3.6 and more`.

### Dependencies Installation
Before running the pip installation command for the project, few dependencies need to be installed manually:
* [PyAudio](https://pypi.org/project/PyAudio/)
* [Librosa](https://librosa.org/doc/latest/install.html)
* [Matplotlib](https://pypi.org/project/matplotlib/)

**_librosa_** and **_matplotlib_** packages are only required for plotting audio data.

### pip packages
Command to install all the Python libraries required:
```
pip3 install -r requirements.txt
```

## Demo
After installing the packages, you can run the Demo script.

Command to run the demo:
 ```
 python3 voice_assistant/core/demo.py
 ```

## Generating Custom Wakewords
### Generating new audio samples
Before starting, there is a structure to follow. A directory, for example named `sound`, must be created, in which contains resp. `0` and `1` directories. These will serve to classify the new wakeword and the other sounds.

To generate a custom wakeword, it is necessary to create a new dataset of the new hotword. For it, run the `collect_audio_data.py` file in the `data` directory to generate audio sample:
```
python3 collect_audio --seconds 2 --samples_save_path [PATH]
```
(**PATH** looks like `/sound/0/` or `/sound/1/` )

### JSON dataset files
The current model needs json files to load the dataset. Hence, run the following command to create them:
```
python3 create_wakeword_json.py --zero_label_dir [PATH0] --one_label_dir [PATH1] --save_json_path [PATH]
```
where **PATH0** and **PATH1** are the directories containing resp. the zero and one labels. **PATH** is the directory path for saving the train and test json files.

### Train the model
The train model code have multiple arguments, such as the number of epochs, the batch size or the learning rate. By default, they are put resp. at 100, 32, 1e-3. It can also disactivate cuda:
```
python3 train_CNN.py --save_path [PATH] --train_data_json [TRAIN] --test_data_json [TEST] --no_cuda
```
where **PATH** is the location to save the train model (named `model_cnn.pth`). **TRAIN** and **TEST** are the path of the resp. json files, created just before.