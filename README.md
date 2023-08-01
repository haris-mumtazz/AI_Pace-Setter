> Python version: 3.7 \
This README was last modified on __July 31st, 2023__
# AI_Pace-Setter

## Overview

An AI pace setter or object identifier using Jetson Nano. Firstly, the camera will capture photos from the webcam. Object detection of images will be done using the TensorFlow framework. If an obstacle is on the right, the screen will prompt you to turn left and vice versa.

## Table of Contents

1. [Devices Needed](#devices-needed)
2. [Architecture Design](#architecture-design)
3. [Step By Step Guide](#step-by-step-guide)
4. [Potential Problems](#potential-problems)
5. [Reference](#reference)

## Devices Needed

- Jetson Nano
- Camera
- Monitor
- Keyboard & Mouse
- Mini HDMI converter
- USB OTG cable (Micro USB to USB)
- Micro USB power source

## Architecture Design

- PyCharm (Python)

## Step By Step Guide

### 3. Raspberry Pi Environment Setup
#### Install the below packages using apt-get __before__ pip installing `requirements.txt`
```console
sudo apt-get install libatlas-base-dev
sudo apt-get install python3-opencv
sudo apt-get install python3-numpy
```
#### Install the packages listed
[requirements.txt](https://github.com/Vinci-AI-Academy/AI-pace-setter/blob/main/requirements.txt)

#### Upgrade OpenCV (for using the camera)
Upgrade using the following command:
```console
sudo apt-get install git
git clone https://github.com/opencv/opencv.git
cd opencv
git checkout `4.5.1`
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local ..
make -j4
sudo make install
```
### 4. Explanation of Code
Step 1: Build your own whitelist:  
```python
whitelist=["Bottle", "Liquid", "Ball pen", "T-shirt", "Human eye",
            "Beer", "Document", "Chopsticks", "Smartphone", "Thumb",
            "Laptop", "Person", "Mask", "Computer keyboard", "Shirt",
            "Aisle", "Coat", "Suit", "Coca-cola", "Cookie",
            "Box", "Video camera", "Napkin", "Iphone", "Cap",
            "Door", "Hat", "Mug", "Lock"]
```
You can select the labels you want and put them in the whitelist above. [Full Label File](https://github.com/Vinci-AI-Academy/AI-pace-setter/blob/main/class-descriptions.csv)

Step 2: Capture Screen from Web Cam and automatic save as jpg file 

```python
cam= cv2.VideoCapture('nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1280, height=720, format=(string)NV12, framerate=(fraction)120/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink', cv2.CAP_GSTREAMER)
```
You can reset parameters of the VideoCapture.
If you test in MacBook, then it will become
```python
cam = cv2.VideoCapture(0)
```

Step 3: Cut the image in half to determine the obstacles side
```python
    width_cutoff = width // 2
    imgL = img[:, :width_cutoff]
    imgR = img[:, width_cutoff:]
    cv2.imwrite('imgL.jpg', imgL)
    cv2.imwrite('imgR.jpg', imgR)
    file_nameL = os.path.abspath('imgL.jpg')
    file_nameR = os.path.abspath('imgR.jpg')
```

Step 4: Object Detection of Image From Vision API by Left and Right side
```python
#Left side
 with io.open(file_nameL, 'rb') as image_file:
        contentL = image_file.read()

    imageL = vision.Image(content=contentL)

    response = client.label_detection(image=imageL)
    labels = response.label_annotations
```

```python
#Right side
    with io.open(file_nameR, 'rb') as image_file:
        contentR = image_file.read()

    imageR = vision.Image(content=contentR)

    response = client.label_detection(image=imageR)
    labels = response.label_annotations
```   
Step 5: The total_label will change if fulfilling these two requirements:
- Label score of more than 0.7
- In the white List

L is True if the label appears in left side.
R is True if the label appears in right side.
```python
#Left
    for label in labels:
        # print(label.description)
        if label.score > 0.7 and label.description in whitelist:
            word = translate_text("zh-TW", label.description)
            total_label = 'be careful, turn right '
            L = True
```

```python
#Right
    for label in labels:
        # print(label.description)
        if label.score > 0.7 and label.description in whitelist:
            word = translate_text("zh-TW", label.description)
            total_label = 'be careful, turn left '
            R = True
```

If both L and R is True then the following code will be run:

```python
    if (L is True and R is True):
        total_label = 'be careful, keep go forward'

```

Step 6: Generate to mp3 and play it

```python
    client2 = texttospeech.TextToSpeechClient()
    # Build the voice request, select the language code ("en-US") and the ssml
    # voice gender ("neutral")
    synthesis_input = texttospeech.SynthesisInput(text=total_label)

    voice = texttospeech.VoiceSelectionParams(
        language_code="yue-HK", ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )
    # Select the type of audio file you want returned
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3, speaking_rate=0.7
    )

    response = client2.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    with open("output.mp3", "wb") as out:
        # Write the response to the output file.
        out.write(response.audio_content)
        # print('Audio content written to file "output.mp3"')
    # play the sound of output.mp3
    playsound.playsound(os.path.abspath("output.mp3"))
```

Step 7: Using thread for multitasking both voice and Detection function
```python
 voice_thread = threading.Thread(target=voiceFun, args=(label,))
 voice_thread.start()
 voice_thread.join()
```
## Potential Problems
- Cloud Vision may detect unrelated items, such as liquid or gas. For solving this problem, we should create a whitelist for detection. Only items in the whitelist will be generated.
- playsound.playsound may not work in jetson nano

## Reference
- [Vision API](https://cloud.google.com/vision/docs/libraries)
- [Translation API](https://codelabs.developers.google.com/codelabs/cloud-translation-python3#0)
- [Text-to-speech API](https://cloud.google.com/text-to-speech/docs/libraries)
- [Language of Translation API](https://cloud.google.com/translate/docs/languages)
- Text-to-Speech supports configuring the speaking rate, pitch, volume, and sample rate hertz:  [https://cloud.google.com/text-to-speech/docs/reference/rest/v1/text/synthesize#audioconfig](https://cloud.google.com/text-to-speech/docs/reference/rest/v1/text/synthesize#audioconfig)

