# Face Recognition System

## Summary
This project is a face recognition system that uses the MTCNN and InceptionResnetV1 models to detect and recognize faces from images or webcam feed. The system matches detected faces with a pre-trained database of known faces and labels them accordingly. If the detected face matches an authorized user's face, it allows access to certain functionalities.

## Methodology
The face recognition system utilizes a combination of two powerful deep learning models:

1. Multi-Task Cascaded Convolutional Networks (MTCNN):
MTCNN is a popular deep learning model used for face detection. It is capable of detecting faces in an image and providing bounding box coordinates for each detected face. The model is designed to handle faces at different scales and orientations, making it robust for real-world scenarios.

2. InceptionResnetV1:
InceptionResnetV1 is a pre-trained deep learning model that is widely used for face recognition tasks. It can transform facial images into a compact representation known as an embedding vector. These embeddings encode the unique features of each face, making it easier to compare and recognize faces from a database.
Workflow:

3. Data Preparation:
The system requires a database of pre-registered faces for recognition. Users' images should be organized in separate folders inside the 'photos' directory, with each folder named after the corresponding user.

4. Data Loading and Embedding Extraction:
The system loads the pre-registered faces and their corresponding labels using the datasets.ImageFolder from PyTorch. It then applies MTCNN to detect and crop the faces from the images. Next, the cropped faces are passed through the InceptionResnetV1 model to obtain embedding vectors for each face. These embeddings serve as a reference for face recognition.

5. Real-time Face Recognition:
Using the webcam feed, the system captures frames and applies MTCNN for face detection. For each detected face, the system calculates the embedding vector using InceptionResnetV1. It then compares the embedding of the detected face with the embeddings from the database to find the closest match.

6. Authentication and Access Control:
If the detected face matches an authorized user's face (based on a pre-defined threshold), the system displays the user's name and confidence level on the webcam feed. It also grants access to specific functionalities, such as launching a web browser to access a specific URL.

7. Unauthorized User Handling:
If the detected face does not match any authorized user's face or the confidence level is below the threshold, the system indicates that the user is unauthorized. The system continues to monitor for authorized users.

# Getting Started
To get the face recognition system up and running on your local machine, follow these steps:

## Installation
Make sure you have Python 3.x installed. Then, install the required dependencies using the following command:

```
pip install opencv-python
pip install facenet-pytorch
pip install torch
pip install torchvision
pip install selenium
pip install geckodriver-autoinstaller
```

You can install these libraries by running the pip install command as shown above. This will ensure that all the necessary dependencies are installed on your system.
Please note that the geckodriver-autoinstaller library is used to automatically install the latest geckodriver for Firefox. If you prefer to manually install geckodriver, you can use the commented-out line in your code:

`# browser = webdriver.Firefox(executable_path='/home/aditya/geckodriver')`
Replace '/home/aditya/geckodriver' with the path to your geckodriver executable.

# Configuration
No specific configuration is required for this project.

# Data
The system uses a database of pre-registered faces for recognition. Place the images of authorized users in the 'photos' folder. Each user's images should be placed in a separate folder named after the user.

# How to Use
Follow these steps to use the face recognition system:

1. Clone the repository:
`git clone https://github.com/your_username/face-recognition.git`
2. Change directory to the project folder:
`cd face-recognition`
3. Place the images of authorized users in the 'photos' folder. Create a separate folder for each user and put their images inside.

4. Run the face detection script:
`python face_detection.py`

5. The webcam feed will open, and the system will start recognizing faces. If the detected face matches an authorized user's face, it will display the name and confidence level on the webcam feed. It will also grant access to specific functionalities, such as launching a web browser to access a specific URL.

6. If the detected face does not match any authorized user's face or the confidence level is below the threshold, the system will indicate that the user is unauthorized. The system continues to monitor for authorized users.


