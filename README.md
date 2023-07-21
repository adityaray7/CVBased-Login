# Face Recognition System

## Summary
This project is a face recognition system that uses the MTCNN and InceptionResnetV1 models to detect and recognize faces from images or webcam feed. The system matches detected faces with a pre-trained database of known faces and labels them accordingly. If the detected face matches an authorized user's face, it allows access to certain functionalities.

##Methodology
The face recognition system utilizes a combination of two powerful deep learning models:

1. Multi-Task Cascaded Convolutional Networks (MTCNN):
MTCNN is a popular deep learning model used for face detection. It is capable of detecting faces in an image and providing bounding box coordinates for each detected face. The model is designed to handle faces at different scales and orientations, making it robust for real-world scenarios.

2. InceptionResnetV1:
InceptionResnetV1 is a pre-trained deep learning model that is widely used for face recognition tasks. It can transform facial images into a compact representation known as an embedding vector. These embeddings encode the unique features of each face, making it easier to compare and recognize faces from a database.
Workflow:

3. Data Preparation:
The system requires a database of pre-registered faces for recognition. Users' images should be organized in separate folders inside the 'photos' directory, with each folder named after the corresponding user.
Data Loading and Embedding Extraction:

The system loads the pre-registered faces and their corresponding labels using the datasets.ImageFolder from PyTorch. It then applies MTCNN to detect and crop the faces from the images. Next, the cropped faces are passed through the InceptionResnetV1 model to obtain embedding vectors for each face. These embeddings serve as a reference for face recognition.

4. Real-time Face Recognition:
Using the webcam feed, the system captures frames and applies MTCNN for face detection. For each detected face, the system calculates the embedding vector using InceptionResnetV1. It then compares the embedding of the detected face with the embeddings from the database to find the closest match.

5. Authentication and Access Control:
If the detected face matches an authorized user's face (based on a pre-defined threshold), the system displays the user's name and confidence level on the webcam feed. It also grants access to specific functionalities, such as launching a web browser to access a specific URL.
Unauthorized User Handling:

If the detected face does not match any authorized user's face or the confidence level is below the threshold, the system indicates that the user is unauthorized. The system continues to monitor for authorized users.
