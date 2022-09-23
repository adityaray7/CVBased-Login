import cv2
import time
import os
from facenet_pytorch import MTCNN,InceptionResnetV1
from PIL import Image
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
from webdriver_manager.firefox import GeckoDriverManager

from user import userauth,id,password

def browse(usrname,passwd):

    browser = webdriver.Firefox(executable_path=GeckoDriverManager().install())

    #browser = webdriver.Firefox(executable_path='/home/aditya/geckodriver')
    options = webdriver.ChromeOptions()
    options.add_experimental_option("detach", True)


    browser.get('https://students.iitmandi.ac.in/moodle/login/index.php')
    browser.implicitly_wait(2)


    browser.find_element(By.ID,'username').send_keys(usrname)
    browser.find_element(By.ID,'password').send_keys(passwd)
    browser.find_element(By.ID,'loginbtn').submit()


    browser.implicitly_wait(2)



# initializing MTCNN and InceptionResnetV1 

mtcnn0 = MTCNN(image_size=240, margin=0, keep_all=False, min_face_size=40) # keep_all=False
mtcnn = MTCNN(image_size=240, margin=0, keep_all=True, min_face_size=40) # keep_all=True
resnet = InceptionResnetV1(pretrained='vggface2').eval()              #or use casia-webface'


# Read data from folder

dataset = datasets.ImageFolder('photos') # photos folder path 
idx_to_class = {i:c for c,i in dataset.class_to_idx.items()} # accessing names of peoples from folder names

def collate_fn(x):
    return x[0]

loader = DataLoader(dataset, collate_fn=collate_fn)

name_list = [] # list of names corrospoing to cropped photos
embedding_list = [] # list of embeding matrix after conversion from cropped faces to embedding matrix using resnet

for img, idx in loader:
    face, prob = mtcnn0(img, return_prob=True) 
    if face is not None and prob>0.92:
        emb = resnet(face.unsqueeze(0))
        embedding_list.append(emb.detach())
        name_list.append(idx_to_class[idx])        

# save data
data = [embedding_list, name_list] 
torch.save(data, 'data.pt') # saving data.pt file



# Using webcam recognize face

# loading data.pt file
load_data = torch.load('data.pt') 
embedding_list = load_data[0] 
name_list = load_data[1] 

cam = cv2.VideoCapture(0) 
authlist=[]

while True:

    img = np.zeros((600, 1000, 3), np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    auth="AUTHORIZED"
    unauth="UNAUTHORIZED"

    textsize1 = cv2.getTextSize(auth, font, 1, 2)[0]
    textsize2 = cv2.getTextSize(unauth, font, 1, 2)[0]

    textX1 = int((img.shape[1] - textsize1[0]) / 2)
    textY1 = int((img.shape[0] + textsize1[1]) / 1.5)
    textX2 = int((img.shape[1] - textsize2[0]) / 2)
    textY2 = int((img.shape[0] + textsize2[1]) / 1.5  )  


    ret, frame = cam.read()
    if not ret:
        print("Failed to grab Frame")
        break
        
    img = Image.fromarray(frame)
    img_cropped_list, prob_list = mtcnn(img, return_prob=True) 
    
    if img_cropped_list is not None:
        boxes, _ = mtcnn.detect(img)
                
        for i, prob in enumerate(prob_list):
            if prob>0.90:
                emb = resnet(img_cropped_list[i].unsqueeze(0)).detach() 
                
                dist_list = [] # list of matched distances, minimum distance is used to identify the person
                
                for idx, emb_db in enumerate(embedding_list):
                    dist = torch.dist(emb, emb_db).item()
                    dist_list.append(dist)

                min_dist = min(dist_list) # get minumum dist value
                min_dist_idx = dist_list.index(min_dist) # get minumum dist index
                name = name_list[min_dist_idx] # get name corrosponding to minimum dist
                
                box = boxes[i] 
                
                original_frame = frame.copy() # storing copy of frame before drawing on it
                
                if min_dist<0.90:
                    frame = cv2.putText(frame, name+' '+str(min_dist)[2:4]+'.'+str(min_dist)[4:7]+"%", (box[0],box[1]), font, 1, (0,255,0),1, cv2.LINE_AA)
                    
                
                if name in userauth and min_dist>0.4:
                    frame=cv2.putText(frame, auth,(textX1,textY1), font, 2, (0,255,0),8, cv2.LINE_AA)
                    frame = cv2.rectangle(frame, (box[0],box[1]) , (box[2],box[3]), (255,0,0), 2)
                    if name not in authlist:
                        authlist.append(name)
                    
           
                else:
                    frame=cv2.putText(frame, unauth,(textX2,textY2), font, 2, (0,0,255),8, cv2.LINE_AA)


                frame = cv2.rectangle(frame, (box[0],box[1]) , (box[2],box[3]), (255,0,0), 2)

    cv2.imshow("LOGIN", frame)
        
    
    
    k = cv2.waitKey(1)
    if k % 256 == ord('q') or k % 256 == ord('Q'):  # Q
        break
        
    elif k % 256 == ord('s') or k % 256 == ord('S'):  # space to save image
        print('Enter your name :')
        name = input()
        
        # create directory if not exists
        if not os.path.exists('photos/'+name):
            os.mkdir('photos/'+name)
            
        img_name = "photos/{}/{}.jpg".format(name, int(time.time()))
        cv2.imwrite(img_name, original_frame)
        print(" saved: {}".format(img_name))

for x in userauth:
    if x in authlist:
        time.sleep(2)
        cam.release()
        cv2.destroyAllWindows()
        time.sleep(2)
        browse(id,password)   