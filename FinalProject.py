import numpy as np
import cv2
import pandas as pd
import face_recognition as fc
import time
import random as rd
import smtplib
import xlrd
import datetime
import argparse

conf='yolov3.cfg'
weights='yolov3.weights'
txt='yolov3.txt'
classes = None
flag = 0

fcc=0
v=cv2.VideoCapture(0)
fd=cv2.CascadeClassifier(r"haarcascade_frontalface_alt2.xml")
Data = pd.read_excel("Data.xlsx")
df = pd.DataFrame(Data)
def cap():
        ret,i=v.read()
        j=cv2.cvtColor(i,cv2.COLOR_BGR2GRAY)
        f=fd.detectMultiScale(j)
        if len(f)==1:
            for(x,y,w,h) in f:
                image=i[y:y+h,x:x+w].copy()
                fl=fc.face_locations(image)
                fcl=fc.face_encodings(image,fl)
                cv2.imshow('image',image)
                k= cv2.waitKey(5)
        
                return fcl
                break
        else:
            print("Face not Detected")
    

def enterdata():
    name=input("Name: ")
    location=input("Where did you met: ")
    number=int(input("Mobile Number: "))
    email=input("E-Mail: ")
    notes=input("Enter some notes for the person without pressing Enter\n")
    print("Hold Still The Camera will initialize to detect your face in few seconds")
    print("Name: ",name,"\nLocation: ",location,"\nNumber: ",number,"\nEmail: ",email,"\nNotes: ",notes)
    time.sleep(2)
    q=0
    while(q!=1):
        try:
            fcc=cap()
            if len(fcc) != 0:
                print("Successfully Gathered Data")    
                q=1
                return name,location,number,email,notes,fcc
        except:
            pass

def sendmail():
    server=smtplib.SMTP('smtp.gmail.com',587)
    server.starttls()
    server.login("tvseriescut@gmail.com","shakiralisayed0@gmail.com")
    msg="Subject: You have added new contact: "+name+" \nWelcome!\nA new contact has been added to your logbook with the following details\nName:"+name+"\nWhere did we met:"+location+"\nNumber:"+str(number)+"\nE-mail:"+email+"\nNotes:"+notes+""
    server.sendmail("Leonardo Da Vinci","tvseriescut@gmail.com",msg)


def Start_Remember():
        global df,name,location,number,email,notes,fcc
        name,location,number,email,notes,fcc = enterdata()        
        y=datetime.datetime.now()
        dates=y.strftime("%x")
        times=y.strftime("%X")
        v.release()
        sendmail()
        print("Mail Sent")
        dataf=pd.DataFrame({"Name":[name],
                            "Number":[number],
                            "Email":[email],
                            "location_of_meeting":[location],
                            "date_of_first_meeting":[dates],
                            "time_of_first_meeting":[times],
                            "Notes":[notes],
                            "Encoding":list(fcc)})
        df=df.append(dataf,ignore_index=True,sort=False)
        df.to_excel("Data.xlsx",index=False)

        print("Successfully Entered Data")

def StringListtoFloatArray(column):
        column=column.strip("[")
        column=column.strip("]")
        a=column.split(' ')
        newlist=[]
        r=[x for x in a if x!='']
        for x in r:
            if '\n' in x:
                f=float(x[:-2])
                newlist.append(f)
            elif '[' in x:
                f=float(x[1:])
                newlist.append(f)
            elif ']' in x:
                f=float(x[:-2])
                newlist.append(f)
            else:
                newlist.append(float(x))
        return newlist
    
def Start_Recall():
    print("Press 'r' to recognize and 'q' to quit")
    while(1):
        ret,i=v.read()
        j=cv2.cvtColor(i,cv2.COLOR_BGR2GRAY)
        f=fd.detectMultiScale(j)
        if len(f) == 0:
            continue
        for(x,y,w,h) in f:
            img=i[y:y+h,x:x+w].copy()
        cv2.imshow('face',img)
        k=cv2.waitKey(5)
        if k==ord('q'):
            cv2.destroyAllWindows()
            break
        elif k==ord('r'):
            fm=fc.face_locations(img)
            fcm=fc.face_encodings(img,fm)
            if len(fcm) == 0:
                continue
            length=len(df['Encoding'])
            for w in range(length):
                column=df['Encoding'][w]
                newlist=StringListtoFloatArray(column)
                new_array=np.array(newlist,dtype='float64')
                d=fc.compare_faces([new_array],fcm[0])
                if d==[False]:
                    continue
                else:
                    name,number,email,l,d,t,notes = df.iloc[0,0:7]
                    print("You have met that person here are the details:\nName",name,"\nNumber",number,"\nEmail",email,"\nLocation",l,"\ndate",d,"\ntime",t)

def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, classes,class_id, confidence,COLORS,x, y, x_plus_w, y_plus_h):

    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    
def Recognize(image):
    global flag
    flag = 1
    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392

    classes = None

    with open(txt, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

    net = cv2.dnn.readNet(weights, conf)

    blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

    net.setInput(blob)

    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4


    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])


    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        draw_prediction(image, classes,class_ids[i], confidences[i],COLORS, round(x), round(y), round(x+w), round(y+h))
    return image
    
image=1
hogaya=0
while True:
    if flag==0:
        print("1.To add someone to your logbook\n2.To Recall from logbook\n3.Object Detection in a static image \n4. To Exit")
        choice = int(input("Enter your choice "))
        if choice == 1 and flag==0:
            Start_Remember()
            v.release()
        elif choice == 2 and flag==0:
            Start_Recall()
            v.release()
        elif choice == 3 and flag==0:
            flag=1
            print('Enter The Path of image with image name and extension')
            print(r'For Example C:\Users\HP\Documents\Machine Learning Python\Project - Jarvis 2.0\dog.jpg')
            inp= input('')
            image = cv2.imread(inp)
            image=Recognize(image)
            hogaya=1
            print('Press "q" for saving and quit IMAGE .')
        elif choice == 4 and flag==0:
            break
        else:
            print("Enter Correct Choice")
    elif hogaya==1:
            cv2.imshow("object detection", image)
            key=cv2.waitKey(5)
            if key==ord('q'):
                flag = 0
                hogaya=0
                cv2.imwrite("object-detection.jpg", image)
                cv2.destroyAllWindows()
