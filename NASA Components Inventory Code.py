import numpy as np
import cv2
import ollama
import os


def question ():
    
    prompt = """look at this image and tell me what you see, give the the labels and what each thing is equal to..."""
    
    print("sending to ollama...")



    responce = ollama.chat (
        model =  'llama3.2-vision',
        messages = [{
            "role":"user", 
            "content":prompt,
            'images': ['label.jpg']
            }]
    )



cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    cv2.imshow('frame', frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

    if key == ord(' '):
        cv2.imwrite("label.jpg", frame)
        cv2.imshow('captured snapshot',frame)


cap.release()
cv2.destroyAllWindows()

if os.path.exists("label.jpg"):
    ai_responce = question()

    print("====================================responce")
    print(ai_responce)
    print("==========================================")
else:
    print("NO IMAGE TAKEN BUDDY!")