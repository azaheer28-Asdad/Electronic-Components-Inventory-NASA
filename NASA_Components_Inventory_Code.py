import numpy as np
import cv2
import ollama
import os
import easyocr
'''def question ():
    prompt = """you are a professional image analyzer. you are the world's best analyzer, so I chose you. 
    I want you to look at this image and analyze it carefully. 
    I want you to extract the part number the description, the size, the date code, the Lot code, and the quantity.
     It is very important that you make sense of these numbers, so that they arent random but actually logical and therefore correct.
      you may not see the size of the component in the label, but you can just look at the part number and extract adn translate the size from there.
       if a value on the lable is crossed out, I want you to look at the handwritten note on there that corrects it.
        if you still can't find that number that corrects the crossed out number, and this goes for any value, if its crossed out, then look for the correction,
         then if you still cnat find it, output '?'. DO NOT USE COMMAS in the output, and not spaces, just use underscores_ instead.
         After carefully analyzing the label, ensuring high accuracy, then I want you to return to me with the following return syntax: 
         partnumber$Description$Size$Date_code$Lot_Code$quantity$"""


    print("sending to ollama...")
    responce = ollama.chat (
        model = 'llava',
        messages = [{
            "role":"user", 
            "content":prompt,
            'images': [open('label.jpg', 'rb').read()]
            }]
    )
    return responce['message']['content']'''

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    cv2.imshow('frame', frame)

    #to black and white
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #denoising
    

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

    if key == ord(' '):
        cv2.imwrite("label.jpg", frame)
        cv2.imshow('captured snapshot',frame)
        
        if os.path.exists("label.jpg"):

            reader = easyocr.Reader(['en'])
            result = reader.readtext("label.jpg", detail = 0)

            print(result)

        else:
            print("NO IMAGE TAKEN BUDDY!")

cap.release()
cv2.destroyAllWindows()



#-----------------------------------------------------------------------------------------------------------------------------------PANDAS experimental testing
'''responce = "M55342K11B10E0R$RESISTOR_10k_OHMS_1%$0402$0406J$124168$570"
Box = str("1")

row = responce.split("$")

row.append(Box)

print(row)
#print(row[6])



'''





