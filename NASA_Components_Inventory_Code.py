import numpy as np
import cv2
import os
from paddleocr import PaddleOCR


cap = cv2.VideoCapture(0)

#-----------------------------------------------------------OCR initialization
ocr = PaddleOCR(
    lang='en', 
    use_textline_orientation=False, 
    use_doc_orientation_classify=False,
    enable_mkldnn=False
)

while True:
    ret, frame = cap.read()
    cv2.imshow('frame', frame)

    #to black and white
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #denoising
    #cv2.fastNlMeansDenoising(frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

    if key == ord(' '):
        cv2.imwrite("label.jpg", frame)
        cv2.imshow('captured snapshot',frame)
        
        if os.path.exists("label.jpg"):
#------------------------------------------------------------------------------------------OCR Reading            
            result = ocr.predict("label.jpg")
            result = result[0]['rec_texts']

            print(result)
        else:
            print("NO IMAGE TAKEN BUDDY!")

cap.release()
cv2.destroyAllWindows()



#---------------------------------------------------------------------------------------------------------------PANDAS experimental testing
'''responce = "M55342K11B10E0R$RESISTOR_10k_OHMS_1%$0402$0406J$124168$570"
Box = str("1")

row = responce.split("$")

row.append(Box)

print(row)
#print(row[6])



'''