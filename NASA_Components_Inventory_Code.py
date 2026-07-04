import numpy as np
import cv2
import os
from paddleocr import PaddleOCR
import re




def label_box(result_text, match, result):

    for index, item in enumerate(result_text):
        if item == match:
            break

    #index is where first_match shows up 'our' in our case

    anchor_box = result[0]['dt_polys'][index]

    # Point 0
    x0 = anchor_box[0][0]
    y0 = anchor_box[0][1]

    # Point 1
    x1 = anchor_box[1][0]
    y1 = anchor_box[1][1]

    # Point 2
    x2 = anchor_box[2][0]
    y2 = anchor_box[2][1]

    # Point 3
    x3 = anchor_box[3][0]
    y3 = anchor_box[3][1]

    return x0, y0, x1, y1, x2, y2, x3, y3




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
            result_text = result[0]['rec_texts']

            print(result_text)
        else:
            print("NO IMAGE TAKEN BUDDY!")

cap.release()
cv2.destroyAllWindows()


#-----------------------------------------------------------------------------------call function for label rect vals
match = 'OUR'

x0, y0, x1, y1, x2, y2, x3, y3 = label_box(result_text, match, result)

#-----------------------------------------------------------------------------------calc slope
# Top line runs from Point 0 to Point 1
m_top = (y1 - y0) / ((x1 - x0) + 1e-5)

# Bottom line runs from Point 3 to Point 2
m_bottom = (y2 - y3) / ((x2 - x3) + 1e-5)

best_match_text = None
shortest_distance = float('inf') # Start with an infinitely large distance

for cand_index, cand_text in enumerate(result_text):
    if cand_text == match:
        continue

    # Grab the bounding box for this specific candidate word
    cand_box = result[0]['dt_polys'][cand_index]
    
    # Extract the Top-Left corner of this candidate
    cx0, cy0 = cand_box[0][0], cand_box[0][1]
    
    # Extract the Bottom-Left corner
    cx3, cy3 = cand_box[3][0], cand_box[3][1]

    if cx0 > x1:
        expected_y_top = (m_top * (cx0 - x1)) + y1
        expected_y_bottom = (m_bottom * (cx0 - x2)) + y2

        buffer = 0

        if (expected_y_top <= (cy3 + buffer)) and (expected_y_bottom >= (cy0 - buffer)):
            
            dist = cx0 - x1
            if dist < shortest_distance:
                shortest_distance = dist
                best_match_text = cand_text


if best_match_text:
    print(f"BINGO! The value closest to 'OUR' is: {best_match_text}")
else:
    print("Could not find any overlapping values to the right.")

#make function out of all of that! and then find a way to use it without pressing q and just right after the space, maybe call it right after space is pressed or smth
#change this for all of the vals ou want by doing the above and changing that one word. also dont forget to add the stuff in the () for the def func











#---------------------------------------------------------------------------------------------------------------PANDAS experimental testing
'''responce = "M55342K11B10E0R$RESISTOR_10k_OHMS_1%$0402$0406J$124168$570"
Box = str("1")

row = responce.split("$")

row.append(Box)

print(row)
#print(row[6])



'''