import numpy as np
import cv2
import os
from paddleocr import PaddleOCR
import pygame
import time
import pandas as pd
import csv


#=======================================================================================editable section: SIZE

    



def label_val(result_text, result, match = None):

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

    #---------------------------------------------------------------calculating val
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

            buffer = 2
            
            # The box is fully swallowed by the beam
            trapped_inside = (expected_y_top <= (cy3 + buffer)) and (expected_y_bottom >= (cy0 - buffer))

            # Check if top line hits, bottom line hits, OR it's trapped inside
            if (expected_y_top <= (cy3 + buffer) and expected_y_top >= (cy0 - buffer)) or (expected_y_bottom >= (cy0 - buffer) and expected_y_bottom <= (cy3 + buffer)) or trapped_inside:
                
                dist = cx0 - x1
                if dist < shortest_distance:
                    shortest_distance = dist
                    best_match_text = cand_text

    if match is not None:
        return best_match_text

#------------------------------------------------------------------------------------list indexing (6 slots)
#===============================================================================================================================================================================================================================================================================================================
    #label (all caps, pay attension!!) MAKE SURE THE VALS DONT REPEAT!!!!!!:

    box_number = 1 #insert box number!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    sorted_list = []

    #part number:
    match = 'OUR'
    best_match_text = label_val(result_text, result, match)
    sorted_list.insert(0,best_match_text)

    part_number = best_match_text
    #description:
    match = 'VALUE'
    best_match_text = label_val(result_text, result, match)
    sorted_list.insert(1,best_match_text)
    #size:
    MIL_SIZE_MAP = {
    "01": "0502",
    "02": "0505",
    "03": "1005",
    "04": "1505",
    "05": "2208",
    "06": "0705",
    "07": "1206",
    "08": "2010",
    "09": "2512",
    "10": "1010",
    "11": "0402",
    "12": "0603",
    "13": "0302"
    }
    if (part_number is not None) and (len(part_number) >=9):
        size_code = part_number[7:9]
        size = MIL_SIZE_MAP.get(size_code, "None")
        sorted_list.insert(2, size)
    else:
        size = "None"
        sorted_list.insert(2, size)

    #date Code:
    match = 'DATE'
    best_match_text = label_val(result_text, result, match)
    sorted_list.insert(3,best_match_text)
    #Lot Code:
    match = 'LOT'
    best_match_text = label_val(result_text, result, match)
    sorted_list.insert(4,best_match_text)
    #quantity:
    match = 'QTY.'
    best_match_text = label_val(result_text, result, match)
    sorted_list.insert(5,best_match_text)
    #box number:
    sorted_list.insert(6,box_number)

#==========================================================================================================================================================================================================================================================================================================================   
#--------------------------------------------------------------------list indexing (6 slots)      
    return sorted_list




pygame.init()
pygame.mixer.init()
def good():
    good_sfx = pygame.mixer.Sound("good.mp3")
    good_sfx.set_volume(0.3)
    good_sfx.play()
    time.sleep(1)

def bad():
    bad_sfx = pygame.mixer.Sound("bad.mp3")
    bad_sfx.set_volume(1)
    print("ERROR!!!!!")
    bad_sfx.play()
    time.sleep(0.4)
    bad_sfx.play()
    time.sleep(0.4)
    bad_sfx.play()
    time.sleep(1)


cap = cv2.VideoCapture(0)

#-----------------------------------------------------------OCR initialization
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 500)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)

# Update your OCR initialization to include use_gpu=True
ocr = PaddleOCR(use_textline_orientation=True, lang='en', device="gpu:0")

while True:
    ret, frame = cap.read()
    cv2.imshow('frame', frame)

    #to black and white
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #denoising
    #cv2.fastNlMeansDenoising(frame)

    key = cv2.waitKey(1)

    if key == ord(' '):
        cv2.imwrite("label.jpg", frame)
        cv2.imshow('captured snapshot',frame)

        time.sleep(0.08)

        ret, frame1 = cap.read()
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        cv2.imwrite("label1.jpg", frame1)

        print("images captured")

       
        if os.path.exists("label.jpg") and os.path.exists("label1.jpg"):
#------------------------------------------------------------------------------------------OCR Reading            
            result = ocr.predict("label.jpg")
            result_text = result[0]['rec_texts']

            print(result_text)

            result1 = ocr.predict("label1.jpg")
            result_text1 = result1[0]['rec_texts']

            print(result_text1)
        else:
            print("NO IMAGE TAKEN BUDDY!")
#---------------------------------------------------------------------------------------------------------run all processing code here (2 tab):
        sorted_list = label_val(result_text, result)
        print(sorted_list)

        sorted_list1 = label_val(result_text1, result1)
        print(sorted_list1)

        if (None not in sorted_list) and (None not in sorted_list1) and ("None" not in sorted_list) and ("None" not in sorted_list1) and (sorted_list == sorted_list1):
            
            csv_file_name = f"Box {sorted_list[6]} Contents.csv"

            with open(csv_file_name, mode='a', newline='') as csvFile:
                writer = csv.writer(csvFile)

                writer.writerow(sorted_list)
                
                good()
        
        else:
            print("SCAN AGAIN!")
            bad()


    if key == ord('q'):
        header_list = ['Part Number', 'Description', 'Size', 'Date Code', 'Lot Code', 'Quantity', 'Box Number']
        df = pd.read_csv(csv_file_name, header=None)
        df.to_csv(csv_file_name, header=header_list, index=False)

        break


            
cap.release()
cv2.destroyAllWindows()