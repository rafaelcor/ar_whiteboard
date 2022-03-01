import cv2
import mediapipe
import numpy as np
import math

WINDOW_WIDTH, WINDOW_HEIGHT = 640, 480
 
drawingModule = mediapipe.solutions.drawing_utils
handsModule = mediapipe.solutions.hands
 
capture = cv2.VideoCapture(0)
#capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

blank_image = np.full((WINDOW_HEIGHT, WINDOW_WIDTH, 3), 255)
blank_image = blank_image.astype(np.uint8)

blank_image_show = np.full((WINDOW_HEIGHT, WINDOW_WIDTH, 3), 255)
blank_image_show = blank_image.astype(np.uint8)



def distance(x1, x2, y1, y2):
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)

def intersects_rectangle(rec_top_x, rec_top_y, rec_bottom_x, rec_bottom_y, x0, y0):
    print(f"rec_top_x: {rec_top_x}, rec_top_y: {rec_top_y}, rec_bottom_x: {rec_bottom_x}, rec_bottom_y: {rec_bottom_y}, x0: {x0}, y0: {y0}")
    print()
    return rec_top_x <= x0 <= rec_bottom_x and rec_top_y <= y0 <= rec_bottom_y

def show_controls(image):
    ################controles
    start_point = (0,0)
    end_point = (250,80)

    # Blue color in BGR
    color = (255, 0, 0)
      
    # Line thickness of 2 px
    thickness = -1 # fill all rectangle

    image = cv2.rectangle(image, start_point, end_point, color, thickness)

    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
      
    # org
    org = (50, 50)
      
    # fontScale
    fontScale = 1
       
    # Blue color in BGR
    color = (0, 0, 0)
      
    # Line thickness of 2 px
    thickness = 2

    image = cv2.putText(image, 'Borrar', org, font,  fontScale, color, thickness, cv2.LINE_AA)

    ##################

 
with handsModule.Hands(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=1) as hands:
 
    while (True):
 
        ret, frame = capture.read()

        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        image_height, image_width, _ = frame.shape

        blank_image_show = blank_image.copy()
        show_controls(blank_image_show)
 
        if results.multi_hand_landmarks != None:
            for hand_landmarks in results.multi_hand_landmarks:
                x_index, y_index = math.floor(hand_landmarks.landmark[handsModule.HandLandmark.INDEX_FINGER_TIP].x * image_width), math.floor(hand_landmarks.landmark[handsModule.HandLandmark.INDEX_FINGER_TIP].y * image_height)
                
                x_thumb, y_thumb = math.floor(hand_landmarks.landmark[handsModule.HandLandmark.THUMB_TIP].x * image_width), math.floor(hand_landmarks.landmark[handsModule.HandLandmark.THUMB_TIP].y * image_height)

                x_index, y_index = abs(x_index-WINDOW_WIDTH), y_index
                x_thumb, y_thumb = abs(x_thumb-WINDOW_WIDTH), y_thumb

                finger_distance = distance(x_index, x_thumb, y_index, y_thumb)
                
                is_over_delete_button = intersects_rectangle(0, 0, 250, 80, x_index, y_index)

                if finger_distance <= 80:
                    print(is_over_delete_button)
                    if not is_over_delete_button:
                        blank_image[y_index-10:y_index+10,x_index-10:x_index+10] = np.array([0,0,0])
                    else:
                        print("borrar pizarra")
                        blank_image[:] = (255, 255, 255)
                    

                


                #############cursor

                try:
                    blank_image_show[y_index-25:y_index+25,x_index-25:x_index+25] = [0,0,0]
                except:
                    pass

                #################

                #drawingModule.draw_landmarks(frame, hand_landmarks, handsModule.HAND_CONNECTIONS)
 
        cv2.imshow('Test hand', frame[:, ::-1, :])
        cv2.imshow('whiteboard', blank_image_show)
 
        if cv2.waitKey(1) == 27:
            break
 
cv2.destroyAllWindows()
capture.release()