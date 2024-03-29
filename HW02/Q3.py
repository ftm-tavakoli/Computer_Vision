import cv2
import Q2
import numpy as np
import copy

cap = cv2.VideoCapture('vid1.mp4')

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break


    edges = q2.blur_Canny(frame)

    mask = q2.make_mask(image=edges, point=[490, 280])
    linesP = cv2.HoughLinesP(edges*mask, 1, np.pi / 180, 50, None, 20, 10)
    final_result = q2.draw_lane_lines(image=copy.deepcopy(frame), lines=q2.lane_lines(frame, linesP))
    out.write(final_result)

    cv2.waitKey(25)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
out.release()
