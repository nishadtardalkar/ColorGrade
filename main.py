from colorgrade import ColorGrade
import cv2
import numpy as np
import matplotlib.pyplot as plt

leftclickdown = False
x_start = y_start = x_prev = y_prev = 0
def mouseevent(event, x, y, flags, param):
    global leftclickdown, x_start, y_start, x_prev, y_prev
    global wdivider
    global activeOption
    global colorgrade
    global PH, PW
    global H, W
    global preview
    
    if event == cv2.EVENT_LBUTTONDOWN:
        leftclickdown = True
        x_start = x
        y_start = y
        x_prev = x
        y_prev = y
    elif event == cv2.EVENT_LBUTTONUP:
        leftclickdown = False
    elif event == cv2.EVENT_MOUSEMOVE:
        if leftclickdown:
            if activeOption == 'Edit-Divider':
                wdivider = min(max(x, 0), PW)
            elif activeOption == 'Edit-R':
                colorgrade.add_cc_val('R', (x-PW*0.1)/(PW*0.8), (y_prev-y)/PH)
                
            elif activeOption == 'Edit-G':
                colorgrade.add_cc_val('G', x/PW, (y_prev-y)/PH)
                
            elif activeOption == 'Edit-B':
                colorgrade.add_cc_val('B', x/PW, (y_prev-y)/PH)
                
            elif activeOption == 'Edit-R-Res':
                colorgrade.set_cc_resolution('R', colorgrade.cc_r_res + (y_prev-y))
                
            elif activeOption == 'Edit-G-Res':
                colorgrade.set_cc_resolution('G', colorgrade.cc_g_res + (y_prev-y))
                
            elif activeOption == 'Edit-B-Res':
                colorgrade.set_cc_resolution('B', colorgrade.cc_b_res + (y_prev-y))
                
            elif activeOption == 'Scale':
                PH += y_prev-y
                PW = int(W*PH/H)
                preview = np.zeros((PH,PW,3), dtype=np.float32)
                colorgrade.set_size(PH,PW)

        x_prev = x
        y_prev = y
            
# SETTINGS
videofile = "sample.mp4"


cap = cv2.VideoCapture(videofile)
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
FPS = cap.get(cv2.CAP_PROP_FPS)
PH = H
PW = W

colorgrade = ColorGrade()

# PREVIEW
wdivider = PW//2
activeOption = 'Edit-Divider'
paused = -1
cv2.startWindowThread()
cv2.namedWindow('preview')
cv2.setMouseCallback('preview', mouseevent)
preview = np.zeros((PH,PW,3), dtype=np.float32)
colorgrade.set_size(PH, PW)
while cap.isOpened():
    if paused != -1:
        cap.set(cv2.CAP_PROP_POS_MSEC, paused)
    ret, frame = cap.read()            
    if not ret:
        break
    frame = frame/255
    frame = cv2.resize(frame, (PW, PH))

    preview[:,:wdivider,:] = frame[:,:wdivider,:]
    frame = colorgrade.apply(frame)
    preview[:,wdivider:,:] = frame[:,wdivider:,:]
    
    if activeOption == 'Edit-R' or activeOption == 'Edit-R-Res':
        preview = colorgrade.get_cc_contour(preview, 'R')
    elif activeOption == 'Edit-G' or activeOption == 'Edit-G-Res':
        preview = colorgrade.get_cc_contour(preview, 'G')
    elif activeOption == 'Edit-B' or activeOption == 'Edit-B-Res':
        preview = colorgrade.get_cc_contour(preview, 'B')
        
    cv2.imshow('preview', preview)
    k = cv2.waitKey(1)
    #print(k)
    
    if k == 27:
        activeOption = 'Edit-Divider'
    elif k == 114:
        activeOption = 'Edit-R'
    elif k == 18:
        activeOption = 'Edit-R-Res'
    elif k == 103:
        activeOption = 'Edit-G'
    elif k == 7:
        activeOption = 'Edit-G-Res'
    elif k == 98:
        activeOption = 'Edit-B'
    elif k == 2:
        activeOption = 'Edit-B-Res'
    elif k == 115:
        activeOption = 'Scale'
    elif k == 32:
        if paused == -1:
            paused = cap.get(cv2.CAP_PROP_POS_MSEC)
        else:
            paused = -1
    elif k == 46:
        cap.set(cv2.CAP_PROP_POS_MSEC, cap.get(cv2.CAP_PROP_POS_MSEC)+5000)
    elif k == 44:
        cap.set(cv2.CAP_PROP_POS_MSEC, cap.get(cv2.CAP_PROP_POS_MSEC)-5000)
    elif k == 113:
        break
cv2.destroyAllWindows()

# RENDER
colorgrade.set_size(H, W)
cap = cv2.VideoCapture(videofile)
frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
out = cv2.VideoWriter('output.avi', fourcc, FPS, (W,H))
c = 0
while cap.isOpened():
    ret, frame = cap.read()            
    if not ret:
        break
    frame = frame/255
    frame = colorgrade.apply(frame)
    out.write((frame*255).astype(np.int8))
    c += 1
    print(c,'/',frames)
out.release()
cap.release()

