"""Experimental bounded frame masking; no GT, OCR text or filename decisions."""
import cv2
import numpy as np
from segmentation import preprocess_plate,segment_characters

CONFIG={'horizontal_length_fraction':.65,'vertical_length_fraction':.70,
        'frame_band_fraction':.12,'protected_height_fraction':[.18,.75],
        'protected_aspect':[.10,1.25],'adaptive_block':21,'adaptive_C':7,
        'row_gap_height_fraction':.55}
VARIANTS=('baseline','border_components','v2_a','v2_b','v2_c','v2_d')


def segment_v2(crop,variant):
    from segmentation_benchmark import experimental_segment
    if variant in ('baseline','border_components'):
        segments,trace=experimental_segment(crop,variant)
        trace['frame_mask']=np.zeros(crop.shape[:2],np.uint8)
        trace['protected_mask']=np.zeros(crop.shape[:2],np.uint8)
        return segments,trace
    if variant not in VARIANTS:raise ValueError('Unknown frozen V2 variant')
    binary=preprocess_plate(crop)
    if variant=='v2_c':
        gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
        enhanced=cv2.createCLAHE(clipLimit=2,tileGridSize=(8,8)).apply(gray)
        binary=cv2.adaptiveThreshold(enhanced,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,21,7)
    h,w=binary.shape
    horizontal=cv2.morphologyEx(binary,cv2.MORPH_OPEN,np.ones((1,max(3,int(w*.65))),np.uint8))
    vertical=cv2.morphologyEx(binary,cv2.MORPH_OPEN,np.ones((max(3,int(h*.70)),1),np.uint8))
    band=np.zeros_like(binary);by=max(1,int(h*.12));bx=max(1,int(w*.12))
    band[:by]=255;band[-by:]=255;band[:,:bx]=255;band[:,-bx:]=255
    frame=cv2.bitwise_and(cv2.bitwise_or(horizontal,vertical),band)
    protected=np.zeros_like(binary)
    if variant=='v2_b':
        n,labels,stats,_=cv2.connectedComponentsWithStats(binary,8)
        for i in range(1,n):
            x,y,cw,ch,area=stats[i]
            if .18*h<=ch<=.75*h and .10<=cw/ch<=1.25:
                protected[labels==i]=255
        frame[protected>0]=0
    cleaned=binary.copy();cleaned[frame>0]=0
    trace={};segments=segment_characters(crop,trace=trace,binary_override=cleaned)
    if variant=='v2_d' and len(segments)>1:
        ordered=sorted(segments,key=lambda p:p[1][1]+p[1][3]/2)
        centers=np.array([b[1]+b[3]/2 for _,b in ordered]);gaps=np.diff(centers)
        split=int(gaps.argmax())+1;median=np.median([b[3] for _,b in ordered])
        if gaps.max()>.55*median:
            segments=sorted(ordered[:split],key=lambda p:p[1][0])+sorted(ordered[split:],key=lambda p:p[1][0])
        else:segments=sorted(ordered,key=lambda p:p[1][0])
    trace.update(frame_mask=frame,protected_mask=protected,original_binary=binary)
    return segments,trace