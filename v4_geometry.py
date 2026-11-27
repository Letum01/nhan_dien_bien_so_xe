"""Conservative geometry and optional evidence-limited plate decoding."""
import re
import cv2
import numpy as np


def rectify(crop, mode):
    h,w=crop.shape[:2]
    if mode=='raw' or min(h,w)<20:return crop,np.eye(3),'identity'
    gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
    edges=cv2.Canny(cv2.GaussianBlur(gray,(5,5),0),55,150)
    contours,_=cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    for contour in sorted(contours,key=cv2.contourArea,reverse=True)[:12]:
        area=cv2.contourArea(contour)
        if not .35*h*w<=area<=.94*h*w:continue
        q=cv2.approxPolyDP(contour,.025*cv2.arcLength(contour,True),True)
        if len(q)!=4 or not cv2.isContourConvex(q):continue
        xy=q.reshape(4,2).astype(np.float32); sums=xy.sum(1); diff=xy[:,0]-xy[:,1]
        idx=[sums.argmin(),diff.argmax(),sums.argmax(),diff.argmin()]
        if len(set(idx))!=4:continue
        tl,tr,br,bl=xy[idx]
        a,b=np.linalg.norm(tr-tl),np.linalg.norm(br-bl);c,d=np.linalg.norm(bl-tl),np.linalg.norm(br-tr)
        if min(a,b,c,d)<16 or min(a,b)/max(a,b)<.63 or min(c,d)/max(c,d)<.63:continue
        angle=np.degrees(np.arctan2(*(tr-tl)[::-1]))
        if abs(angle)>16:continue
        if mode=='deskew':
            if abs(angle)<2:continue
            H=np.vstack([cv2.getRotationMatrix2D((w/2,h/2),float(angle),1),[0,0,1]])
        else:
            inset=.08; target=np.float32([[w*inset,h*inset],[w*(1-inset),h*inset],[w*(1-inset),h*(1-inset)],[w*inset,h*(1-inset)]])
            H=cv2.getPerspectiveTransform(xy[idx],target)
        # Keep the entire original input: expand the canvas to contain its mapped corners.
        corners=project([[0,0],[w,0],[w,h],[0,h]],H)
        if not np.isfinite(corners).all():continue
        lo=np.floor(corners.min(0));hi=np.ceil(corners.max(0)); W,T=map(int,hi-lo)
        if min(W,T)<16 or W>2*w or T>2*h:continue
        H=np.array([[1,0,-lo[0]],[0,1,-lo[1]],[0,0,1]])@H
        return cv2.warpPerspective(crop,H,(W,T),borderMode=cv2.BORDER_REPLICATE),H,mode
    return crop,np.eye(3),'insufficient_geometry'


def project(points,H):return cv2.perspectiveTransform(np.float64(points)[None],H)[0]


def mapped_box(box,H):
    x,y,w,h=box; p=project([[x,y],[x+w,y],[x+w,y+h],[x,y+h]],H)
    a,b=p.min(0); c,d=p.max(0);return [float(a),float(b),float(c-a),float(d-b)]


def decode(prob,classes):
    """At most one anchored substitution from top-3; abstain outside research templates."""
    if not len(prob):return ''
    order=np.argsort(-prob,axis=1); raw=''.join(classes[i] for i in order[:,0]); template=r'[0-9]{2}[A-Z][A-Z0-9]{0,2}[0-9]{4,6}'
    if re.fullmatch(template,raw):return raw
    candidates=[]
    for i,(best,*rest) in enumerate(order):
        for j in rest[:2]:
            # Near-tie evidence; do not present softmax as calibrated confidence.
            if prob[i,j]<.18 or prob[i,best]-prob[i,j]>.10:continue
            text=raw[:i]+classes[j]+raw[i+1:]
            if re.fullmatch(template,text):candidates.append((float(prob[i,j]/prob[i,best]),text))
    return max(candidates)[1] if candidates else raw