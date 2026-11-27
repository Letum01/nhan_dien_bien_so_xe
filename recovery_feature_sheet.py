"""Visualize existing reviewed glyphs; no classifier inference or relabeling."""
import csv

import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from skimage.morphology import skeletonize

from train_ocr_v2 import REAL, ROOT


def main():
    with (REAL/'characters.csv').open(encoding='utf-8-sig',newline='') as f:
        rows=[r for r in csv.DictReader(f) if r['split'] in ('train','validation')]
    wanted='0ODQ1IT2Z38A45S6GBKNMU V'.replace(' ','')
    sheet=np.full((64*(len(wanted)+1),640,3),245,np.uint8)
    names=['Glyph','HOG','LBP','Sobel','Contours','Distance','Skeleton','Projection']
    for j,name in enumerate(names):cv2.putText(sheet,name,(80*j+2,21),0,.42,(0,0,0),1)
    for i,c in enumerate(wanted,1):
        row=next((r for r in rows if r['class_label']==c),None)
        cv2.putText(sheet,c,(2,i*64+58),0,.4,(0,0,0),1)
        if row is None:
            cv2.putText(sheet,'No reviewed sample',(30,i*64+25),0,.45,(0,0,0),1);continue
        im=cv2.imread(str(REAL/row['image_path']),0); ink=(im>0).astype(np.uint8)
        _,hv=hog(im,orientations=9,pixels_per_cell=(8,8),cells_per_block=(2,2),visualize=True)
        dx=cv2.Sobel(im,cv2.CV_32F,1,0);dy=cv2.Sobel(im,cv2.CV_32F,0,1)
        contours,_=cv2.findContours(ink,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        edge=cv2.drawContours(np.zeros_like(im),contours,-1,255,1)
        proj=np.zeros_like(im)
        for j,n in enumerate(ink.sum(0)):proj[32-int(n):,j]=255
        for j,tile in enumerate([im,hv,local_binary_pattern(im,8,1,'uniform'),cv2.magnitude(dx,dy),edge,
                                  cv2.distanceTransform(ink,cv2.DIST_L2,3),skeletonize(ink).astype(np.uint8),proj]):
            tile=cv2.normalize(tile,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
            sheet[i*64:i*64+48,j*80+12:j*80+60]=cv2.cvtColor(cv2.resize(tile,(48,48),interpolation=cv2.INTER_NEAREST),cv2.COLOR_GRAY2BGR)
    if not cv2.imwrite(str(ROOT/'reports/accuracy_recovery/feature_sheet.png'),sheet):raise OSError('Cannot save feature sheet')


if __name__=='__main__':main()