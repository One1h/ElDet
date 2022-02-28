# ElDet
We propose an Anchor-free general ellipse object detector that can better detect ellipse objects of any class based on object shape information, and can be applied to downstream tasks such as face detection with a few modifications.
<img src="/imgs/overview.jpg">

## Detection results
### GED dataset
<img src="/imgs/GED.jpg">

### FDDB dataset
<img src="/imgs/FDDB.jpg">

## Environment configuration
1. Use the requirements.txt The basic environment;
2. DCNv2.

## Data format
We adapt **COCO** format and **bbox = [cx, cy, a, b, theta]**, where *cx, cy* are center point coordinates, *a, b* are major axis and minor axis of ellipse, *theta*\in (-90, 90] is rotation angle of ellipse.
