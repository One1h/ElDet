# ElDet
We propose an anchor-free general ellipse object detector that can better detect ellipse objects of any class based on object shape information, and can be applied to downstream tasks such as face detection with a few modifications.
<img src="/imgs/overview.jpg">

## Detection results
### 1. GED dataset
<img src="/imgs/GED.jpg">

### 2. FDDB dataset
<img src="/imgs/FDDB.jpg">

## Environment configuration
- python>=3.6
- torch>=1.7.0
- *others see requirements.txt*

1. Use the requirements.txt build the basic environment;
2. [DCNv2](https://github.com/jinfagang/DCNv2_latest.git).

## Data format
We adapt **COCO** format and **bbox = [cx, cy, a, b, theta]**, where *cx, cy* are center point coordinates, *a, b* are major axis and minor axis of ellipse, *theta*\in (-90, 90] is rotation angle of ellipse.
