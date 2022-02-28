# ElDet
We propose an anchor-free general ellipse object detector that can better detect ellipse objects of any class based on object shape information, and can be applied to downstream tasks such as face detection with a few modifications.
<img src="/imgs/overview.jpg" width=800>

## Detection Results
### 1. GED dataset
<img src="/imgs/GED.jpg" width=500>

### 2. FDDB dataset
<img src="/imgs/FDDB.jpg" width=500>

## Environment Configuration
- python>=3.6
- torch>=1.7.0
- *others see requirements.txt*

1. Use the requirements.txt to build the basic environment;
2. [DCNv2](https://github.com/jinfagang/DCNv2_latest.git).
  
    ```bash
    cd DCNv2
    sh maks.sh
    ```
    

## Data
### Data annotation
We use [VIA](https://www.robots.ox.ac.uk/~vgg/software/via/) to make the labels, and export *.json* file. The transform the **JSON** format to **COCO** format. 

*data_process.py* is a routine for format transformation.

### Data format
We adapt **COCO** format and **bbox = [cx, cy, a, b, theta]**, where *cx, cy* are center point coordinates, *a, b* are major axis and minor axis of ellipse, *theta*\in (-90, 90] is rotation angle of ellipse.

### Data folder format
```
--data
  --your data name
    --images
      --1.jpg
      --2.jpg
      ...    
    --annotations
      --train.json
      --test.json
```
