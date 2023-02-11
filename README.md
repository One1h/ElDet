# ElDet
We propose an anchor-free general ellipse object detector that can better detect ellipse objects of any class based on object shape information, and can be applied to downstream tasks such as face detection with a few modifications.
<div align=center>
<img src="/imgs/overview.jpg" width=800>
</div>

## 1. Environment Configuration
- python>=3.6
- torch>=1.7.0
- *others see requirements.txt*

1. Use the requirements.txt to build the basic environment;
2. [DCNv2](https://github.com/jinfagang/DCNv2_latest.git)
  
    ```bash
    cd DCNv2
    sh make.sh
    ```
3. Copy the folder *./DCNv2/build* to *./dcn*.
    

## 2. Data Format
### 2.1 Data annotation
We use [VIA](https://www.robots.ox.ac.uk/~vgg/software/via/) to make the labels, and export *.json* file. The transform the **JSON** format to **COCO** format. 

*data_process.py* is a routine for format transformation.

### 2.2 Data format
We adapt **COCO** format and **bbox = \[cx, cy, a, b, θ]**, where *cx, cy* are center point coordinates, *a, b* are major axis and minor axis of ellipse, *θ* ∈[-90, 90） is rotation angle of ellipse.

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

## 3. Detection Results
### 3.1. GED dataset
**GED dataset download link ([Baidu Netdisk](https://pan.baidu.com/s/1HZ8buHahd-jx39hTklf-7A?pwd=ezgs ))**
<div align=center>
<img src="/imgs/GED.jpg" width=500>
</div>

### 3.2. FDDB dataset
<div align=center>
<img src="/imgs/FDDB.jpg" width=500>
</div>

# 4. Citataion
```
@inproceedings{liao2020speech2video,
  title={ElDet: An Anchor-free General Ellipse Object Detector},
  author={Wang, Tian hao and Lu, Changsheng and Shao, Ming and Yuan, Xiaohui and Xia, Siyu},
  booktitle={Proceedings of the Asian Conference on Computer Vision},
  year={2022}
}
```
```
@article{lu2019arc,
  title={Arc-Support Line Segments Revisited: An Efficient High-Quality Ellipse Detection},
  author={Lu, Changsheng and Xia, Siyu and Shao, Ming and Fu, Yun},
  journal={IEEE Transactions on Image Processing},
  volume={29},
  pages={768--781},
  year={2020},
  publisher={IEEE}
}
```
