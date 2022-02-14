import numpy as np
import cv2

def iou_ellipse(bbox1, bbox2, shape):
    board1 = np.zeros((shape[0], shape[1]), np.uint8)
    cv2.ellipse(board1, (int(bbox1[0]), int(bbox1[1])), (int(bbox1[2]), int(bbox1[3])), int(bbox1[4]),
                startAngle=0, endAngle=360, color=1, thickness=-1)
    board2 = np.zeros((shape[0], shape[1]), np.uint8)
    cv2.ellipse(board2, (int(bbox2[0]), int(bbox2[1])), (int(bbox2[2]), int(bbox2[3])), int(bbox2[4]),
                startAngle=0, endAngle=360, color=1, thickness=-1)
    board = board1 + board2

    inter = len(board[np.where(board > 1)])
    union = len(board[np.where(board > 0)])

    iou = 1.0 * inter / union if union else 0
    return iou

if __name__ == '__main__':
    bbox1 = [256, 256, 50, 25, 45]
    bbox2 = [256, 256, 50, 25, 75]
    print(iou_ellipse(bbox1, bbox2, (512, 512)))