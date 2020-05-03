#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2 as cv
import numpy as np
import dlib


def norm(a, b, p=2):
    a = np.array(a)
    b = np.array(b)
    return np.power(np.sum(np.power(a - b, p)), 1/p)

def kmeans(prev_frame, frame):
    centers = None
    flags = cv.KMEANS_RANDOM_CENTERS
    if prev_frame is not None:
        prev_frame = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame_diff = cv.subtract(frame, prev_frame)
        fx, fy = np.array(np.where(frame_diff > 6))
        frame_xy = np.stack([fy, fx], 1)
        if len(frame_xy) == 0:
            return None
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 1000, 2)
        _, _, centers = cv.kmeans(np.float32(frame_xy), 6, None, criteria, 5, flags)
        corners = set()
        for i, center in enumerate(centers):
            for j, other in enumerate(centers[i + 1:]):
                dist = norm(center, other)
                if dist < 120:
                    corners = corners.union({tuple(center), tuple(other)})
        if len(corners) == 4:
            min_x, min_y = None, None
            max_x, max_y = None, None
            for corner in corners:
                if min_x is None or corner[0] < min_x:
                    min_x = int(corner[0])
                if max_x is None or corner[0] > max_x:
                    max_x = int(corner[0])
                if min_y is None or corner[1] < min_y:
                    min_y = int(corner[1])
                if max_y is None or corner[1] > max_y:
                    max_y = int(corner[1])
            return (min_x, min_y, max_x - min_x, max_y - min_y)
    return None

def calibrate(frame):
    r = dlib.get_frontal_face_detector()(frame)
    return (1.8 / r.pop().top()) if len(r) else None


def track():
    cap = cv.VideoCapture(0)
    if cap.isOpened() == False:
        print("Error opening video stream or file")
    tracker = cv.TrackerTLD_create()
    init_bb = None
    prev_frame = None
    prev_box = None
    dist_unit = None
    while cap.isOpened():
        ret, frame = cap.read()
        if dist_unit is None:
            dist_unit = calibrate(frame)
        elif init_bb is not None:
            tick_count = cv.getTickCount()
            success, box = tracker.update(frame)
            fps = cv.getTickFrequency() / (cv.getTickCount() - tick_count)
            if success:
                x, y, w, h = (int(i) for i in box)
                frame = cv.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (200, 70, 0)
                )
                if prev_box is not None:
                    dist = norm(box[0:1], prev_box[0:1])
                    speed = dist * dist_unit * fps
                    frame = cv.putText(
                        frame,
                        f"Speed: {speed:.3F}ms",
                        (50, 50),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (200, 70, 0),
                        2
                    )
            prev_box = box
        else:
            init_bb = kmeans(prev_frame, frame)
            if init_bb is not None:
                tracker.init(frame, init_bb)
        cv.imshow('Frame', frame)
        prev_frame = frame
        if ret:
            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        else:
            break
    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    print("Hello there, starting capture...")
    track()
    print("Bye.")
