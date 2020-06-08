import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.ndimage import uniform_filter
import cv2


def init_focal_point(config, focal_size=None, seed=None):
    if seed is not None:
        np.random.seed(seed)
    fs_x, fs_y = focal_size if focal_size is not None else config.focal_size
    max_x, max_y = config.frame_size
    x = np.random.randint(fs_x + 1, max_x - fs_x - 1)
    y = np.random.randint(fs_y + 1, max_y - fs_y - 1)
    return (x, y)

def move_focal_point(action, focal_point, config, focal_size=None):
    x, y = focal_point
    focal_step = config.focal_step
    max_x, max_y = config.frame_size
    fs_x, fs_y = config.focal_size
    if action == 0:
        pass
    elif action == 1:
        if x + focal_step < max_x - fs_x - 1:
            x += focal_step
    elif action == 2:
        if y + focal_step < max_y - fs_y - 1:
            y += focal_step
    elif action == 3:
        if x - focal_step > fs_x + 1:
            x -= focal_step
    elif action == 4:
        if y - focal_step > fs_y + 1:
            y -= focal_step
    else:
        raise ValueError
    return (x, y)

def distort_frame(frame, config, focal_point, focal_size=None, blurr=None):
    x, y = focal_point
    fs_x, fs_y = focal_size if focal_size is not None else config.focal_size
    blurr = blurr if blurr is not None else config.blurr
    x0, x1 = max(0, x - fs_x), x + fs_x + 1
    y0, y1 = max(0, y - fs_y), y + fs_y + 1
    if blurr:
        frame_out = cv2.blur(frame, (blurr, blurr), cv2.BORDER_DEFAULT)
        frame_out[x0:x1, y0:y1] = frame[x0:x1, y0:y1]
    else:
        if clip:
            frame_out = frame[x0:x1, y0:y1]
        else:
            frame_out = np.zeros_like(frame)
            frame_out[x0:x1, y0:y1] = frame[x0:x1, y0:y1]
    return frame_out

def display_focal_area(frame, config, focal_point, focal_size=None, cmap='gray'):
    frame = frame.copy()
    x, y = focal_point
    fs_x, fs_y = focal_size if focal_size is not None else config.focal_size
    if cmap == 'gray':
        frame[x - fs_x, y - fs_y:y + fs_y + 1] = 255
        frame[x + fs_x, y - fs_y:y + fs_y + 1] = 255
        frame[x - fs_x:x + fs_x + 1, y - fs_y] = 255
        frame[x - fs_x:x + fs_x + 1, y + fs_y] = 255
    elif cmap == 'RGB':
        colors = [255, 255, 255]
        for n, color in enumerate(colors):
            frame[x - fs_x, y - fs_y:y + fs_y + 1, n] = color
            frame[x + fs_x, y - fs_y:y + fs_y + 1, n] = color
            frame[x - fs_x:x + fs_x + 1, y - fs_y, n] = color
            frame[x - fs_x:x + fs_x + 1, y + fs_y, n] = color
    return frame

def upscale_point(point, shape=(84, 84), upscaled_shape=(210, 160)):
    wf = upscaled_shape[0] / shape[0]
    hf = upscaled_shape[1] / shape[1]
    x, y = point
    upscaled_point = (round(x * wf), round(y * hf))
    return upscaled_point

if __name__ == '__main__':
    pass
