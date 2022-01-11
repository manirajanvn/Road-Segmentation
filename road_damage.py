from fastai.basics import *
from fastai.vision.all import *
from fastai.callback.all import *
import cv2 as cv
from fastbook import *
import pathlib
import numpy

plt1 = platform.system()
if plt1 == 'Windows': pathlib.PosixPath = pathlib.WindowsPath

BASE_DIR = "D:/ws/dldata/data/road/road_damaged_download"
path = Path(BASE_DIR)
lbl_names = get_image_files(path / 'mask')


def n_codes(fnames, is_partial=True):
    "Gather the codes from a list of `fnames`"
    vals = set()
    if is_partial:
        random.shuffle(fnames)
        fnames = fnames[:10]
    for fname in fnames:
        msk = np.array(PILMask.create(fname))
        for val in np.unique(msk):
            if val not in vals:
                vals.add(val)
    vals = list(vals)
    p2c = dict()
    for i, val in enumerate(vals):
        p2c[i] = vals[i]
    return p2c


vals = n_codes(lbl_names)


def get_my_y(fname: Path):
    fn = path / 'mask' / f'{fname.stem}.png'
    msk = np.array(PILMask.create(fn))
    mx = np.max(msk)
    for i, val in enumerate(vals):
        msk[msk == vals[i]] = val
    return PILMask.create(msk)


learn_inference = load_learner("D:/ws/dldata/data/road/road_damaged_download/model/road_damage.pkl")

cap = cv.VideoCapture('D:/ws/dldata/data/road/road_damaged_download/test/road_test_4.mp4')
frame_size = (224, 224)
fps = 20
output = cv.VideoWriter('D:/ws/dldata/data/road/road_damaged_download/test/output_video4.avi', cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20, frame_size)
while cap.isOpened():
    _, img = cap.read()
    if img is None:
        break
    img = cv.resize(img, (224, 224))
    pred = learn_inference.predict(img)
    plt.imsave('D:/ws/dldata/data/road/road_damaged_download/test/Z1.png', pred[0])
    img2 = cv.imread('D:/ws/dldata/data/road/road_damaged_download/test/Z1.png')
    img2 = cv.resize(img2, (224, 224))

    result = cv.addWeighted(img, 0.7, img2, 0.3, 0)
    cv.imshow('img', result)
    output.write(result)
    if cv.waitKey(2) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()