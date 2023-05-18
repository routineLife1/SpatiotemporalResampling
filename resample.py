import os
import cv2
import torch
from tqdm import tqdm
import warnings
import _thread
from queue import Queue
from scdet import TransitionDetectionBase
from models.model_union_2.RIFE import Model

warnings.filterwarnings("ignore")
torch.set_grad_enabled(False)

video = r'E:\Video\01.mkv'  # input video path
save = r'E:\Work\VFI\Algorithm\GMFwSS\output'  # output img sequence path
scale = 1.0  # flow scale(suggest 1.0)
global_size = (1920,1080)  # output resolution
times = 8 # output_fps = (input_fps / 2) * times
scene_det = False  # enable scene change detection


class TransitionDetection(TransitionDetectionBase):
    def save_scene(self, title):
        pass

scene_detector = TransitionDetection(8, scdet_threshold=12,
                                     no_scdet=False,
                                     use_fixed_scdet=False,
                                     fixed_max_scdet=80)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
model = Model()
if not hasattr(model, 'version'):
    model.version = 0
model.load_model('train_logs/v', -1)
print("Loaded model")
model.eval()
model.device()


def to_tensor(img):
    return torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float().cuda() / 255.


def load_image(img, _scale):
    h, w, _ = img.shape
    while h * _scale % 64 != 0:
        h += 1
    while w * _scale % 64 != 0:
        w += 1
    img = cv2.resize(img, (w, h))
    img = to_tensor(img)
    return img


def get(): 
    return read_buffer.get()


output_counter = 0 


def put(things):  
    global output_counter
    output_counter += 1
    write_buffer.put([output_counter, things])


def build_read_buffer(r_buffer, v):
    ret, __x = v.read()
    while ret:
        r_buffer.put(cv2.resize(__x, global_size))
        ret, __x = v.read()
    r_buffer.put(None)


def clear_write_buffer(w_buffer):
    while True:
        item = w_buffer.get()
        if item is None:
            break
        num = item[0]
        content = item[1]
        cv2.imwrite(os.path.join(save, "{:0>9d}.png".format(num)), cv2.resize(content, global_size))


video_capture = cv2.VideoCapture(video)
total_frames_count = video_capture.get(7)
read_buffer = Queue(maxsize=100)
write_buffer = Queue(maxsize=-1)
_thread.start_new_thread(build_read_buffer, (read_buffer, video_capture))
_thread.start_new_thread(clear_write_buffer, (write_buffer,))


t_step = 1 / (times - 1)
t_stamps = [t_step * i for i in range(1, times)]
pbar = tqdm(total=total_frames_count)


i0 = get()
for i in range(times - 1):
    put(i0)


i1 = get()
if scene_det and scene_detector.check_scene(i0, i1):
    x = [i0, True]
else:
    _i0, _i1 = load_image(i0, scale), load_image(i1, scale)
    reuse_things = model.reuse(_i0, _i1, scale)
    out = model.inference(_i0, _i1, reuse_things, 0.5)
    x = [out.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255., False]
pbar.update(2)

for i in range(2, int(total_frames_count), 2):
    put(x[0])
    i2 = get()
    if i2 is None:
        break
    r = scene_detector.check_scene(i1, i2) if scene_det else False
    i3 = get()
    if i3 is None:
        break
    if scene_det and scene_detector.check_scene(i2, i3):
        y = [i2, True]
    else:
        _i2, _i3 = load_image(i2, scale), load_image(i3, scale)
        reuse_things = model.reuse(_i2, _i3, scale)
        out = model.inference(_i2, _i3, reuse_things, 0.5)
        y = [out.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255., False]
    if x[1] or r:
        for a in range(times - 1):
            put(x[0])
    else:
        _x, _y = load_image(x[0], scale), load_image(y[0], scale)
        reuse_things = model.reuse(_x, _y, scale)
        for t in t_stamps:
            out = model.inference(_x, _y, reuse_things, t)
            put(out.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.)
    x = y
    i1 = i3
    pbar.update(2)
