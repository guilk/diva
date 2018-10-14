import os
import random
import cv2


def generate_colors(n=1):
    ret = []
    r = int(random.random() * 256)
    g = int(random.random() * 256)
    b = int(random.random() * 256)
    step = 256 / n
    for i in range(n):
        r += step
        g += step
        b += step
        r = int(r) % 256
        g = int(g) % 256
        b = int(b) % 256
        ret.append((r,g,b))
    return ret

def loading_tracking(tracking_video_path):
    tracking_data = {}
    with open(tracking_video_path, 'rb') as fr:
        lines = fr.readlines()
        for line in lines:
            line_data = [int(s) for s in line.split(',')]
            frame_id = line_data[0]
            object_id = line_data[1]
            x,y,w,h = line_data[2],line_data[3],line_data[4],line_data[5]
            if frame_id in tracking_data:
                tracking_data[frame_id].append((object_id, x, y, w, h))
            else:
                tracking_data[frame_id] = [(object_id, x, y, w, h)]
    return tracking_data

def visualize_tracking(src_frames_folder, dst_frames_folder, tracking_dict):
    color_dict = {}
    for counter,frame_id in enumerate(tracking_dict):
        print 'Process {}th of {} frames'.format(counter, len(tracking_dict))
        frame_name = '{}.jpg'.format(str(frame_id-1).zfill(5))
        frame_path = os.path.join(src_frames_folder, frame_name)
        frame = cv2.imread(frame_path)
        object_bboxes = tracking_dict[frame_id]
        for object_bbox in object_bboxes:
            object_id = object_bbox[0]
            x,y,w,h = object_bbox[1],object_bbox[2],object_bbox[3], object_bbox[4]
            if object_id not in color_dict:
                color_dict[object_id] = generate_colors()[0]
            bbox_color = color_dict[object_id]
            cv2.rectangle(frame, (x, y), (x + w, y + h), bbox_color, 2)
        dst_frame_path = os.path.join(dst_frames_folder, frame_name)
        cv2.imwrite(dst_frame_path, frame)

def copy_remaining_frames(src_frames_folder, dst_frames_folder):
    src_imgs = os.listdir(src_frames_folder)
    dst_imgs = os.listdir(dst_frames_folder)

    remaining_imgs = list(set(src_imgs) - set(dst_imgs))
    for img_name in remaining_imgs:
        src_img_path = os.path.join(src_frames_folder, img_name)
        dst_img_path = os.path.join(dst_frames_folder, img_name)
        cmd = 'scp {} {}'.format(src_img_path, dst_img_path)
        os.system(cmd)


def convert_frames_to_video(img_root, video_path, fps):
    frame_array = []
    files = [f for f in os.listdir(img_root) if os.path.exists(os.path.join(img_root, f))]
    img_files = sorted(files)
    img_path = os.path.join(img_root, img_files[0])
    img = cv2.imread(img_path)
    height, width, layers = img.shape
    size = (width, height)

    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    for index, img_file in enumerate(img_files):
        print 'Write {}th of {} images'.format(index, len(img_files))
        img_path = os.path.join(img_root, img_file)
        img = cv2.imread(img_path)
        out.write(img)
    out.release()

if __name__ == '__main__':
    tracking_root = '/home/liangke/diva/datasets/virat/offline'
    img_root = '/home/liangke/diva/datasets/virat/frames'
    dst_root = '/home/liangke/diva/datasets/virat/tracking_frames'
    video_root = '/home/liangke/diva/datasets/virat/tracking_videos'
    tracking_types = ['car', 'person']
    video_folder = 'VIRAT_S_050101_02_000400_000470'

    # videos = ['VIRAT_S_040103_00_000000_000120.mp4', 'VIRAT_S_000204_04_000738_000977.mp4',
    #           'VIRAT_S_040003_04_000758_001118.mp4', 'VIRAT_S_050101_02_000400_000470.mp4']


    tracking_type = 'car'
    tracking_video_path = os.path.join(tracking_root, tracking_type, '{}.txt'.format(video_folder))
    src_frames_folder = os.path.join(img_root, video_folder)
    dst_frames_folder = os.path.join(dst_root, video_folder)
    if not os.path.exists(dst_frames_folder):
        os.makedirs(dst_frames_folder)
    tracking_dict = loading_tracking(tracking_video_path)
    visualize_tracking(src_frames_folder, dst_frames_folder, tracking_dict)

    tracking_type = 'person'
    tracking_video_path = os.path.join(tracking_root, tracking_type, '{}.txt'.format(video_folder))
    src_frames_folder = os.path.join(dst_root, video_folder)
    dst_frames_folder = os.path.join(dst_root, video_folder)
    if not os.path.exists(dst_frames_folder):
        os.makedirs(dst_frames_folder)
    tracking_dict = loading_tracking(tracking_video_path)
    visualize_tracking(src_frames_folder, dst_frames_folder, tracking_dict)

    src_frames_folder = os.path.join(img_root, video_folder)
    dst_frames_folder = os.path.join(dst_root, video_folder)
    copy_remaining_frames(src_frames_folder, dst_frames_folder)

    frames_root = os.path.join(dst_root, video_folder)
    video_path = os.path.join(video_root, '{}.avi'.format(video_folder))
    fps = 25.0
    convert_frames_to_video(frames_root, video_path, fps)

