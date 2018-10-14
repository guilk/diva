import os
import cv2
import numpy as np

# def convert_frames_to_video(img_root, video_path, fps):
#     frame_array = []
#     files = [f for f in os.listdir(img_root) if os.path.exists(os.path.join(img_root, f))]
#     img_files = sorted(files)
#     img_path = os.path.join(img_root, img_files[0])
#     img = cv2.imread(img_path)
#     height, width, layers = img.shape
#     size = (width, height)
#
#     out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
#
#     for index, img_file in enumerate(img_files):
#         print 'Write {}th of {} images'.format(index, len(img_files))
#         img_path = os.path.join(img_root, img_file)
#         img = cv2.imread(img_path)
#         out.write(img)
#         # height, width, layers = img.shape
#         # size = (width, height)
#
#         # frame_array.append(img)
#         # if index > 10:
#         #     break
#
#
#     out.release()
#
#     # for i in range(len(frame_array)):
#     #     out.write(frame_array[i])

def convert_frames_to_video(refined_mask_root, visualized_root, dst_path, fps):
    files = [f for f in os.listdir(refined_mask_root) if os.path.exists(os.path.join(refined_mask_root, f))]
    img_files = sorted(files)
    img_path = os.path.join(refined_mask_root, img_files[0])
    # print img_path
    img = cv2.imread(img_path)
    height, width, layers = img.shape
    size = (width, height/2)

    out = cv2.VideoWriter(dst_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    for index, img_file in enumerate(img_files):
        print 'Write {}th of {} images'.format(index, len(img_files))
        img_path = os.path.join(visualized_root, img_file)
        visualized_frame = cv2.imread(img_path)
        img_path = os.path.join(refined_mask_root, img_file)
        refined_mask_frame = cv2.imread(img_path)
        visualized_frame = cv2.resize(visualized_frame, (width / 2, height / 2))
        refined_mask_frame = cv2.resize(refined_mask_frame, (width / 2, height / 2))
        img = np.concatenate((visualized_frame, refined_mask_frame), axis=1)
        out.write(img)
    out.release()

if __name__ == '__main__':

    # video_folders = ['VIRAT_S_000204_04_000738_000977']

    # video_folders = ['VIRAT_S_000204_04_000738_000977', 'VIRAT_S_000205_02_000409_000566', 'VIRAT_S_000207_02_000498_000530',
    #           'VIRAT_S_040003_04_000758_001118', 'VIRAT_S_040103_00_000000_000120', 'VIRAT_S_040104_05_000939_001116']
    # videos = ['VIRAT_S_040103_00_000000_000120']
    # videos = ['VIRAT_S_040103_00_000000_000120', 'VIRAT_S_000204_04_000738_000977', 'VIRAT_S_040003_04_000758_001118']
    videos = ['VIRAT_S_040003_04_000758_001118']


    # img_root = '../../../datasets/virat/resized_frames/VIRAT_S_040103_00_000000_000120'
    # img_root = '/home/liangke/diva/datasets/virat/visualized_frames/VIRAT_S_040103_00_000000_000120'
    # img_root = '/home/liangke/diva/datasets/virat/refined_masks/VIRAT_S_040103_00_000000_000120'

    refined_mask_root = '/home/liangke/diva/datasets/virat/refined_masks/'
    visualized_root = '/home/liangke/diva/datasets/virat/visualized_frames/'
    dst_root = '/home/liangke/diva/datasets/virat/compared_videos/'
    if not os.path.exists(dst_root):
        os.makedirs(dst_root)
    fps = 25.0

    for video_folder in videos:
        dst_path = os.path.join(dst_root, '{}.avi'.format(video_folder))
        refined_mask_folder = os.path.join(refined_mask_root, video_folder)
        visualized_folder = os.path.join(visualized_root, video_folder)
        convert_frames_to_video(refined_mask_folder, visualized_folder, dst_path, fps)





    # img_root = '../../../datasets/virat/visualized_frames/VIRAT_S_000204_04_000738_000977'
    # img_root = '../../../datasets/virat/visualized_frames/VIRAT_S_050101_02_000400_000470'
    # video_path = './mask_VIRAT_S_040103_00_000000_000120.avi'
    #
    #
    #
    #
    # fps = 25.0
    # convert_frames_to_video(img_root, video_path, fps)