import os


if __name__ == '__main__':
    gt_root = '../../../datasets/virat/gt_annotations'
    frames_root = '../../../datasets/virat/frames'

    src_video_folders = os.listdir(gt_root)
    # print src_video_folders
    dst_video_folders = set(os.listdir(frames_root))

    for video_folder in src_video_folders:
        # print video_folder
        dst_video_folders.remove(video_folder)

    for video_folder in dst_video_folders:
        video_folder_path = os.path.join(frames_root, video_folder)

        cmd = 'rm -rf {}'.format(video_folder_path)
        print cmd
        os.system(cmd)

