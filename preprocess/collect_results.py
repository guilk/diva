import os


if __name__ == '__main__':
    result_root = '../../../output/'
    dst_root = '../grid_search_results'
    if not os.path.exists(dst_root):
        os.makedirs(dst_root)
    folders = os.listdir(result_root)
    for folder in folders:
        folder_path = os.path.join(result_root, folder)
        dst_path = os.path.join(dst_root, folder)
        # print dst_path
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)

        img_files = [file for file in os.listdir(folder_path) if file.endswith('.png')]
        for img_file in img_files:
            src_result_path = os.path.join(folder_path, img_file)
            dst_result_path = os.path.join(dst_path, '.')
            cmd = 'scp {} {}'.format(src_result_path, dst_result_path)

            os.system(cmd)
        # print src_result_path
        # print dst_result_path