import numpy as np

def to_rle(mask):
    """
    Convert mask (as 2-D numpy array) to run-length encoding
    """
    rle = ''
    y_pos, x_pos = np.where(mask == 1)
    pixel_nums = []
    for i in range(y_pos.shape[0]):
        pixel_nums.append(x_pos[i] * mask.shape[0] + y_pos[i] + 1)
    pixel_nums_sorted = sorted(pixel_nums)
    run_start = np.where(np.diff(pixel_nums_sorted) != 1)[0] + 1
    if len(pixel_nums_sorted) > 0:
        run_start = np.concatenate(([0], run_start))
    run_length = np.diff(run_start)
    if len(pixel_nums_sorted) > 0:
        run_length = np.concatenate((run_length, [len(pixel_nums_sorted) - run_start[-1]]))
    if run_start.shape[0] != run_length.shape[0]:
        raise Exception('run_start and run_length do not have the same length.')

    if run_start.shape[0] > 0:
        rle += '{} {}'.format(pixel_nums_sorted[run_start[0]], run_length[0])
    for i in range(1, run_start.shape[0]):
        rle += ' {} {}'.format(pixel_nums_sorted[run_start[i]], run_length[i])
    return rle

