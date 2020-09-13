
try:
    import os
    import json
    import glob
    import argparse

    import numpy as np
    from scipy import signal as sg
    import scipy.ndimage as ndimage
    from scipy.ndimage.filters import maximum_filter
    from PIL import Image
    import matplotlib.pyplot as plt

except ImportError:
    print("Need to fix the installation")
    raise


def find_max_coords(image):
    d_x, d_y = 18, 18
    M, N = image.shape
    coord_x = []
    coord_y = []

    for x in range(0, M-d_x+1, d_x):

        for y in range(0, N-d_y+1, d_y):
            window = image[x:x+d_x, y:y+d_y]
            local_max = np.amax(window)
            max_coord = np.argmax(window)

            if local_max > 80:
                coord_x.append(x + max_coord // d_x)
                coord_y.append(y + max_coord % d_x)

    return coord_x, coord_y


def find_img2d_candidates(image, **kwargs):
    """
    Detect candidates for TFL lights. Use c_image, kwargs and you imagination to implement
    :param c_image: The image itself as np.uint8, shape of (H, W, 3)
    :param kwargs: Whatever config you want to pass in here
    :return: 4-tuple of x_red, y_red, x_green, y_green
    """

    # filter_kernel = np.array([[-1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225],
    #                  [-1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225],
    #                  [-1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225],
    #                  [-1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225],
    #                  [-1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225],
    #                  [-1/225, -1/225, -1/225, -1/225, -1/225, 8/225, 8/225, 8/225, 8/225, 8/225, -1/225, -1/225, -1/225, -1/225, -1/225],
    #                  [-1/225, -1/225, -1/225, -1/225, -1/225, 8/225, 8/225, 8/225, 8/225, 8/225, -1/225, -1/225, -1/225, -1/225, -1/225],
    #                  [-1/225, -1/225, -1/225, -1/225, -1/225, 8/225, 8/225, 8/225, 8/225, 8/225, -1/225, -1/225, -1/225, -1/225, -1/225],
    #                  [-1/225, -1/225, -1/225, -1/225, -1/225, 8/225, 8/225, 8/225, 8/225, 8/225, -1/225, -1/225, -1/225, -1/225, -1/225],
    #                  [-1/225, -1/225, -1/225, -1/225, -1/225, 8/225, 8/225, 8/225, 8/225, 8/225, -1/225, -1/225, -1/225, -1/225, -1/225],
    #                  [-1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225],
    #                  [-1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225],
    #                  [-1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225],
    #                  [-1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225],
    #                  [-1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225, -1/225]])

    filter_kernel = np.array([[-2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2/324, -2/324, -2/324],
                       [-2 / 324, -2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1/324, -2/324, -2/324],
                       [-2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1/324, -1/324, -2/324],
                       [-2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1/324, -1/324, -2/324],
                       [-2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -2 / 324, -2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -2 / 324, -1 / 324, -1 / 324, -1/324, -1/324, -2/324],
                       [-2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -1 / 324, -1 / 324, -1/324, -1/324, -2/324],
                       [-2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -2 / 324, 11 / 324, 11 / 324, 11 / 324, 11 / 324, 11 / 324, 11 / 324, -2 / 324, -1 / 324, -1 / 324, -1/324, -1/324, -2/324],
                       [-2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -2 / 324, 11 / 324, 11 / 324, 11 / 324, 11 / 324, 11 / 324, 11 / 324, -2 / 324, -1 / 324, -1 / 324, -1/324, -1/324, -2/324],
                       [-2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -2 / 324, 11 / 324, 11 / 324, 11 / 324, 11 / 324, 11 / 324, 11 / 324, -2 / 324, -1 / 324, -1 / 324, -1/324, -1/324, -2/324],
                       [-2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -2 / 324, 11 / 324, 11 / 324, 11 / 324, 11 / 324, 11 / 324, 11 / 324, -2 / 324, -1 / 324, -1 / 324, -1/324, -1/324, -2/324],
                       [-2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -2 / 324, 11 / 324, 11 / 324, 11 / 324, 11 / 324, 11 / 324, 11 / 324, -2 / 324, -1 / 324, -1 / 324, -1/324, -1/324, -2/324],
                       [-2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -2 / 324, 11 / 324, 11 / 324, 11 / 324, 11 / 324, 11 / 324, 11 / 324, -2 / 324, -1 / 324, -1 / 324, -1/324, -1/324, -2/324],
                       [-2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -1 / 324, -1 / 324, -1/324, -1/324, -2/324],
                       [-2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -2 / 324, -2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -2 / 324, -2 / 324, -1 / 324, -1 / 324, -1/324, -1/324, -2/324],
                       [-2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1/324, -1/324, -2/324],
                       [-2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1/324, -1/324, -2/324],
                       [-2 / 324, -2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1/324, -2/324, -2/324],
                       [-2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2/324, -2/324, -2/324]])

    res = sg.convolve2d(image, filter_kernel, mode='same', boundary='fill', fillvalue=0)
    coord_x, coord_y = find_max_coords(np.absolute(res))

    return coord_x, coord_y


def find_tfl_lights(image):
    red_x, red_y = find_img2d_candidates(image[:, :, 0], some_threshold=42)
    green_x,green_y = find_img2d_candidates(image[:, :, 1],  some_threshold=42)

    return red_x, red_y, green_x, green_y


def show_image_and_gt(image, objs, fig_num=None):
    plt.figure(fig_num).clf()
    plt.imshow(image)
    labels = set()

    if objs is not None:

        for o in objs:
            poly = np.array(o['polygon'])[list(np.arange(len(o['polygon']))) + [0]]
            plt.plot(poly[:, 0], poly[:, 1], 'r', label=o['label'])
            labels.add(o['label'])

        if len(labels) > 1:
            plt.legend()


def test_find_tfl_lights(image_path, json_path=None, fig_num=None):
    """
    Run the attention code
    """
    image = np.array(Image.open(image_path))

    if json_path is None:
        objects = None
    else:
        gt_data = json.load(open(json_path))
        what = ['traffic light']
        objects = [o for o in gt_data['objects'] if o['label'] in what]

    find_tfl_lights(image)

  
def main(argv=None):
    """It's nice to have a standalone tester for the algorithm.
    Consider looping over some images from here, so you can manually exmine the results
    Keep this functionality even after you have all system running, because you sometime want to debug/improve a module
    :param argv: In case you want to programmatically run this"""
    parser = argparse.ArgumentParser("Test TFL attention mechanism")
    parser.add_argument('-i', '--image', type=str, help='Path to an image')
    parser.add_argument("-j", "--json", type=str, help="Path to json GT for comparison")
    parser.add_argument('-d', '--dir', type=str, help='Directory to scan images in')
    args = parser.parse_args(argv)
    default_base = '../data'
    if args.dir is None:
        args.dir = default_base
    flist = glob.glob(os.path.join(args.dir, '*_leftImg8bit.png'))
    for image in flist:
        json_fn = image.replace('_leftImg8bit.png', '_gtFine_polygons.json')
        if not os.path.exists(json_fn):
            json_fn = None
        test_find_tfl_lights(image, json_fn)
    if len(flist):
        print("You should now see some images, with the ground truth marked on them. Close all to quit.")
    else:
        print("Bad configuration?? Didn't find any picture to show")
    plt.show(block=True)


if __name__ == '__main__':
    main()
