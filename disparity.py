rom posixpath import join
import cv2
import numpy as np

def distance(x, y):
    return abs(x - y)

def pixel_wise_matching(left_img, right_img, disparity_range, save_result=True):
    left = cv2.imread(left_img, 0)
    right = cv2.imread(right_img, 0)
    left = left . astype (np. float32 )
    right = right . astype (np. float32 )

     height , width = left . shape [:2]

 # Create blank disparity map
    depth = np. zeros (( height , width ) , np. uint8 )
    scale = 16
    max_value = 255

    for y in range(height):
        for x in range(width):

            disparity = 0
            cost_min = max_value

            for j in range(disparity_range):
                cost = max_value if (x - j) < 0 else distance(int(left [y , x]), int(right [y , x - j]))
                if cost < cost_min:
                    cost_min = cost
                    disparity = j

            depth [y , x] = disparity * scale

    if save_result == True:
        print('Saving result...')
        #Save result
        cv2.imwrite(f'pixel_wise_l1.png', depth)
        cv2.imwrite(f'pixel_wise_l1_color.png', cv2.applyColorMap(depth, cv2.COLORMAP_JET))

        print('Done.')
    return depth

left_img_path = 'tsukuba/left.png'
right_img_path = 'tsukuba/right.png '
disparity_range = 16

pixel_wise_result_l1 = pixel_wise_matching_l1(
    left_img_path ,
    right_img_path ,
    disparity_range ,
    save_result = True
    )

# Saving result ...
# Done .

pixel_wise_result_l2 = pixel_wise_matching_l2 (
    left_img_path ,
    right_img_path ,
    disparity_range ,
    save_result = True
)

# Saving result ...
# Done.

import cv2
import numpy as np

def distance(x, y):
    return abs(x - y)

def window_based_matching(left_img, right_img, disparity_range, kernel_size =5,
                          save_result= True):
    left = cv2.imread(left_img, 0)
    right = cv2.imread(right_img, 0)

    left = left . astype (np.float32 )
    right = right . astype (np.float32 )

    height , width = left.shape[:2]


    depth = np.zeros((height,width ),np.uint8)

    kernel_half = int (( kernel_size - 1) / 2)
    scale = 3
    max_value = 255 * 9

    for y in range(kernel_half ,height - kernel_half ) :
        for x in range ( kernel_half , width - kernel_half ) :
            disparity = 0
            cost_min = 65534

            for j in range ( disparity_range ) :
                total = 0
                value = 0

                for v in range ( - kernel_half , kernel_half + 1) :
                    for u in range ( - kernel_half , kernel_half + 1) :
                        value = max_value
                        if (x + u - j) >= 0:
                            value = distance(
                                int(left[y + v, x + u]) , int(right[y + v, (x +
u)-j]))
                            total += value
                if total < cost_min:
                    cost_min = total
                    disparity = j
            depth[y, x] = disparity * scale

    if save_result == True:
        print('Saving result...')
        cv2.imwrite(f'window_based_l1.png', depth)
        cv2.imwrite(f'window_based_l1_color.png', cv2.applyColorMap(depth, cv2.COLORMAP_JET))

        print('Done.')
    return depth

left_img_path = 'ALoe/Aloe_left_1.png'
right_img_path = 'ALoe/Aloe_right_1.png'
disparity_range = 64
kernel_size = 3

window_based_result = window_based_matching_l1(
    left_img_path,
    right_img_path,
    disparity_range,
    kernel_size = kernel_size,
    save_result = True
)
#Saving result...
#Done

window_based_result = window_based_matching_l2(
    ledt_img_path,
    right_img_path,
    disparity_range,
    kernel_size= kernel_size,
    save_result = True
)

from posixpath import join
import cv2
import numpy as np

def cosine_similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

def window_based_matching(left_img, right_img, disparity_range, kernel_size = 5, save_result = True):
    left = cv2.imread(left_img, 0)
    right = cv2.imread(right_img, 0)

    left = left.astype(np.float32)
    right = right.astype(np.float32)

    height, width = left.shape[:2]

    depth = np.zeros((height, width), np.uint8)

    kernel_half = int((kernel_size - 1) / 2)
    scale = 3

    height, width = left.shape[:2]

    depth = np.zeros((height, width), np.unit8)
    kernel_half = int((kernel_size - 1) / 2)
    scale = 3

    for y in range(kernel_half, height - kernel_half):
        for x in range(kernel_half, width - kernel_half):
            disparity = 0
            cost_optimal = -1

            for j in range (disparity_range):
                d = x- j
                cost = -1
                if (d- kernel_half) > 0:
                    wp = left[(y - kernel_half):(y + kernel_half + 1), (x - kernel_half):(x + kernel_half + 1)]
                    wqd = right[(y - kernel_half):(y + kernel_half + 1), (d - kernel_half):(d + kernel_half + 1)]

                    wp_flattened  = wp.flatten()
                    wqd_flattened = wqd.flatten()

                    cost = cosine_similarity(wp_flattened, wqd_flattened)

                if cost > cost_optimal:
                    cost_optimal = cost
                    disparity = j

            depth[y, x] = disparity * scale

    if save_result == True:
        print('Saving result...')
        cv2.imwrite(f'window_based_cosine_similarity.png', depth)
        cv2.imwrite(f'window_based_cosine_similarity_color.png', cv2.applyColorMap(depth, cv2.COLORMAP_JET))

    print('Done.')
    return depth