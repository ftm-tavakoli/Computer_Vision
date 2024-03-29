import copy
import cv2
import numpy as np


def blur_Canny(image):
    blurred = cv2.blur(image, (3, 3))
    image_edges = cv2.Canny(blurred, 100, 200, apertureSize=3, L2gradient=True)
    return image_edges


def make_mask(image, point):
    contours = np.array([[0, image.shape[0]], point, [image.shape[1], image.shape[0]]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, pts=[contours], color=255)
    return mask


def average_slope_intercept(lines):
    """
    Find the slope and intercept of the left and right lanes of each image.
        Parameters:
            lines: The output lines from Hough Transform.
    """
    left_lines = []  # (slope, intercept)
    left_weights = []  # (length,)
    right_lines = []  # (slope, intercept)
    right_weights = []  # (length,)

    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2:
                continue
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - (slope * x1)
            length = np.sqrt(((y2 - y1) ** 2) + ((x2 - x1) ** 2))
            if slope < 0:
                left_lines.append((slope, intercept))
                left_weights.append((length))
            else:
                right_lines.append((slope, intercept))
                right_weights.append((length))
    left_lane = np.dot(left_weights, left_lines) / np.sum(left_weights) if len(left_weights) > 0 else None
    right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if len(right_weights) > 0 else None
    return left_lane, right_lane


def pixel_points(y1, y2, line):
    """
    Converts the slope and intercept of each line into pixel points.
        Parameters:
            y1: y-value of the line's starting point.
            y2: y-value of the line's end point.
            line: The slope and intercept of the line.
    """
    if line is None:
        return None
    slope, intercept = line
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    y1 = int(y1)
    y2 = int(y2)
    return np.array([[x1, y1, x2, y2],])
    # return (x1, y1), (x2, y2)


def lane_lines(image, lines):
    """
    Create full lenght lines from pixel points.
        Parameters:
            image: The input test image.
            lines: The output lines from Hough Transform.
    """
    left_lane, right_lane = average_slope_intercept(lines)
    y1 = image.shape[0]
    y2 = y1 * 0.6
    left_line = pixel_points(y1, y2, left_lane)
    right_line = pixel_points(y1, y2, right_lane)
    return np.array([left_line, right_line])


def draw_lane_lines(image, lines, color=[255, 0, 0], thickness=5):
    """
    Draw lines onto the input image.
        Parameters:
            image: The input test image.
            lines: The output lines from Hough Transform.
            color (Default = red): Line color.
            thickness (Default = 12): Line thickness.
    """

    if lines is not None:
        for i in range(0, len(lines)):
            if lines[i] is not None:
                l = lines[i][0]
                cv2.line(image, (l[0], l[1]), (l[2], l[3]), color, thickness)

    return image

if __name__ == "__main__":

    img1 = cv2.imread('img1.jpg')

    # part 1: make edge image
    img1_edges= blur_Canny(img1)
    cv2.imwrite('image1.jpg', img1_edges)
    cv2.waitKey(0)


    # part 2: make mask
    mask = make_mask(image=img1_edges, point=[950, 900])
    cv2.imwrite("Mask.jpg", mask)
    cv2.waitKey(0)

    # part 3: detect lines
    linesP = cv2.HoughLinesP(img1_edges * mask, 1, np.pi / 180, 100, None, 20, 10)

    img1_multiline= draw_lane_lines(image=copy.deepcopy(img1), lines=linesP,color= (0, 0, 255))
    cv2.imwrite("multi_line.jpg", img1_multiline)
    cv2.waitKey(0)


    final_result = draw_lane_lines(image=copy.deepcopy(img1), lines=lane_lines(img1, linesP))
    cv2.imwrite("final_result.jpg", final_result)
    cv2.waitKey(0)
