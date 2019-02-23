import cv2
import numpy as np

def compare_image(img1,img2):
    original = cv2.imread(img1)
    compared = cv2.imread(img2)

    sift = cv2.xfeatures2d.SIFT()
    kp_1, desc_1 = sift.detectAndCompute(original, None)
    kp_2, desc_2 = sift.detectAndCompute(compared, None)

    index_params = {'algorithm': 5, 'trees': 5}
    search_params = {}
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(desc_1, desc_2, k=2)

    good_points = []
    for m, n in matches:
        if m.distance < 0.6 * n.distance:
            good_points.append(m)


    number_keypoints = 0
    if len(kp_1) <= len(kp_2):
        number_keypoints = len(kp_1)
    else:
        number_keypoints = len(kp_2)

    print("Keypoints 1st image: " + str(len(kp_1)))
    print("Keypoints 2nd Image: " + str(len(kp_2)))
    print("Good Matches:", len(good_points))
    percentage_match = (len(good_points) / number_keypoints) * 100

    result = cv2.drawMatches(original, kp_1, image_to_compare, kp_2, good_points,
                             None)

    cv2.imshow("result", cv2.resize(result, None, fx=0.4, fy=0.4))
    cv2.imwrite("feature_matching.jpg", result)

    cv2.imshow("Original", cv2.resize(original, None, fx=0.4, fy=0.4))
    cv2.imshow("Duplicate", cv2.resize(compared, None, fx=0.4, fy=0.4))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


print(compare_image("clooney1.jpg","clooney2.jpg"))


