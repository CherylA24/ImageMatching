# Import Necessary Library
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Load Images
image_scene = cv.imread("bigbenscene.jpeg")
image = cv.imread("bigben.jpg")

"""
STEP 1 : Finding Key Points (Feature Extraction)
"""
# In this Code, For finding keypoints of the images, i will used Open-CV library of SIFT and SURF

# Finding key points with SIFT 
sift_obj = cv.xfeatures2d.SIFT_create()

sift_image_keypoint, sift_image_descriptor = sift_obj.detectAndCompute(image, None)
sift_image_scene_keypoint, sift_image_scene_descriptor = sift_obj.detectAndCompute(image_scene, None)

si = cv.drawKeypoints(image_scene, sift_image_scene_keypoint, None)

# Finding Key Points with SURF 
surf_obj = cv.xfeatures2d.SURF_create()

surf_image_keypoint, surf_image_descriptor = surf_obj.detectAndCompute(image, None)
surf_image_scene_keypoint, surf_image_scene_descriptor = surf_obj.detectAndCompute(image_scene, None)

su = cv.drawKeypoints(image_scene, surf_image_scene_keypoint, None)

"""
STEP 2: Feature Matching (FLANN & BF-Matchers)
"""
# FLANN

KDINDEX = 0 #algorithm
TREE_CHECKS = 50 # Traverse tree

flann = cv.FlannBasedMatcher(dict(algorithm = KDINDEX), dict(checks = TREE_CHECKS))

# Flann Matching with SIFT
sift_match = flann.knnMatch(sift_image_descriptor, sift_image_scene_descriptor, 2)

sift_matchesMask = []
for i in range(0, len(sift_match)):
    sift_matchesMask.append([0,0])

for i, (bm, sbm) in enumerate(sift_match):
    if (bm.distance < 0.7 * sbm.distance):
        sift_matchesMask[i] = [1,0]

flann_sift_res = cv.drawMatchesKnn(
    image, sift_image_keypoint,
    image_scene, sift_image_scene_keypoint,
    sift_match, None,
    matchColor = [0, 255, 0], 
    singlePointColor= [255, 0, 0],
    matchesMask= sift_matchesMask
)

# Flann Matching with SURF
surf_match = flann.knnMatch(surf_image_descriptor, surf_image_scene_descriptor, 2)
surf_matchesMask = []
for i in range(0, len(surf_match)):
    surf_matchesMask.append([0,0])

for i, (bm, sbm) in enumerate(surf_match):
    if (bm.distance < 0.7 * sbm.distance):
        surf_matchesMask[i] = [1,0]

flann_surf_res = cv.drawMatchesKnn(
    image, surf_image_keypoint,
    image_scene, surf_image_scene_keypoint,
    surf_match, None,
    matchColor = [0, 255, 0], 
    singlePointColor= [255, 0, 0],
    matchesMask= surf_matchesMask
)

# BF MATCHERS

bf = cv.BFMatcher(cv.NORM_L1, False)
sift_matches = bf.match(sift_image_descriptor, sift_image_scene_descriptor)
sift_matches = sorted(sift_matches, key = lambda x:x.distance)
bf_sift_res = cv.drawMatches(
    image, sift_image_keypoint, 
    image_scene, sift_image_scene_keypoint,
    sift_matches[:50], 2)

surf_matches = bf.match(surf_image_descriptor, surf_image_scene_descriptor)
surf_matches = sorted(surf_matches, key = lambda x:x.distance)
bf_surf_res = cv.drawMatches(
    image, surf_image_keypoint, 
    image_scene, surf_image_scene_keypoint,
    surf_matches[:50], 2)


# Show FLANN Based-Matcher and BF Matching for SIFT 
plt.subplot(2, 2, 1)
plt.imshow(flann_sift_res)
plt.title('FLANN SIFT')
plt.xticks([])
plt.yticks([])

plt.subplot(2, 2, 2)
plt.imshow(bf_sift_res)
plt.title('BF MATCHING SIFT')
plt.xticks([])
plt.yticks([])

plt.subplot(2, 2, 3)
plt.imshow(flann_surf_res)
plt.title('FLANN SURF')
plt.xticks([])
plt.yticks([])

plt.subplot(2, 2, 4)
plt.imshow(bf_surf_res)
plt.title('BF MATCHING SURF')
plt.xticks([])
plt.yticks([])


plt.show()