import cv2 as cv
import numpy as np
import random
from matplotlib import pyplot as plt
import math

def ComputeHomography(world_coordinates,image_coordinates):
    A=np.empty((0,9), float)
    for i in range(len(world_coordinates)):
        x_dash=image_coordinates[i][0]
        y_dash=image_coordinates[i][1]
        x=world_coordinates[i][0]
        y=world_coordinates[i][1]
        A = np.append(A, [np.array([x,y,1,0,0,0,-x_dash*x,-x_dash*y,-x_dash])], 0)
        A = np.append(A, [np.array([0,0,0,x,y,1,-y_dash*x,-y_dash*y,-y_dash])], 0)
    
    A_transpose=A.T
    Mat=A_transpose.dot(A)
    eigVal, eigVec=np.linalg.eig(Mat)
    index_smallest_eigenVal=np.argmin(eigVal)
    solution=eigVec[:,index_smallest_eigenVal]

    HomographyMat=np.array([[solution[0],solution[1],solution[2]],[solution[3],solution[4],solution[5]],[solution[6],solution[7],solution[8]]])
    return HomographyMat

def IsPointAnInlier(src_pt,des_pt,H,threshold=0.02):
    is_inlier=False
    transformed_src = np.dot(H, src_pt)
    error=np.sqrt((des_pt[0]-transformed_src[0])**2+(des_pt[1]-transformed_src[1]))
    if error <= threshold:
        is_inlier=True
    return is_inlier


def ComputeHomography_RANSAC(source_pts,destination_pts):
    # Get the points in required format
    src_pts=np.empty((0,3), float)
    des_pts=np.empty((0,3), float)

    for i in range(len(source_pts)):
        src_pts = np.append(src_pts, [np.array([source_pts[i][0][0],source_pts[i][0][1],1])], 0)
        des_pts = np.append(des_pts, [np.array([destination_pts[i][0][0],destination_pts[i][0][1],1])], 0)
    
    N=100000000000000000000000000000000
    p=0.99
    sample_count=0
    num_inliers=[]
    H_values=[]


    while N> sample_count:
        # Choose 4 point matches randomly
        sample_indices=[]
        while len(sample_indices)<4:
            d = random.randrange(source_pts.shape[0]-1)
            if not(id in sample_indices):
                sample_indices.append(id)
    
        # Find homography using 4 matches
        world_coordinates=[]
        image_coordinates=[]
        for i in range(len(sample_indices)):
            world_coordinates.append((src_pts[sample_indices[i]][0],src_pts[sample_indices[i]][1]))
            image_coordinates.append((des_pts[sample_indices[i]][0],des_pts[sample_indices[i]][1]))
        
        H=ComputeHomography(world_coordinates,image_coordinates)

        inlier_count=0
        for i in range(len(src_pts)):
            if (IsPointAnInlier(src_pts[0],des_pts[0],H)):
                inlier_count+=1

        num_inliers.append(inlier_count)
        H_values.append(H)

        e=1-(inlier_count/len(src_pts))

        N=math.log(1-p)/math.log(1-(1-e)*(1-e)*(1-e))

        sample_count+=1

    sol_index=num_inliers.index(max(num_inliers))
    final_H=H_values[sol_index]
    return final_H

    
def StitchTwoImages(image1,image2):
    grayimage1 = cv.cvtColor(image1,cv.COLOR_BGR2GRAY)
    grayimage2 = cv.cvtColor(image2,cv.COLOR_BGR2GRAY)

    # sift feature detector
    sift=cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(grayimage1, None)
    kp2, des2 = sift.detectAndCompute(grayimage2, None)

    flannMatcher=cv.FlannBasedMatcher()
    matches=flannMatcher.knnMatch(des1,des2,k=2)

    good_matches=[]
  
    for m, n in matches:
        #append the points according
        #to distance of descriptors
        if(m.distance < 0.6*n.distance):
            good_matches.append(m)

    # Display the matched points
    showmatches = cv.drawMatches(image1, kp1, image2, kp2, good_matches, None, flags=2)
    # plt.imshow(showmatches)

    # Create np arrays of matched points in source and destination images
    source_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    destination_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Compute homography
    Homo, mask = cv.findHomography(destination_pts, source_pts, cv.RANSAC, 5.0)
    # Homo=ComputeHomography_RANSAC(destination_pts,source_pts)

    # warp the two images
    warped_img = cv.warpPerspective(image2, Homo, (image1.shape[1]+image2.shape[1], image2.shape[0]))
    warped_img[0:image1.shape[0],0:image1.shape[1]] = image1

    grayimage = cv.cvtColor(warped_img,cv.COLOR_BGR2GRAY)
    _,thresh = cv.threshold(grayimage,0,255,cv.THRESH_BINARY)

    #  Trim black part
    contours,hierarchy = cv.findContours(thresh,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x,y,w,h = cv.boundingRect(cnt)

    # Now crop the image, and save it into another image.
    crop = warped_img[y:y+h,x:x+w]

    # plt.show()
    return crop


image_paths=['image_1.jpg','image_2.jpg','image_3.jpg','image_4.jpg']

imgs = []
  
for i in range(len(image_paths)):
    imgs.append(cv.imread(image_paths[i]))


first_image=imgs[0]

for i in range(len(imgs)-1):
    first_image=StitchTwoImages(first_image,imgs[i+1])

plt.imshow(first_image)

plt.show()