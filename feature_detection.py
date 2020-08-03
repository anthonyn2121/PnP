import numpy as np 
import scipy.signal as signal  
# from PIL import Image 
import cv2 
import matplotlib.pyplot as plt 
# import matplotlib.patches as patches 
import glob 
import scipy.spatial as spatial 

class calcCorner(object): 
    def __init__(self, img): 
        self.img = img 
        ## Given information
        self.corner_patch_size = 9
        self.harris_kappa = 0.08

        # my work       
        kh = np.array([
            [-1, 0, 1], 
            [-2, 0, 2], 
            [-1, 0, 1]
        ])

        kv = np.array([
            [-1, -2, -1], 
            [0, 0, 0],
            [1, 2, 1]
        ])
        self.Ix = signal.convolve2d(self.img, kh, mode='valid') 
        self.Iy = signal.convolve2d(self.img, kv, mode='valid')
        self.Ixx = self.Ix ** 2
        self.Iyy = self.Iy ** 2
        self.Ixy = np.multiply(self.Ix, self.Iy)
      
        self.box_filter = np.ones((self.corner_patch_size, self.corner_patch_size)) # or patch 
        self.patch_radius = int(np.floor(self.corner_patch_size/2))
        self.A = signal.convolve2d(self.Ixx, self.box_filter, mode='valid')
        self.B = signal.convolve2d(self.Iyy, self.box_filter, mode='valid')
        self.C = signal.convolve2d(self.Ixy, self.box_filter, mode='valid')
        
    def ShiTomasi(self): 
        # scores = trace/2 - ((trace/2).^2 - determinant).^0.5;
        det_M = self.A*self.B - self.C**2 
        trace_M = self.A + self.B 
        R = trace_M/2 - ((trace_M/2)**2 - det_M)**(0.5)
        R[R < 0] = 0 
        # print(R)
        R = np.pad(R, self.patch_radius+1, 'constant')
        return R 

    def Harris(self): 
        ''' 
        https://muthu.co/harris-corner-detector-implementation-in-python/
        ''' 
        img_copy_for_corners = np.copy(self.img)
        det_M = self.A*self.B - self.C**2 
        trace_M = self.A + self.B 
        R = det_M - self.harris_kappa*(trace_M**2)
        R[R < 0] = 0 
        R = np.pad(R,self.patch_radius+1,'constant' )
        return R 


    def get_corners(self, algo='Harris'):
        ''' 
        Runs either shi-tomasi or harris corner detector 
        Arguments: 
            algo = 'Harris' or 'ShiTomasi' 
        Returns: 
            scores returned by selected algorithm 
        ''' 
        pass 
        if algo == 'Harris': 
            R = self.Harris()
            return R
        elif algo == 'ShiTomasi':
            R = self.ShiTomasi
            return R

def get_image(idx): 
    fname = sorted(glob.glob('data/*.png'))
    return cv2.imread(fname[idx]) 


def selectKeyPoints(corners, num_best_scores, r): 
    '''
    Thresholding the scores resulting from the corner detection algorithms, and then non-max supression 
    Args: 
        corners = scores resulting from Harris or Shi-Tomasi algorithms 
        num_best_scores = number of scores you would like to keep 
        r = radius; supression of a (2r + 1)*(2r + 1) box around the current maximum.
    Returns: 
        keypoints = keypoints of current frame as numpy array  
    '''
    keypoints = np.zeros((2,num_best_scores))
    temp_scores = np.pad(corners, r, 'constant') # pad the scores so we don't lose information around the border
    for i in range(0,num_best_scores): 
        (row, col) = np.where(temp_scores == np.amax(temp_scores.flatten('F')))
        keypoints[0, i] = int(row[0])-r # subtract radius here so that the keypoint index (u,v) matches the image 
        keypoints[1, i] = int(col[0])-r
        temp_scores[int(row[0])-r:int(row[0])+r, int(col[0])-r:int(col[0])+r] = 0 # set the pixels around the keypoint equal to 0
    return keypoints 


def describeKeyPoints(img, keypoints, r):
    '''
    Describe the keypoints by the pixel intensities (or values) around the keypoint 
    Args: 
        img = current gray-scale image (one dimensional) 
        keypoints = detected keypoints of the current frame 
        r = patch radius 
    Returns: 
        descriptors = dxk matrix, where d = descriptor dimension(total amt of pixels in patch) and k = amount of keypoints 
                      the i-th column describes the intensities of the i-th keypoint 
    ''' 
    descriptors = np.zeros(((2*r)**2, keypoints.shape[1]))
    padded_img = np.pad(img, r, 'constant') # pad the image so we don't lose information on the image borders
    for i in range(keypoints.shape[1]): 
        i_kp = keypoints[:,i] + r # new i-th keypoint indices
        i_desc = padded_img[int(i_kp[0])-r:int(i_kp[0])+r, int(i_kp[1])-r:int(i_kp[1])+r].reshape((2*r)**2) # i-th descriptor 
        descriptors[:,i] = i_desc[:]
        # if i == 3: 
        #     break
    return descriptors 

def matchDescriptors(database_descriptors, query_descriptors, match_lambda): 
    '''
    Match descriptors between two frames by matching the image patches that differ the least. 
    Only matches descriptors which have a smaller descriptor distance between the two frames. 
    Arguments: 
        database_descriptors = previous frame's descriptors matrix 
        query_descriptors = current frame's descriptors matrix 
        match_lambda = constant for the adaptive threshold; delta = lambda * dmin where dmin is the smallest non-zero distance b/w descriptors 
    Returns: 
        matches = a 1xQ matrix where the i-th coefficient is the index of the database descriptor which matches to the i-th query descriptor
    '''
    pass 
    dists = np.zeros(len(database_descriptors.T))
    matches = np.zeros(len(database_descriptors.T))
    for i,p in enumerate(query_descriptors.T): 
        ed = np.sqrt(np.sum((p-database_descriptors.T)**2, axis=1))
        dists[i] = ed.min()
        matches[i] = ed.argmin()
    
    sorted_dists = np.sort(dists)
    sorted_dists = sorted_dists[sorted_dists != 0]
    dmin = sorted_dists[0]

    matches[dists >= (match_lambda*dmin)] = 0 
    
    # remove double matches 
    unique_matches = np.zeros(matches.shape)
    _, unique_match_idxes = np.unique(matches, return_index=True)
    unique_matches[unique_match_idxes] = matches[unique_match_idxes]
    matches = unique_matches
    return matches
    

if __name__ == '__main__':
    # Given variables 
    num_keypoints = 200
    nonmaximum_supression_radius = 8
    descriptor_radius = 9
    match_lambda = 4

    out = cv2.VideoWriter('output.avi', 0, 1, (1241,376))
    for frame_num in range(len(glob.glob('data/*.png'))): 
        print('frame #: ', frame_num)
        image = get_image(frame_num)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corner = calcCorner(gray)

        R = corner.get_corners('Harris')
        curr_keypoints = selectKeyPoints(R, num_keypoints, nonmaximum_supression_radius)
        curr_descriptors = describeKeyPoints(gray, curr_keypoints, descriptor_radius)
        
        if frame_num > 0: 
            matches = matchDescriptors(prev_descriptors, curr_descriptors, match_lambda)

        show_image = 1
        if show_image:
            for i in range(curr_keypoints.shape[1]): 
                cv2.circle(image,(int(curr_keypoints[1,i]), int(curr_keypoints[0,i])), 5, (0,255,0), 2)
            if frame_num > 0: 
                query_indices = (np.nonzero(matches)[0])
                match_indices = matches[matches.nonzero()].astype(np.int64)
                x_from, y_from = curr_keypoints[1, query_indices], curr_keypoints[0, query_indices]
                x_to, y_to = prev_keypoints[1, match_indices], prev_keypoints[0, match_indices]
                for j in range(len(x_to)): 
                    starting_point = (x_from[j].astype(np.float32),y_from[j].astype(np.float32)) 
                    end_point = (x_to[j].astype(np.float32), y_to[j].astype(np.float32))
                    cv2.line(image, starting_point, end_point, color=(0,0,255), thickness=2)
            cv2.imwrite(f'MY SOLUTION/results/000{frame_num}.png', image)
            # cv2.imshow('keypoints', image)
            # cv2.waitKey(15)
        
        prev_keypoints = curr_keypoints
        prev_descriptors = curr_descriptors

