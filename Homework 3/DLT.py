import numpy as np 
import cv2 
import glob
from scipy.spatial.transform import Rotation 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.patches import Rectangle
import celluloid 
from celluloid import Camera # couldn't save animation with ArtisticAnimation, TO DO

# datadir = 'data/'
# detected_corners = np.loadtxt(datadir + 'detected_corners.txt') # pixel coords (u,v) of detected corners 
# K = np.loadtxt(datadir + 'K.txt') # camera matrix 
# Pw_corners = .01 * np.loadtxt('data/p_W_corners.txt', delimiter=',') # [12x3] world coords of detected corners in centimeters

class DLT(object): 
    def __init__(self, K, detected_corners, Pw_corners, reproject_points=False): 
        self.K = K 
        self.p = detected_corners
        self.Pw = Pw_corners
        self.reproject_points = reproject_points


    def getimg(self, idx): 
        images = sorted(glob.glob(datadir + 'images_undistorted/*.jpg'))
        return cv2.imread(images[idx])


    # def currFrame(detected_corners, K, Pw_corners, frame_idx): 
    def currFrame(self, frame_idx):
        # get normalized coordinates [x;y;1]
        u = self.p[frame_idx][0:-1:2]
        v = self.p[frame_idx][1::2]
        p = np.linalg.inv(self.K) @ np.vstack((u,v,np.ones(u.shape[0])))

        # get 3d world coordinates [X; Y; Z; 1]
        P = np.vstack((self.Pw.T, np.ones(self.Pw.shape[0])))

        return p, P


    def estimatePoseDLT(self, p, P,idx): 
        '''
        DLT algorithm. Refer to http://www.kwon3d.com/theory/dlt/dlt.html for in-depth analysis 
        Solves for projection matrix M = [R|t], given the n 2D-3D points corresponding to p_i and P_i
        ***Note: Matrix Q is built using the /normalized/ coordinates of p_i 
                SVD returns V already transposed
        Args: 
            p = given 2D coordinates (u,v) of the projections of the referenced 3D points in the undistorted image
            P = given position coordinates of the n reference 3D points given in the world coordinates 
            K = given camera matrix 
        Returns: 
            M = The solved projection matrix (for normalized coordinates)
        '''
        # construct Q matrix  
        for col_idx in range(0, P.shape[1]):
            if col_idx == 0:
                Q = np.array([
                    [P[0,col_idx], P[1,col_idx], P[2,col_idx], 1, 0, 0, 0, 0, -p[0, col_idx]*P[0,col_idx], -p[0, col_idx]*P[1,col_idx], -p[0, col_idx]*P[2,col_idx], -p[0, col_idx]], 
                    [0, 0, 0, 0, P[0,col_idx], P[1,col_idx], P[2,col_idx], 1, -p[1, col_idx]*P[0,col_idx], -p[1, col_idx]*P[1,col_idx], -p[1, col_idx]*P[2,col_idx], -p[1, col_idx]]
                ])
            else: 
                currQ = np.array([
                    [P[0,col_idx], P[1,col_idx], P[2,col_idx], 1, 0, 0, 0, 0, -p[0, col_idx]*P[0,col_idx], -p[0, col_idx]*P[1,col_idx], -p[0, col_idx]*P[2,col_idx], -p[0, col_idx]], 
                    [0, 0, 0, 0, P[0,col_idx], P[1,col_idx], P[2,col_idx], 1, -p[1, col_idx]*P[0,col_idx], -p[1, col_idx]*P[1,col_idx], -p[1, col_idx]*P[2,col_idx], -p[1, col_idx]]
                ]) 
                Q = np.vstack((Q,currQ)).astype(np.float32)
        
        U, S, V = np.linalg.svd(Q, full_matrices=True)
        M = V[-1:]
        M = M.reshape((3,4)) # reshape to true projection matrix  
        if np.linalg.det(M[:,:3]) < 0: 
            M = -M 
        '''
        Orthogonal Procrustes problem: 
            Did not impose any constraints on R from M = [R|t] is actually a rotation matrix; 
            Need to compute matrix R_tilde, the matrix closest to the true "R" in the sense of Frobenius norm  
        '''
        R = M[:,:3] # rotation matrix 
        U,S,V = np.linalg.svd(R)
        R_tilde = U @ V
        
        # M is not true M in this case, but alpha*M where alpha is the scale 
        alpha = np.linalg.norm(R_tilde, ord='fro') / np.linalg.norm(R, ord='fro') 
        M = np.hstack((R_tilde, alpha*M[:,-1].reshape((3,1))))
        return M

    def reprojectPoints(self, P, M):
        ''' 
        Reprojects the 3D points P_i in the current image using the estimated projection matrix M 
        and camera matrix K. Use this to show on image to double check that reprojected points p_i' fall close to 
        points p_i. 
        Args: 
            P = referenced 3D world coordinates 
            M = Projection matrix solved from estimatePoseDLT
            org_image = the original image, needed to project points onto 
        Returns: 
            reprojected_pts = self-explanatory 
        ''' 
        homo_mtx = (K @ M @ P).T
        homo_mtx[:,0] = homo_mtx[:,0] / homo_mtx[:,2]
        homo_mtx[:,1] = homo_mtx[:,1] / homo_mtx[:,2]

        reprojected_pts = homo_mtx[:,:2]
        # print(reprojected_pts)
        return reprojected_pts

def plotTrajectory3D(M, fig, output_filename='motion.avi'): 
    R = M[:,:3].T
    t = M[:,-1]
    
    rotMat = Rotation.from_matrix(R) # Rotation object instance 
    quat = rotMat.as_quat()
    quat = np.roll(quat, 1)
    transl = -R @ t
    
    # prelims 
    plt.clf()
    # ax = fig.add_subplot(111, projection='3d')
    ax = plt.axes(projection='3d')
    camera = Camera(fig)
    ax.set(xlim=(-.1,.4), ylim=(-.2,.3), zlim=(-.3, 0))
    ax.set_xlabel('Z')
    ax.set_ylabel('X')
    ax.set_zlabel('Y')
    ax.scatter(-Pw_corners[:,2], Pw_corners[:,0], -Pw_corners[:,1]) # draw given corners 
    # draw rectangles at corners
    r = Rectangle((0, -.22), width=.105, height=.14, color='blue', fill=False, hatch='/')
    ax.add_patch(r)
    art3d.pathpatch_2d_to_3d(r, z=0, zdir='x') 
    r1 = Rectangle((.11,-.25), width=.13, height=.1, color='red', fill=False, hatch='/')
    ax.add_patch(r1)
    art3d.pathpatch_2d_to_3d(r1, z=.2, zdir='y') 
    r2 = Rectangle((.11, 0), width=.13, height=.11, color='green', fill=False, hatch='/')
    ax.add_patch(r2)
    art3d.pathpatch_2d_to_3d(r2, z=-.265, zdir='z')   

    # draw camera coordinate frame onto image 
    rotMat = rotMat.as_matrix()
    ax.quiver(-transl[2], transl[0], -transl[1], -rotMat[2,0], rotMat[0,0], -rotMat[1,0], color='red', length=.1)
    ax.quiver(-transl[2], transl[0], -transl[1], -rotMat[2,1], rotMat[0,1], -rotMat[1,1], color='green', length=.1)
    ax.quiver(-transl[2], transl[0], -transl[1], -rotMat[2,2], rotMat[1,2], -rotMat[1,2], color='blue', length=.1)
    # print([-transl[2], transl[0], -transl[1], -rotMat[2,0], rotMat[0,0], -rotMat[1,0]])
    camera.snap()
    


if __name__ == "__main__": 
    # Given info 
    datadir = 'data/'
    detected_corners = np.loadtxt(datadir + 'detected_corners.txt') # pixel coords (u,v) of detected corners 
    K = np.loadtxt(datadir + 'K.txt') # camera matrix 
    Pw_corners = .01 * np.loadtxt('data/p_W_corners.txt', delimiter=',') # [12x3] world coords of detected corners in centimeters

    # Iterate through each picture 
    file_list = sorted(glob.glob(datadir + 'images_undistorted/*.jpg'))
    # num_images = len(glob.glob(datadir + 'images_undistorted/*.jpg'))
    num_images = len(file_list)
    projection = DLT(K, detected_corners, Pw_corners, reproject_points=False) 
    fig = plt.figure()
    for img_idx in range(0, num_images): 
        image = projection.getimg(img_idx) # get current image in directory 

        p, P = projection.currFrame(img_idx) # get normalized 2D pixel points and 3D world points in correct format 
        M = projection.estimatePoseDLT(p, P, img_idx) # get projection matrix M = [R|t]
        
        reprojected_pts = projection.reprojectPoints(P, M) # reproject P_i onto image 
        if projection.reproject_points: 
            # show reprojected points on image
            for point in reprojected_pts: 
                estimate = point.astype(np.float32) # my estimated points 
                cv2.circle(image, tuple(estimate), radius=5, color=(0,0,255), thickness=2)

            for u, v in zip(detected_corners[img_idx][0::2], detected_corners[img_idx][1::2]): 
                u = u.astype(np.float32)
                v = v.astype(np.float32)
                cv2.circle(image, (u,v), radius=5, color=(0,255,0), thickness=2)

            cv2.imshow('img', image)
            cv2.waitKey(34) # 30 FPS 

        plotTrajectory3D(M, fig) 
        fname = 'my_results' + '/' + file_list[img_idx][28:28+4] + '.png'
        plt.savefig(fname)    # to create animation, save all of the figures into my_results directory, then run animate.py, which will produce a video 
