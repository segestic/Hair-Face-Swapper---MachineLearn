import cv2
import dlib
import numpy as np
import os
#new
import io
from PIL import Image
#nwend
class MoreThanOneFaceException(Exception): pass
class NoFaceException(Exception): pass

class FaceSwap:
    # ================================== CONSTANTS =====================================

    # PREDICTOR_PATH = f"{os.getcwd()}/shape_predictor_68_face_landmarks.dat"
    PREDICTOR_PATH = "./shape_predictor_68_face_landmarks.dat"
    #Execution Constants
    SCALE_FACTOR = 1
    FEATHER_AMOUNT = 11
    COLOUR_CORRECT_BLUR_FRAC = 0.6

    #Face Structure Declarations
    MOUTH_POINTS = list(range(48, 61))
    RIGHT_BROW_POINTS = list(range(17, 22))
    LEFT_BROW_POINTS = list(range(22, 27))
    RIGHT_EYE_POINTS = list(range(36, 42))
    LEFT_EYE_POINTS = list(range(42, 48))
    NOSE_POINTS = list(range(27, 35))

    # Points used to line up the images.
    ALIGN_POINTS = (
        LEFT_BROW_POINTS + 
        RIGHT_EYE_POINTS + 
        LEFT_EYE_POINTS +
        RIGHT_BROW_POINTS + 
        NOSE_POINTS + 
        MOUTH_POINTS
    )

    # Points from the second image to overlay on the first. The convex hull of each
    # element will be overlaid.
    OVERLAY_POINTS = [
        LEFT_EYE_POINTS + 
        RIGHT_EYE_POINTS + 
        LEFT_BROW_POINTS + 
        RIGHT_BROW_POINTS,
        NOSE_POINTS + 
        MOUTH_POINTS,
    ]

    #Runtime Variables
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
    
    # =================================================================================


    def get_landmarks(self, im):
        rects = self.detector(im, 1)

        if len(rects) > 1:
            print("There seems to be more than one face in the picture!")
            raise MoreThanOneFaceException
        if len(rects) == 0:
            print("Could not detect any faces!")
            raise NoFaceException

        return np.matrix([[p.x, p.y] for p in self.predictor(im, rects[0]).parts()])

    def draw_convex_hull(self, im, points, color):
        points = cv2.convexHull(points)
        cv2.fillConvexPoly(im, points, color=color)

    def get_face_mask(self, im, landmarks):
        im = np.zeros(im.shape[:2], dtype=np.float64)

        for group in self.OVERLAY_POINTS:
            self.draw_convex_hull(im,
                             landmarks[group],
                             color=1)

        im = np.array([im, im, im]).transpose((1, 2, 0))

        im = (cv2.GaussianBlur(im, (self.FEATHER_AMOUNT, self.FEATHER_AMOUNT), 0) > 0) * 1.0
        im = cv2.GaussianBlur(im, (self.FEATHER_AMOUNT, self.FEATHER_AMOUNT), 0)
        return im

    def transformation_from_points(self, points1, points2):
        """
        Return an affine transformation [s * R | T] such that:
            sum ||s*R*p1,i + T - p2,i||^2  is minimized.

        Solve the procrustes problem by subtracting centroids, scaling by the
        standard deviation, and then using the SVD to calculate the rotation.
        More Details: https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
        """

        points1 = points1.astype(np.float64)
        points2 = points2.astype(np.float64)

        c1 = np.mean(points1, axis=0)
        c2 = np.mean(points2, axis=0)
        points1 -= c1
        points2 -= c2

        s1 = np.std(points1)
        s2 = np.std(points2)
        points1 /= s1
        points2 /= s2

        U, S, Vt = np.linalg.svd(points1.T * points2)

        R = (U * Vt).T

        return np.vstack([
                np.hstack(
                    (
                        (s2 / s1) * R,
                        c2.T - (s2 / s1) * R * c1.T,
                    )
                ),
                np.matrix([0., 0., 1.])
            ])

    def read_im_and_landmarks(self, fname):
        # im = cv2.imread(fname, cv2.IMREAD_COLOR) #edit
        im = cv2.imdecode(fname, cv2.IMREAD_COLOR) #edit
        im = cv2.resize(im, (im.shape[1] * self.SCALE_FACTOR,
                             im.shape[0] * self.SCALE_FACTOR))
        s = self.get_landmarks(im)

        return im, s

    def warp_im(self, im, M, dshape):
        output_im = np.zeros(dshape, dtype=im.dtype)
        cv2.warpAffine(
            im,
            M[:2],
            (dshape[1], dshape[0]),
            dst=output_im,
            borderMode=cv2.BORDER_TRANSPARENT,
            flags=cv2.WARP_INVERSE_MAP
        )
        return output_im

    def correct_colours(self, im1, im2, landmarks1):
        blur_amount = self.COLOUR_CORRECT_BLUR_FRAC * np.linalg.norm(
            np.mean(landmarks1[self.LEFT_EYE_POINTS], axis=0) -
            np.mean(landmarks1[self.RIGHT_EYE_POINTS], axis=0))
        blur_amount = int(blur_amount)
        if blur_amount % 2 == 0:
            blur_amount += 1
        im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
        im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

        # Avoid divide-by-zero errors.
        im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)

        return (im2.astype(np.float64) * im1_blur.astype(np.float64) /
                im2_blur.astype(np.float64))
    
    def generate_name(self, p1, p2):
        return os.path.splitext(os.path.basename(p1))[0], os.path.splitext(os.path.basename(p2))[0]
    
    def __init__(self, image1, image2):
        print("Started Swap")
        
        # image = cv2.imread(imagefile)
        # img1 = Image.open(image_bytes1)
        # img2 = Image.open(image_bytes2)
        #im = cv2.imdecode(fname, cv2.IMREAD_COLOR) #edit
        #image = np.asarray(bytearray(image1.read()), dtype="uint8") 
        #image1 = cv2.imdecode(image1, cv2.IMREAD_COLOR) 
        #np.asarray(bytearray(imagefile.read()),
        # image1 = io.BytesIO(bytes(image1.read()))
        ###########
        
        image1 = bytes(image1.read())#open(mode='rb'))
        # nparr = np.fromstring(image1, np.uint8)
        # img_np = cv2.imdecode(nparr, cv2.CV_LOAD_IMAGE_COLOR) # cv2.IMREAD_COLOR in OpenCV 3.1
        image2 = bytes(image2.read())#open(mode='rb'))
        # nparr2 = np.fromstring(image2, np.uint8)
        # img_np = cv2.imdecode(nparr2, cv2.CV_LOAD_IMAGE_COLOR) # cv2.IMREAD_COLOR in OpenCV 3.1

        # image2 = image2.open(mode='rb')
        #= str(file.read())
        #package.logo_image.image.open(mode='rb')
        ############
        # img1 = Image.open(image_bytes1)
        ########################
        # imgByteArr = io.BytesIO()
        # image.save(imgByteArr, format=image.format)
        # imgByteArr = imgByteArr.getvalue()
        ####
        # image_bytes2 = io.BytesIO(bytes(image2.read()))
        # img2 = Image.open(image_bytes2)
        # nparr = np.fromstring(img_str, np.uint8)
        # img_np = cv2.imdecode(nparr, cv2.CV_LOAD_IMAGE_COLOR) # cv2.IMREAD_COLOR in OpenCV 3.1
        # buffer = io.BytesIO()
        # image_bytes2.seek(0)
        # array2 = np.asarray(bytearray(image_bytes2.read()), dtype=np.uint8)
        # image2 = cv2.imdecode(array2, cv2.IMREAD_COLOR)
        # image_bytes1.seek(0)
        # array1 = np.asarray(bytearray(image_bytes1.read()), dtype=np.uint8)
        # image2 = cv2.imdecode(array1, cv2.IMREAD_COLOR)
        # image1 = np.asarray(bytearray(image_bytes1.getvalue()))#, dtype="uint8") 
        # image2 = np.asarray(bytearray(image_bytes2.getvalue()))#, dtype="uint8") 
        # image1 = cv2.imdecode(np.frombuffer(image_bytes1.getvalue() , np.uint8), cv2.IMREAD_UNCHANGED)
        # image2 = cv2.imdecode(np.frombuffer(image_bytes1.getvalue() , np.uint8), cv2.IMREAD_UNCHANGED)
        #image2 = cv2.imdecode(image2, cv2.IMREAD_COLOR) 
        im1, landmarks1 = self.read_im_and_landmarks(str(image1))
        im2, landmarks2 = self.read_im_and_landmarks(str(image2))
        # im1, landmarks1 = self.read_im_and_landmarks(str(image1))
        # im2, landmarks2 = self.read_im_and_landmarks(str(image2))
        M = self.transformation_from_points(
            landmarks1[self.ALIGN_POINTS],
            landmarks2[self.ALIGN_POINTS]
        )
        mask = self.get_face_mask(im2, landmarks2)
        warped_mask = self.warp_im(mask, M, im1.shape)

        mask = self.get_face_mask(im2, landmarks2)
        warped_mask = self.warp_im(mask, M, im1.shape)
        combined_mask = np.max(
            [
                self.get_face_mask(im1, landmarks1), 
                warped_mask,
            ],
            axis=0,
        )

        warped_im2 = self.warp_im(im2, M, im1.shape)
        warped_corrected_im2 = self.correct_colours(im1, warped_im2, landmarks1)

        output_im = im1 * (1.0 - combined_mask) + \
                    warped_corrected_im2 * combined_mask

        i1, i2 = self.generate_name(image1, image2)
        path = os.path.join(os.getcwd(), 'jjj', 'output', f'{i1}-{i2}.jpg')
        cv2.imwrite(path, output_im)  # saves the image to the path
        print(f"Output Added to Path: {path}")
    
   

#Running Swap Directly
if(__name__ == '__main__'):
    img1 = './hairstyle1.jpg'
    img2 = './hairstyle2.jpg'
    FaceSwap(image1=img1, image2=img2)