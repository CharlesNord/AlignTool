import cv2
import dlib
import numpy as np
from matlab_cpt2tform import get_similarity_transform_for_cv2

TEMPLATE = np.float32([
    (0.0792396913815, 0.339223741112), (0.0829219487236, 0.456955367943),
    (0.0967927109165, 0.575648016728), (0.122141515615, 0.691921601066),
    (0.168687863544, 0.800341263616), (0.239789390707, 0.895732504778),
    (0.325662452515, 0.977068762493), (0.422318282013, 1.04329000149),
    (0.531777802068, 1.06080371126), (0.641296298053, 1.03981924107),
    (0.738105872266, 0.972268833998), (0.824444363295, 0.889624082279),
    (0.894792677532, 0.792494155836), (0.939395486253, 0.681546643421),
    (0.96111933829, 0.562238253072), (0.970579841181, 0.441758925744),
    (0.971193274221, 0.322118743967), (0.163846223133, 0.249151738053),
    (0.21780354657, 0.204255863861), (0.291299351124, 0.192367318323),
    (0.367460241458, 0.203582210627), (0.4392945113, 0.233135599851),
    (0.586445962425, 0.228141644834), (0.660152671635, 0.195923841854),
    (0.737466449096, 0.182360984545), (0.813236546239, 0.192828009114),
    (0.8707571886, 0.235293377042), (0.51534533827, 0.31863546193),
    (0.516221448289, 0.396200446263), (0.517118861835, 0.473797687758),
    (0.51816430343, 0.553157797772), (0.433701156035, 0.604054457668),
    (0.475501237769, 0.62076344024), (0.520712933176, 0.634268222208),
    (0.565874114041, 0.618796581487), (0.607054002672, 0.60157671656),
    (0.252418718401, 0.331052263829), (0.298663015648, 0.302646354002),
    (0.355749724218, 0.303020650651), (0.403718978315, 0.33867711083),
    (0.352507175597, 0.349987615384), (0.296791759886, 0.350478978225),
    (0.631326076346, 0.334136672344), (0.679073381078, 0.29645404267),
    (0.73597236153, 0.294721285802), (0.782865376271, 0.321305281656),
    (0.740312274764, 0.341849376713), (0.68499850091, 0.343734332172),
    (0.353167761422, 0.746189164237), (0.414587777921, 0.719053835073),
    (0.477677654595, 0.706835892494), (0.522732900812, 0.717092275768),
    (0.569832064287, 0.705414478982), (0.635195811927, 0.71565572516),
    (0.69951672331, 0.739419187253), (0.639447159575, 0.805236879972),
    (0.576410514055, 0.835436670169), (0.525398405766, 0.841706377792),
    (0.47641545769, 0.837505914975), (0.41379548902, 0.810045601727),
    (0.380084785646, 0.749979603086), (0.477955996282, 0.74513234612),
    (0.523389793327, 0.748924302636), (0.571057789237, 0.74332894691),
    (0.672409137852, 0.744177032192), (0.572539621444, 0.776609286626),
    (0.5240106503, 0.783370783245), (0.477561227414, 0.778476346951)])

TPL_MIN, TPL_MAX = np.min(TEMPLATE, axis=0), np.max(TEMPLATE, axis=0)
MINMAX_TEMPLATE = (TEMPLATE - TPL_MIN) / (TPL_MAX - TPL_MIN)


class AlignDlib:
    """
    Use `dlib's landmark estimation <http://blog.dlib.net/2014/08/real-time-face-pose-estimation.html>`_ to align faces.

    The alignment preprocess faces for input into a neural network.
    Faces are resized to the same size (such as 96x96) and transformed
    to make landmarks (such as the eyes and nose) appear at the same
    location on every image.

    Normalized landmarks:

    .. image:: ../images/dlib-landmark-mean.png
    """

    #: Landmark indices.
    INNER_EYES_AND_BOTTOM_LIP = [39, 42, 57]
    OUTER_EYES_AND_NOSE = [36, 45, 33]

    def __init__(self, facePredictor='./shape_predictor_68_face_landmarks.dat'):
        """
        Instantiate an 'AlignDlib' object.

        :param facePredictor: The path to dlib's
        :type facePredictor: str
        """
        assert facePredictor is not None

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(facePredictor)
        self.detector1 = dlib.cnn_face_detection_model_v1('./mmod_human_face_detector.dat')

    def getAllFaceBoundingBoxes(self, rgbImg):
        """
        Find all face bounding boxes in an image.

        :param rgbImg: RGB image to process. Shape: (height, width, 3)
        :type rgbImg: numpy.ndarray
        :return: All face bounding boxes in an image.
        :rtype: dlib.rectangles
        """
        assert rgbImg is not None

        try:
            return self.detector(rgbImg, 1)
        except Exception as e:
            print("Warning: {}".format(e))
            # In rare cases, exceptions are thrown.
            return []

    def getLargestFaceBoundingBox(self, rgbImg, skipMulti=False):
        """
        Find the largest face bounding box in an image.

        :param rgbImg: RGB image to process. Shape: (height, width, 3)
        :type rgbImg: numpy.ndarray
        :param skipMulti: Skip image if more than one face detected.
        :type skipMulti: bool
        :return: The largest face bounding box in an image, or None.
        :rtype: dlib.rectangle
        """
        assert rgbImg is not None

        faces = self.getAllFaceBoundingBoxes(rgbImg)
        if (not skipMulti and len(faces) > 0) or len(faces) == 1:
            return max(faces, key=lambda rect: rect.width() * rect.height())
        else:
            return None

    def findLandmarks(self, rgbImg, bb):
        """
        Find the landmarks of a face.

        :param rgbImg: RGB image to process. Shape: (height, width, 3)
        :type rgbImg: numpy.ndarray
        :param bb: Bounding box around the face to find landmarks for.
        :type bb: dlib.rectangle
        :return: Detected landmark locations.
        :rtype: list of (x,y) tuples
        """
        assert rgbImg is not None
        assert bb is not None

        points = self.predictor(rgbImg, bb)
        return list(map(lambda p: (p.x, p.y), points.parts()))

    def align(self, imgDim, rgbImg, bb=None,
              landmarks=None, landmarkIndices=INNER_EYES_AND_BOTTOM_LIP,
              skipMulti=False):
        r"""align(imgDim, rgbImg, bb=None, landmarks=None, landmarkIndices=INNER_EYES_AND_BOTTOM_LIP)

        Transform and align a face in an image.

        :param imgDim: The edge length in pixels of the square the image is resized to.
        :type imgDim: int
        :param rgbImg: RGB image to process. Shape: (height, width, 3)
        :type rgbImg: numpy.ndarray
        :param bb: Bounding box around the face to align. \
                   Defaults to the largest face.
        :type bb: dlib.rectangle
        :param landmarks: Detected landmark locations. \
                          Landmarks found on `bb` if not provided.
        :type landmarks: list of (x,y) tuples
        :param landmarkIndices: The indices to transform to.
        :type landmarkIndices: list of ints
        :param skipMulti: Skip image if more than one face detected.
        :type skipMulti: bool
        :return: The aligned RGB image. Shape: (imgDim, imgDim, 3)
        :rtype: numpy.ndarray
        """
        assert imgDim is not None
        assert rgbImg is not None
        assert landmarkIndices is not None

        if bb is None:
            bb = self.getLargestFaceBoundingBox(rgbImg, skipMulti)
            if bb is None:
                return

        if landmarks is None:
            landmarks = self.findLandmarks(rgbImg, bb)

        npLandmarks = np.float32(landmarks)
        npLandmarkIndices = np.array(landmarkIndices)

        H = cv2.getAffineTransform(npLandmarks[npLandmarkIndices],
                                   imgDim * MINMAX_TEMPLATE[npLandmarkIndices])
        thumbnail = cv2.warpAffine(rgbImg, H, (imgDim, imgDim))

        return thumbnail

    def align2(self, rgbImg):
        # No landmarks, sphere face alignment, similarity transform
        bb = self.getLargestFaceBoundingBox(rgbImg, skipMulti=False)
        landmarks = self.findLandmarks(rgbImg, bb)
        src_pts = np.array(landmarks)
        left_eye = src_pts[range(36, 42)].mean(0)
        right_eye = src_pts[range(42, 48)].mean(0)
        nose = src_pts[30]
        left_mouth = src_pts[48]
        right_mouth = src_pts[54]
        src_pts = np.vstack([left_eye, right_eye, nose, left_mouth, right_mouth])

        ref_pts = [[30.2946, 51.6963], [65.5318, 51.5014],
                   [48.0252, 71.7366], [33.5493, 92.3655], [62.7299, 92.2041]]
        crop_size = (96, 112)
        src_pts = np.array(src_pts).reshape(5, 2)

        s = np.array(src_pts).astype(np.float32)
        r = np.array(ref_pts).astype(np.float32)

        tfm = get_similarity_transform_for_cv2(s, r)
        face_img = cv2.warpAffine(rgbImg, tfm, crop_size)
        return face_img

    def align3(self, img, ldmk):
        # with landmarks, crop size 144 (preprocess the training data)
        crop_size = (144, 144)
        left_eye = MINMAX_TEMPLATE[range(36, 42)].mean(0)
        right_eye = MINMAX_TEMPLATE[range(42, 48)].mean(0)
        nose = MINMAX_TEMPLATE[30]
        left_mouth = MINMAX_TEMPLATE[48]
        right_mouth = MINMAX_TEMPLATE[54]
        r = np.vstack([left_eye, right_eye, nose, left_mouth, right_mouth])
        s = np.vstack([ldmk[0], ldmk[1], ldmk[10], ldmk[13], ldmk[15]])
        tfm = get_similarity_transform_for_cv2(s, r * crop_size)
        face_img = cv2.warpAffine(img, tfm, crop_size)
        return face_img

    def align4(self, img):
        # without landmark, crop size 128, similarity transform
        crop_size = (128, 128)
        left_eye = MINMAX_TEMPLATE[range(36, 42)].mean(0)
        right_eye = MINMAX_TEMPLATE[range(42, 48)].mean(0)
        nose = MINMAX_TEMPLATE[30]
        left_mouth = MINMAX_TEMPLATE[48]
        right_mouth = MINMAX_TEMPLATE[54]
        r = np.vstack([left_eye, right_eye, nose, left_mouth, right_mouth])

        bb = self.getLargestFaceBoundingBox(img, skipMulti=False)
        landmarks = self.findLandmarks(img, bb)
        src_pts = np.array(landmarks)
        left_eye = src_pts[range(36, 42)].mean(0)
        right_eye = src_pts[range(42, 48)].mean(0)
        nose = src_pts[30]
        left_mouth = src_pts[48]
        right_mouth = src_pts[54]
        s = np.vstack([left_eye, right_eye, nose, left_mouth, right_mouth])

        tfm = get_similarity_transform_for_cv2(s, r * crop_size)
        face_img = cv2.warpAffine(img, tfm, crop_size)
        return face_img

    def align5(self, img, bb):
        # no landmark, but with bouding box given, for images hard to detect face
        crop_size = (128, 128)
        left_eye = MINMAX_TEMPLATE[range(36, 42)].mean(0)
        right_eye = MINMAX_TEMPLATE[range(42, 48)].mean(0)
        nose = MINMAX_TEMPLATE[30]
        left_mouth = MINMAX_TEMPLATE[48]
        right_mouth = MINMAX_TEMPLATE[54]
        r = np.vstack([left_eye, right_eye, nose, left_mouth, right_mouth])

        landmarks = self.findLandmarks(img, bb)
        src_pts = np.array(landmarks)
        left_eye = src_pts[range(36, 42)].mean(0)
        right_eye = src_pts[range(42, 48)].mean(0)
        nose = src_pts[30]
        left_mouth = src_pts[48]
        right_mouth = src_pts[54]
        s = np.vstack([left_eye, right_eye, nose, left_mouth, right_mouth])

        tfm = get_similarity_transform_for_cv2(s, r * crop_size)
        face_img = cv2.warpAffine(img, tfm, crop_size)
        return face_img

    def align6(self, img):
        # no landmarks, with cnn detector, crop size 128, for preprocess the test data
        crop_size = (128, 128)
        left_eye = MINMAX_TEMPLATE[range(36, 42)].mean(0)
        right_eye = MINMAX_TEMPLATE[range(42, 48)].mean(0)
        nose = MINMAX_TEMPLATE[30]
        left_mouth = MINMAX_TEMPLATE[48]
        right_mouth = MINMAX_TEMPLATE[54]
        r = np.vstack([left_eye, right_eye, nose, left_mouth, right_mouth])

        # bb = self.getLargestFaceBoundingBox(img, skipMulti=False)
        faces = self.detector1(img, 1)
        bb = max(faces, key=lambda face: face.rect.width() * face.rect.height())

        landmarks = self.findLandmarks(img, bb.rect)
        src_pts = np.array(landmarks)
        left_eye = src_pts[range(36, 42)].mean(0)
        right_eye = src_pts[range(42, 48)].mean(0)
        nose = src_pts[30]
        left_mouth = src_pts[48]
        right_mouth = src_pts[54]
        s = np.vstack([left_eye, right_eye, nose, left_mouth, right_mouth])

        tfm = get_similarity_transform_for_cv2(s, r * crop_size)
        face_img = cv2.warpAffine(img, tfm, crop_size)
        return face_img

    def align7(self, img):
        # no landmarks, with cnn detector, crop size 128, lightcnn alignment
        crop_size = (128, 128)
        ref_pts = [[42.8165, 41.7989], [86.0301, 41.8347],
                   [64.3580, 65.4530], [44.9035, 89.2332], [84.1586, 89.3568]]

        faces = self.detector1(img, 1)
        bb = max(faces, key=lambda face: face.rect.width() * face.rect.height())
        landmarks = self.findLandmarks(img, bb.rect)
        src_pts = np.array(landmarks)
        left_eye = src_pts[range(36, 42)].mean(0)
        right_eye = src_pts[range(42, 48)].mean(0)
        nose = src_pts[30]
        left_mouth = src_pts[48]
        right_mouth = src_pts[54]
        s = np.vstack([left_eye, right_eye, nose, left_mouth, right_mouth])

        r = np.array(ref_pts).astype(np.float32)

        tfm = get_similarity_transform_for_cv2(s, r)
        face_img = cv2.warpAffine(img, tfm, crop_size)
        return face_img

    def align8(self, img, bb):
        # no landmark, but with bouding box given, for images hard to detect face
        crop_size = (128, 128)
        ref_pts = [[42.8165, 41.7989], [86.0301, 41.8347],
                   [64.3580, 65.4530], [44.9035, 89.2332], [84.1586, 89.3568]]

        r = np.array(ref_pts).astype(np.float32)

        landmarks = self.findLandmarks(img, bb)
        src_pts = np.array(landmarks)
        left_eye = src_pts[range(36, 42)].mean(0)
        right_eye = src_pts[range(42, 48)].mean(0)
        nose = src_pts[30]
        left_mouth = src_pts[48]
        right_mouth = src_pts[54]
        s = np.vstack([left_eye, right_eye, nose, left_mouth, right_mouth])

        tfm = get_similarity_transform_for_cv2(s, r)
        face_img = cv2.warpAffine(img, tfm, crop_size)
        return face_img


    def align9(self, img, ldmk):
        # with landmarks, crop size 144 (preprocess the training data)
        crop_size = (144, 144)
        ref_pts = [[42.8165, 41.7989], [86.0301, 41.8347],
                   [64.3580, 65.4530], [44.9035, 89.2332], [84.1586, 89.3568]]

        r = np.array(ref_pts).astype(np.float32)

        s = np.vstack([ldmk[0], ldmk[1], ldmk[10], ldmk[13], ldmk[15]])
        tfm = get_similarity_transform_for_cv2(s, r)
        face_img = cv2.warpAffine(img, tfm, crop_size)
        return face_img

