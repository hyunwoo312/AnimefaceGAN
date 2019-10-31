'''
data preprocessing for GAN training. . .
'''
import time
import cv2
import os

'''
Timer
'''
class Timer:
    '''
    Initialize a Timer() constructor before fitting GAN
    Then call .timer() on the constructor for the time elapsed
    '''
    def __init__(self):
        self.start_time = time.time()
    def elapsed_time(self,s):
        if s < 60: 
            return '{:7.2f} sec'.format(s)
        elif s < (60 * 60):  
            return '{:7.2f} min'.format(s/60)
        else:
            return '{:7.2f} hr'.format(s/60/60)
    def timer(self):
        print("Time elapsed: {}".format(self.elapsed_time(time.time() - self.start_time)))

'''
OpenCV2 w/ lbpcascade
'''
class AnimeFace:
    def __init__(self):
        self.cascadefile = "datafiles/lbpcascade_animeface.xml" # kaggle/input/datafiles/lbpcascade_animeface.xml
    
    def make_image_for_plt(self, image):
        '''
        For some reasons, calling cv2.imshow(str, cv2_image_object) crashes the Kaggle kernel,
        so I used plt.imshow(cv2_image_object) instead.
        However, OpenCV2 module reads and displays image as BGR format, instead of
        the usual RGB format. So this function converts the color of the image so it can
        be printed with plt without any issues in coloring.
        '''
        newimage = cv2.imread(image, cv2.IMREAD_COLOR)
        RGB_image = cv2.cvtColor(newimage, cv2.COLOR_BGR2RGB)
        RGB_pixels = np.array(RGB_image)
        return RGB_pixels
    
    def get_rectangles(self, img):
        '''
        Python example code extracted from
        https://github.com/nagadomi/lbpcascade_animeface/blob/master/README.md#python-example
        '''
        try:
            cascade = cv2.CascadeClassifier(self.cascadefile) # lbpcascade_animeface
            image = cv2.imread(img, cv2.IMREAD_COLOR) # read img file
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # convert to gray for faster computation
            gray = cv2.equalizeHist(gray)
            faces = cascade.detectMultiScale(gray,
                                             # detector options
                                             scaleFactor = 1.1,
                                             minNeighbors = 5,
                                             minSize = (24, 24))
            return faces
        except:
            return None
  
    def show_result(self, image) -> None:
        '''
        This function is used to confirm that the OpenCV2 has
        successfully cropped the images of face from the
        given file. 
        '''
        faces = self.get_rectangles(image)
        image = self.make_image_for_plt(image)
        for image_vertexes in faces:
            # 0 -> x, 1 -> y, 2 -> w, 3 -> h
            # image_vertexes = list(map(int, image_vertexes))
            cv2.rectangle(
                image,
                (image_vertexes[0], image_vertexes[1]),
                (image_vertexes[0] + image_vertexes[2], image_vertexes[1] + image_vertexes[3]),
                (255, 0, 0),
                2
            )
        plt.imshow(image)
        #cv2.imwrite("/kaggle/input/datafiles/{}_out.png".format(self.instancefilenumbers), image)
        
    def crop(self, image) -> list:
        '''
        This function crops only the faces of the anime characters in the
        given image file for easier degree of computation for my GAN.
        '''
        out = []
        faces = self.get_rectangles(image)
        if faces is None:
            return None
        image = self.make_image_for_plt(image)
        for [x, y, w, h] in faces:
            out.append(image[y:y+h, x:x+w])
        return out

'''
Data Loader
'''
class DataLoader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.dataset = self.get_image_files()
    '''
    returns a list of filepaths for the input files
    '''
    def load_data(self, start):
        out = []
        err = 0
        AF = AnimeFace()
        batch = self.dataset[start:]
        for filepath in batch:
            faces = AF.crop(filepath)
            if faces == None:
                err += 1
            else:
                for ea in faces:
                    out.append(ea)
            if len(out) == 64:
                return out, err
                    
    def get_image_files(self):
        meow = []
        for dirname, _, filenames in os.walk(self.filepath):
            for filename in filenames:
                meow.append(os.path.join(dirname, filename))
        return sorted(meow)