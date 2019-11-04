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