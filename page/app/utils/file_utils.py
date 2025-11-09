
from PIL import Image
import numpy as np
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

class FileUtils:
    @staticmethod
    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    
    @staticmethod
    def preprocess(image_file, target_size = (100,100), gray_mode = False):
        try:
            img = Image.open(image_file)
            
            if(img.mode != 'RGB' and not gray_mode):
                img = img.convert('RGB')
            
            img = img.resize(target_size)
            
            img_array = np.array(img)
            
            img_array = img_array / 255.0
            
            img_array = np.expand_dims(img_array, axis=0)
            
            return img_array
        except Exception as e:
            print(f"Error preprocessing img: {e}")
            return
            