import numpy as np
from app.const.MATCH_DATA import FRUITS_DATA
class Predict:
    
    def get_qualification_of_image(confidence):
        if confidence > 0.9:
            return {
                'grade': 'Loại A - Xuất sắc',
                'criteria': 'Độ tin cậy cao, hình dạng rõ ràng'
            }
        elif confidence > 0.7:
            return {
                'grade': 'Loại B - Tốt',
                'criteria': 'Độ tin cậy khá, có thể nhận diện'
            }
        else:
            return {
                'grade': 'Loại C - Trung bình',
                'criteria': 'Độ tin cậy thấp, cần kiểm tra lại'
            }

    
    @staticmethod
    def predict(model, img_array):
        # send data through api
        predictions = model.predict(img_array)
        predictions_class_index = np.argmax(predictions[0])
        predictions_fruit = FRUITS_DATA[predictions_class_index]
        confidence = float(predictions[0][predictions_class_index])
        return {
            **predictions_fruit,
            "confidence":Predict.get_qualification_of_image(confidence)
        }
        