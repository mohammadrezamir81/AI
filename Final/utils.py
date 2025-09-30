import cv2
class Frame:
    def __init__(self, image , source = None):
        self.image = image  
        self.source = source
        self.cars = []
    
    def rotate(self,image):
        if isinstance(self.source, int):  # webcam
            return image
        h, w = image.shape[:2]
        if w > h:  
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        return image
    

    def DetectCars(self, car_detector):
        self.image = self.rotate(self.image)
        results = car_detector.track(self.image)
        if len(results[0].boxes) == 0:
            return None
        for box in results[0].boxes:
            if box.id is None:
                return None
            track_id = int(box.id[0]) 
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            car_crop = self.image[y1:y2, x1:x2]
            self.cars.append(Car(car_crop, track_id,conf,(x1,y1,x2,y2)))
        return self.cars



class Car:
    def __init__(self, image, id , conf , coords):
        self.image = image
        self.id = id
        self.conf = conf
        self.coords = coords
    def DetectPlate(self, plate_detector):
        results = plate_detector.predict(self.image)
        if len(results[0].boxes) == 0:
            return None
        x1, y1, x2, y2 = map(int, results[0].boxes.xyxy[0])
        plate_crop = self.image[y1:y2, x1:x2]
        return Plate(plate_crop, self.id,(x1,y1,x2,y2))


class Plate:
    def __init__(self,image,id,coords):
        self.image = image
        self.id = id
        self.coords = coords
    def DetectCharactor(self, char_detector):
        results = char_detector.predict(self.image)

        if len(results[0].boxes) == 0:  
            print(f"[!] No characters detected for plate {self.id}")
            return None

   
        if len(results[0].boxes) != 8:  
            print(f"[!] Unexpected number of characters (got {len(results[0].boxes)}) for plate {self.id}")
            return None

        detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()   
            cls = int(box.cls[0].item())          
            char = char_detector.model.names[cls]        
            detections.append((x1, char))

        detections.sort(key=lambda x: x[0])
        predicted_chars = [char for _, char in detections]

        print(f"[+] Plate {self.id}: Detected → {''.join(predicted_chars)}")
        return (self.id, predicted_chars)
    
class VideoStream:
    def __init__(self, source, frame_skip=0):
        self.cap = cv2.VideoCapture(source)
        self.source = source
        if not self.cap.isOpened():
            raise ValueError(f"Unable to open video source: {source}")
        self.frame_skip = frame_skip

    def __iter__(self):
        return self

    def __next__(self):
        ret, frame = self.cap.read()
        if not ret:
            self.release()
            raise StopIteration

        for _ in range(self.frame_skip):
            self.cap.read()
        if isinstance(self.source, int):
            return Frame(frame, self.source)
        else:  
            return Frame(frame)
    def release(self):
        if self.cap.isOpened():
            self.cap.release()

