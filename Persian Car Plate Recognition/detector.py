from ultralytics import YOLO

class Detector:
    def __init__(self, path, imgsz,device ,conf = None,tracker='bytetrack.yaml', verbose=False, save=False,classes = None):
        self.path = path
        self.model = YOLO(self.path)
        self.imgsz = imgsz
        self.conf = conf
        self.tracker = tracker
        self.verbose = verbose
        self.save = save
        self.classes = classes
        self.device = device
    def predict(self, source):
        result = self.model.predict(
            source=source,
            imgsz=self.imgsz,
            conf=self.conf,
            verbose=self.verbose,
            save=self.save,
            device = self.device
        )
        return result

    def track(self, source):
        result = self.model.track(
            source=source,
            imgsz=self.imgsz,
            conf=self.conf,
            tracker=self.tracker,
            verbose=self.verbose,
            save=self.save,
            classes = self.classes,
            persist = True,
            device = self.device
        )
        return result
