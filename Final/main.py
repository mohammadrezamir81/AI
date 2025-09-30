from detector import Detector
from utils import *
from vote import *
from parallelCarProcessor import *
from visualizer import *

if __name__ == "__main__":
    videostreamer = VideoStream('path/',frame_skip = 10)#Insert the video path or Insert 0 to use your webcam
    car_detector = Detector(path='detectors/car_detector_n.pt',conf=0.7, imgsz=352, save=False,classes=[2,5,7,3],device = 'cpu')
    plate_detector = Detector(path='detectors/plate_detector_n.pt',conf=0.2, imgsz=160, save=False,device = 'cpu')
    char_detector = Detector(path='detectors/char_detector_n.pt',conf=0.6, imgsz=352, save=False,device = 'cpu')
    vote_system = Vote(max_frames=3)#number of frames used for voting
    car_pipeline = CarPipeline(plate_detector, char_detector, vote_system, max_workers=4)
    plotter = Visualizer(width = 1080 , height = 1080 , font = 'Font/B Nazanin Bold.ttf')
    while True:
        frame = next(videostreamer)

        cars = frame.DetectCars(car_detector)

        results = car_pipeline.process_all(cars)

        ret = plotter.plot(frame,results)

        if ret == False:
            break
