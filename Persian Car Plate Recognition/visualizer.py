import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import arabic_reshaper
from bidi.algorithm import get_display

class Visualizer:
    def __init__(self, width, height , font):
        self.width = width
        self.height = height
        self.font = font

    def plot(self, frame, results):
        resized_frame = cv2.resize(frame.image, (self.width, self.height))
        h, w = frame.image.shape[:2]
        scale_x = self.width / w
        scale_y = self.height / h
        if results is None:
            cv2.imshow("result", resized_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows() 
                return False
            return True
        for result in results:
            car = result["car"]
            plate = result["plate"]
            final = result["final"]
            if car is not None:
                x1, y1, x2, y2 = car.coords
                x1 = int(x1 * scale_x)
                x2 = int(x2 * scale_x)
                y1 = int(y1 * scale_y)
                y2 = int(y2 * scale_y)
                cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (255, 255, 255), 3)
                cv2.putText(resized_frame, f"Car {car.id} conf: {car.conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            if plate is not None:
                x1, y1, x2, y2 = plate.coords
                x1 = int((x1+car.coords[0]) * scale_x) 
                x2 = int((x2+car.coords[0]) * scale_x)
                y1 = int((y1+car.coords[1]) * scale_y)
                y2 = int((y2+car.coords[1]) * scale_y)
                cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0,0,255), 2)
                if final is not None:
                    reshaped_text = arabic_reshaper.reshape(str(final))
                    persian_text = get_display(reshaped_text)
                    img_pil = Image.fromarray(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(img_pil)
                    font = ImageFont.truetype(self.font, 35)
                    draw.text((x1, y1 - 30), persian_text, font=font, fill=(0,0,255))
                    resized_frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        cv2.imshow("result", resized_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            return False
        return True
