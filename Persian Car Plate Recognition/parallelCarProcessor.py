from concurrent.futures import ThreadPoolExecutor, as_completed

class CarPipeline:
    def __init__(self, plate_detector, char_detector,vote_system, max_workers=4):
        self.PlateDetector = plate_detector
        self.CharDetector = char_detector
        self.vote_system = vote_system
        self.max_workers = max_workers

    def process_car(self, car):
        try:
            plate = car.DetectPlate(self.PlateDetector)
            if plate is None:
                return {"car": car, "plate": None, "chars": None, "final": None}
            if self.vote_system.is_finalized(car.id):
                final_result = self.vote_system.final_results[car.id]
                return {"car": car, "plate": plate, "chars": None, "final": final_result}
            result = plate.DetectCharactor(self.CharDetector)
            if result is None:
                return {"car": car, "plate": plate, "chars": None, "final": None}
            plate_id, chars = result
            self.vote_system.add_prediction(plate_id, chars)
            final_result = self.vote_system.get_final_result(plate_id)
            return {"car": car, "plate": plate, "chars": chars, "final": final_result}

        except Exception as e:
            return {"car": car, "plate": None, "chars": None, "final": None, "error": str(e)}

    def process_all(self, cars):
        """Process all cars in parallel using threads."""
        results = []
        if cars is not None:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                
                futures = [executor.submit(self.process_car, car) for car in cars]
                for future in as_completed(futures):
                    results.append(future.result())
            return results
