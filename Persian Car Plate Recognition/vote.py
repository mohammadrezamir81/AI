from collections import Counter, defaultdict


class Vote:
    def __init__(self, max_frames=10, num_chars=8):
        self.max_frames = max_frames
        self.num_chars = num_chars
        self.predictions = defaultdict(list)
        self.final_results = {}
        

    def add_prediction(self, plate_id, predicted_chars):
        if self.is_finalized(plate_id):
            return  

        self.predictions[plate_id].append(predicted_chars)
        current_count = len(self.predictions[plate_id])
        print(f"[+] ID:{plate_id} → Added prediction {current_count}/{self.max_frames}: {''.join(predicted_chars)}")

        if current_count >= self.max_frames:
            self._finalize_result(plate_id)

    def _finalize_result(self, plate_id):
        if self.is_finalized(plate_id):
            return

        all_predictions = self.predictions[plate_id]

        voted_chars = []
        for char_pos in range(self.num_chars):
            chars_at_pos = [pred[char_pos] for pred in all_predictions if len(pred) > char_pos]
            if not chars_at_pos:
                continue

            counter = Counter(chars_at_pos)
            most_common, freq = counter.most_common(1)[0]
            confidence = freq / len(all_predictions)

            if confidence < 0.6:
                print(f" Warning: Low confidence at position {char_pos}: {confidence*100:.1f}%")

            voted_chars.append(most_common)

        final_text = ''.join(voted_chars)
        self.final_results[plate_id] = final_text
        del self.predictions[plate_id]

        print(f"\n✅ Voting done: ID:{plate_id} → {final_text}")
        print("=" * 60)

    def is_finalized(self, plate_id):
        return plate_id in self.final_results

    def get_final_result(self, plate_id):

        if self.is_finalized(plate_id):
            return self.final_results[plate_id]  
        
   
        current = len(self.predictions[plate_id])
        if self.max_frames > 0:
            progress = current / self.max_frames
        else:
            progress = 0
        return (f"{(progress * 100):.2f}%")  

    def get_prediction_count(self, plate_id):
        if self.is_finalized(plate_id):
            return self.max_frames
        return len(self.predictions[plate_id])

