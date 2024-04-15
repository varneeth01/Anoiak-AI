# lecture_management.py


import os
import pickle

class LectureManager:
    def __init__(self, data_dir="data/"):
        self.data_dir = data_dir
        self.lecture_index = self.build_lecture_index()

    def build_lecture_index(self):
        lecture_index = {}
        for filename in os.listdir(self.data_dir):
            if filename.endswith(".pkl"):
                lecture_id = os.path.splitext(filename)[0]
                lecture_index[lecture_id] = os.path.join(self.data_dir, filename)
        return lecture_index

    def save_lecture(self, lecture_id, video_frames, audio_data, board_content, student_engagement_levels):
        lecture_data = {
            "video_frames": video_frames,
            "audio_data": audio_data,
            "board_content": board_content,
            "student_engagement_levels": student_engagement_levels
        }
        with open(os.path.join(self.data_dir, f"{lecture_id}.pkl"), "wb") as f:
            pickle.dump(lecture_data, f)

    def load_lecture(self, lecture_id):
        if lecture_id in self.lecture_index:
            lecture_path = self.lecture_index[lecture_id]
            with open(lecture_path, "rb") as f:
                lecture_data = pickle.load(f)
            return lecture_data["video_frames"], lecture_data["audio_data"], lecture_data["board_content"], lecture_data["student_engagement_levels"]
        else:
            return None, None, None, None
