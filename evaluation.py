# evaluation.py
import torch
from ai_system import AISystem
from data_preprocessing import DataPreprocessor

class Evaluator:
    def __init__(self, data_dir, device):
        self.data_dir = data_dir
        self.device = device

        # Initialize the data preprocessor
        self.preprocessor = DataPreprocessor(self.data_dir, self.device)

        # Initialize the AI system
        self.ai_system = AISystem(device)

    def evaluate(self, video_path, audio_path, text_path):
        # Preprocess the data
        video_frames = self.preprocessor.preprocess_video(video_path)
        audio_data = self.preprocessor.preprocess_audio(audio_path)
        board_content = self.preprocessor.preprocess_text(text_path)

        # Generate the lecture summary
        lecture_summary = self.ai_system.generate_lecture_summary(video_frames, audio_data, board_content)

        # Extract the lecture topic
        lecture_topic = self.ai_system.extract_lecture_topic(board_content)

        # Find additional resources
        additional_resources = self.ai_system.find_additional_resources(lecture_topic)

        return lecture_summary, lecture_topic, additional_resources

if __:
    # Set the device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set the data directory
    data_dir = "data/"

    # Evaluate the AI system
    evaluator = Evaluator(data_dir, device)
    lecture_summary, lecture_topic, additional_resources = evaluator.evaluate("data/lecture_videos/video1.mp4",
                                                                              "data/lecture_audio/audio1.wav",
                                                                              "data/lecture_transcripts/transcript1.txt")
    print("Lecture Summary:", lecture_summary)
    print("Lecture Topic:", lecture_topic)
    print("Additional Resources:", additional_resources)
