# lecture_topic_extraction.py
class LectureTopicExtractionModule:
    def __init__(self, lang_model, tokenizer):
        self.lang_model = lang_model
        self.tokenizer = tokenizer

    def extract_topic(self, board_content):
        # Use the language model to extract the lecture topic from the board content
        input_ids = self.tokenizer.encode(board_content, return_tensors='pt')
        output, _ = self.lang_model(input_ids)
        topic_tokens = output.argmax(dim=-1)
        topic = self.tokenizer.decode(topic_tokens)
        return topic
