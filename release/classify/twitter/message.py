
class Message:

    def __init__(self, s):
        self.raw_message = s

    def set_clean_words(self, clean_words):
        # self.words = list(set(clean_words))
        self.words = clean_words

