def emo_lt():
    # global emo_list, emo_trans
    emo_list = ['angry', 'happy', 'sad', 'surprise']
    emo_trans = {"happy": "开心的", "sad": "伤心的", "surprise": "惊讶的",
                 "neutral": "中性的", "angry": "愤怒的", "fear": "开心"}
    return emo_list, emo_trans


def reverse_emo(rand_emo, main_emo):
    """这个函数是定义反向的表情"""
    if rand_emo == "happy":
        if main_emo == "sad" or main_emo == "angry":
            return True
    if rand_emo == "angry":
        if main_emo == "happy" or main_emo == "fear":
            return True
    if rand_emo == "sad":
        if main_emo == "happy" or main_emo == "fear":
            return True
    if rand_emo == "neutral":
        if main_emo == "surprise":
            return True
    if rand_emo == "surprise":
        if main_emo == "neutral":
            return True

