from sopel import module
from emo.wdemotions import EmotionDetector

emo = EmotionDetector()

@module.rule('')
def hi(bot, trigger):
    print(trigger, trigger.nick)
    array=emo.detect_emotion_in_raw_np(trigger)
    emotions=array.tolist()
    print(emotions)
    #bot.say('Hi, ' + trigger.nick)
