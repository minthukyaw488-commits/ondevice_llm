"""
Anomaly-detection demo: baseline vs. recent.

Anomaly detection here means "how far has this person's RECENT behaviour drifted
from their OWN normal baseline". This script replays two weeks of history with
timestamps so the four quantified metrics can be seen firing:

  - conversation frequency drop
  - sentiment shift (mood worsening)
  - keyword shift (loneliness / sadness appearing)
  - response-length drop (disengagement)

Run:  python anomaly_demo.py
"""
import time

from src.abnormal_signal import AbnormalSignalDetector

DAY = 86400
now = time.time()

# Baseline (days -13 .. -4): frequent, cheerful, longer messages.
baseline_msgs = [
    "오늘 손자가 놀러 와서 정말 행복했어요, 맛있는 것도 많이 먹었어요",
    "복지관에서 친구들이랑 즐겁게 운동하고 왔어요, 기분이 좋네요",
    "날씨가 좋아서 공원 산책을 오래 했어요, 꽃이 참 예뻤어요",
    "이웃 분이랑 차 마시면서 얘기 많이 나눴어요, 고마운 하루였어요",
    "노래 교실에 다녀왔어요, 새 노래를 배워서 즐거웠어요",
    "손녀가 전화해서 한참 통화했어요, 목소리 들으니 든든해요",
    "텃밭에 상추가 잘 자라서 뿌듯해요, 이웃이랑 나눠 먹었어요",
    "경로당에서 윷놀이 하고 점심도 같이 먹었어요, 재밌었어요",
]

# Recent (days -2 .. 0): sparse, sad/lonely, short.
recent_msgs = [
    "그냥 그래요",
    "외로워요",
]

det = AbnormalSignalDetector()
print("sentiment backend:", det.sentiment.backend, "\n")

# Replay baseline: ~2 messages/day across days -11..-4 (frequent, engaged).
day = 11
count = 0
while day >= 4:
    for msg in baseline_msgs[:2]:                 # 2 messages that day
        det.add_utterance(msg, timestamp=now - day * DAY)
        count += 1
    baseline_msgs = baseline_msgs[2:] + baseline_msgs[:2]   # rotate for variety
    day -= 1

# Replay recent: only 2 messages in the last 2 days (sparse).
for i, msg in enumerate(recent_msgs):
    det.add_utterance(msg, timestamp=now - (2 - i) * DAY)

res = det.evaluate(now=now)

print("=" * 60)
print(" QUANTIFIED ANOMALY METRICS (recent vs baseline)")
print("=" * 60)
m = res.metrics
print(f"  baseline utterances : {m['baseline_n']}   recent utterances : {m['recent_n']}")
print(f"  conversation freq drop : {m['freq_drop']:.0%}")
print(f"  sentiment shift        : +{m['sentiment_shift']:.2f}")
print(f"  loneliness/sad keyword : +{m['keyword_shift']:.0%}")
print(f"  response length drop   : {m['length_drop']:.0%}")

print("\n" + "=" * 60)
print(f" DECISION: {'🚨 ABNORMAL' if res.is_abnormal else '정상'}")
print("=" * 60)
for r in res.reasons:
    print(f"  - {r}")
