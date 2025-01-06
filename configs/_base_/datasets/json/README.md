# Instructions regarding Classes Initialization for end to end methods like E2ESpot

If you are working with new datasets, then you might need to change the classes list accordingly to adapt to that particular dataset labels.

For instance, when you are working with OSL-Action Spotting Dataset "spotting-OSL", it has 17 classes so you will have the classes as 
```python
classes = [
    "Ball out of play",
    "Clearance",
    "Corner",
    "Direct free-kick",
    "Foul",
    "Goal",
    "Indirect free-kick",
    "Kick-off",
    "Offside",
    "Penalty",
    "Red card",
    "Shots off target",
    "Shots on target",
    "Substitution",
    "Throw-in",
    "Yellow card",
    "Yellow->red card",
]
```
in the ```video_dali.py``` and ```video_ocv.py``` files.

But if you are working with, say [SN-Ball Action Spotting Dataset](https://huggingface.co/datasets/SoccerNet/SN-BAS-2025), then you need to make these changes in the ```video_dali.py``` and ```video_ocv.py``` files.

```python
[
    "PASS",
    "DRIVE",
    "HEADER",
    "HIGH PASS",
    "OUT",
    "CROSS",
    "THROW IN",
    "SHOT",
    "BALL PLAYER BLOCK",
    "PLAYER SUCCESSFUL TACKLE",
    "FREE KICK",
    "GOAL"
]
```