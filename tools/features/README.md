

# Extract features

## TF2_ResNET_PCA512
```bash
conda activate oslactionspotting
pip install scikit-video tensorflow==2.3.0 imutils opencv-python==3.4.11.41 SoccerNet moviepy scikit-learn ffmpy protobuf==3.20 ffmpeg ffmpy
```

```bash
python tools/features/extract_features.py --path_video path/to/video.mkv --path_features path/to/features.npy -PCA tools/features/pca_512_TF2.pkl --PCA_scaler tools/features/average_512_TF2.pkl
```

## FWC22

```bash
for i in {01..64}
do
    echo "Welcome $i times"
    python tools/features/extract_features.py --path_video /media/giancos/LaCie/FWC22/FWC2022_ISO_M${i}*mp4 --path_features /media/giancos/LaCie/FWC22/M${i}.npy --PCA tools/features/pca_512_TF2.pkl --PCA_scaler tools/features/average_512_TF2.pkl
done
```

## FFWC19

```bash
for i in {40..52}
do
    echo "Welcome $i times"
    python tools/features/extract_features.py --path_video /media/giancos/Football/SoccerNet_2019_FWWC/2019\ FWWC/M${i}*/*.mp4 --path_features /media/giancos/LaCie/FWWC19/M${i}_TF2_ResNET_PCA512.npy --PCA tools/features/pca_512_TF2.pkl --PCA_scaler tools/features/average_512_TF2.pkl
done
```
