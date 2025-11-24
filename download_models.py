from faster_whisper import download_model

# 저장할 경로 설정
model_dir = "/home/ubuntu/translater/models/whisper-large-v3-ct2"

print(f"Downloading Whisper Large-v3 to {model_dir}...")
# CTranslate2 형식으로 변환된 모델을 다운로드합니다.
model_path = download_model("large-v3", output_dir=model_dir)

print(f"Download Complete! Path: {model_path}")
