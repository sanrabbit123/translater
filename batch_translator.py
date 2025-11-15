import ctranslate2
import transformers
import time
import warnings

# 경고 무시
warnings.filterwarnings("ignore", category=FutureWarning)

MODEL_PATH = "/home/ubuntu/translater/nllb-1.3b-ct2"

print("모델 로딩 중...")
start = time.time()

translator = ctranslate2.Translator(
    MODEL_PATH,
    device="cuda",
    compute_type="float16"
)

tokenizer = transformers.AutoTokenizer.from_pretrained(
    "facebook/nllb-200-distilled-1.3B",
    src_lang="kor_Hang",
    clean_up_tokenization_spaces=True
)

print(f"모델 로딩 완료: {time.time() - start:.2f}초\n")

def batch_translate(korean_texts: list):
    """여러 문장을 한번에 번역 (빠름!)"""
    # 모든 문장 토큰화
    sources = [
        tokenizer.convert_ids_to_tokens(tokenizer.encode(text))
        for text in korean_texts
    ]
    
    # 배치 번역
    results = translator.translate_batch(
        sources,
        target_prefix=[["eng_Latn"]] * len(sources),
        beam_size=4,
        max_batch_size=32
    )
    
    # 디코딩
    translations = []
    for result in results:
        target_ids = tokenizer.convert_tokens_to_ids(result.hypotheses[0])
        translations.append(tokenizer.decode(target_ids, skip_special_tokens=True))
    
    return translations

# 테스트 - 소설 문단
novel_paragraphs = [
    "그는 천천히 문을 열었다.",
    "방 안에는 아무도 없었다.",
    "책상 위에 낡은 편지가 놓여 있었다.",
    "그는 편지를 집어들었다.",
    "봉투는 누렇게 바래 있었다.",
    "그의 손이 떨렸다.",
    "무엇이 적혀 있을까?",
    "그는 천천히 편지를 펼쳤다.",
    "첫 줄을 읽는 순간 그는 얼어붙었다.",
    "이것은 그가 잃어버렸던 과거였다.",
    "어머니의 필체가 눈앞에 선명했다.",
    "그는 눈물을 참을 수 없었다.",
    "편지는 20년 전에 쓰여진 것이었다.",
    "그때 그는 너무 어렸다.",
    "이해할 수 없었던 많은 것들이 이제야 이해되기 시작했다."
]

print(f"총 {len(novel_paragraphs)}개 문장 배치 번역 시작...\n")
start = time.time()

translations = batch_translate(novel_paragraphs)

elapsed = time.time() - start

print("=== 번역 결과 ===")
for i, (ko, en) in enumerate(zip(novel_paragraphs, translations), 1):
    print(f"\n{i}. {ko}")
    print(f"   → {en}")

print(f"\n{'='*60}")
print(f"총 소요시간: {elapsed:.2f}초")
print(f"평균 속도: {elapsed/len(novel_paragraphs):.3f}초/문장")
print(f"총 {len(novel_paragraphs)}문장을 {elapsed:.2f}초에 처리!")
