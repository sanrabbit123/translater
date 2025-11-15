from quart import Quart, request, jsonify
import ctranslate2
import transformers
import warnings
import asyncio
import redis.asyncio as redis
import json
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

warnings.filterwarnings("ignore", category=FutureWarning)

app = Quart(__name__)

# 전역 변수
translator = None
tokenizer = None
redis_client = None

MODEL_PATH = "/home/ubuntu/translater/nllb-1.3b-ct2"

# Redis 설정 (환경변수에서)
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_USERNAME = os.getenv("REDIS_USERNAME")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
REDIS_QUEUE_KEY = os.getenv("REDIS_QUEUE_KEY", "translation_results")

@app.before_serving
async def startup():
    """서버 시작 시 모델 및 Redis 로드"""
    global translator, tokenizer, redis_client
    
    print("=" * 60)
    print("모델 로딩 중...")
    
    # CTranslate2 모델 로드 (동기)
    translator = ctranslate2.Translator(
        MODEL_PATH,
        device="cuda",
        compute_type="float16"
    )
    
    # Tokenizer 로드
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "facebook/nllb-200-distilled-1.3B",
        src_lang="kor_Hang",
        clean_up_tokenization_spaces=True
    )

    redis_client = await redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        username=REDIS_USERNAME,
        password=REDIS_PASSWORD,
        decode_responses=True,
        socket_connect_timeout=5
    )

    # Redis 연결 테스트
    try:
        await redis_client.ping()
        print("Redis 연결 완료!")
    except Exception as e:
        print(f"Redis 연결 실패: {e}")
        raise
    
    print("모델 로딩 완료!")
    print("Redis 연결 완료!")
    print("=" * 60)

@app.after_serving
async def shutdown():
    """서버 종료 시 Redis 연결 해제"""
    if redis_client:
        await redis_client.close()

async def translate_text(text: str) -> str:
    """동기 번역 함수를 비동기로 실행"""
    def _translate():
        source = tokenizer.convert_ids_to_tokens(tokenizer.encode(text))
        results = translator.translate_batch(
            [source],
            target_prefix=[["eng_Latn"]],
            beam_size=4,
            max_batch_size=32
        )
        target_ids = tokenizer.convert_tokens_to_ids(results[0].hypotheses[0])
        return tokenizer.decode(target_ids, skip_special_tokens=True)
    
    # 동기 함수를 별도 스레드에서 실행
    return await asyncio.to_thread(_translate)

async def process_translation(original_text: str):
    """백그라운드에서 번역 처리 및 Redis에 저장"""
    try:
        # 문장 단위로 분리 (줄바꿈 기준)
        sentences = [s.strip() for s in original_text.split("\n") if s.strip()]
        
        results = []
        
        # 각 문장 번역
        for sentence in sentences:
            english = await translate_text(sentence)
            results.append({
                "original": sentence,
                "english": english
            })
        
        # Redis list에 결과 추가
        await redis_client.rpush(
            REDIS_QUEUE_KEY,
            json.dumps(results, ensure_ascii=False)
        )
        
        print(f"✅ 번역 완료 및 Redis 저장: {len(sentences)}개 문장")
        
    except Exception as e:
        print(f"❌ 번역 에러: {e}")

@app.route("/translate", methods=["POST"])
async def translate():
    """번역 요청 받기"""
    data = await request.get_json()
    
    if not data or "text" not in data:
        return jsonify({"error": "text field required"}), 400
    
    korean_text = data["text"]
    
    # 즉시 응답
    response = jsonify({"message": "will do"})
    
    # 백그라운드에서 번역 처리
    asyncio.create_task(process_translation(korean_text))
    
    return response, 202

@app.route("/results", methods=["GET"])
async def get_results():
    """Redis 큐에서 결과 가져오기 (테스트용)"""
    count = await redis_client.llen(REDIS_QUEUE_KEY)
    
    if count == 0:
        return jsonify({"results": [], "count": 0})
    
    # 모든 결과 가져오기
    results = await redis_client.lrange(REDIS_QUEUE_KEY, 0, -1)
    parsed_results = [json.loads(r) for r in results]
    
    return jsonify({
        "results": parsed_results,
        "count": count
    })

@app.route("/results/pop", methods=["GET"])
async def pop_result():
    """Redis 큐에서 결과 하나 꺼내기"""
    result = await redis_client.lpop(REDIS_QUEUE_KEY)
    
    if result is None:
        return jsonify({"result": None, "message": "queue empty"})
    
    return jsonify({
        "result": json.loads(result)
    })

@app.route("/health", methods=["GET"])
async def health():
    """헬스체크"""
    return jsonify({
        "status": "ok",
        "model_loaded": translator is not None,
        "redis_connected": redis_client is not None
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=61888, debug=False)
