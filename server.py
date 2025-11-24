from quart import Quart, request, jsonify
import ctranslate2
import transformers
import warnings
import asyncio
import redis.asyncio as redis
import json
import os
import uuid
from dotenv import load_dotenv
from faster_whisper import WhisperModel

load_dotenv()
warnings.filterwarnings("ignore", category=FutureWarning)
app = Quart(__name__)

translator = None
tokenizer = None
redis_client = None
stt_model = None

MODEL_PATH = "/home/ubuntu/translater/nllb-1.3b-ct2"
STT_MODEL_PATH = "/home/ubuntu/translater/models/whisper-large-v3-ct2"
UPLOAD_DIR = "/tmp/stt_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_USERNAME = os.getenv("REDIS_USERNAME")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
REDIS_QUEUE_KEY = os.getenv("REDIS_QUEUE_KEY", "translation_results")
REDIS_STT_QUEUE_KEY = os.getenv("REDIS_STT_QUEUE_KEY", "stt_results")

gpu_lock = asyncio.Lock()

@app.before_serving
async def startup():
    global translator, tokenizer, redis_client, stt_model
    
    print("=" * 60)
    print("Initializing AI Models...")
    
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
    stt_model = WhisperModel(
        STT_MODEL_PATH,
        device="cuda",
        compute_type="float16",
        local_files_only=True
    )

    redis_client = await redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        username=REDIS_USERNAME,
        password=REDIS_PASSWORD,
        decode_responses=True,
        socket_connect_timeout=5
    )

    try:
        await redis_client.ping()
    except Exception as e:
        print(f"Redis fail: {e}")
        raise
    
    print("model ready complete")
    print("=" * 60)

@app.after_serving
async def shutdown():
    if redis_client:
        await redis_client.close()

async def translate_text(text: str) -> str:
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
    
    async with gpu_lock:
        return await asyncio.to_thread(_translate)

async def process_translation(original_text: str, task_id: str):
    try:
        sentences = [s.strip() for s in original_text.split("\n") if s.strip()]
        results = []
        for sentence in sentences:
            english = await translate_text(sentence)
            results.append({
                "id": task_id,
                "original": sentence,
                "english": english
            })
        
        await redis_client.rpush(
            REDIS_QUEUE_KEY,
            json.dumps(results, ensure_ascii=False)
        )
        
    except Exception as e:
        print(f"error occur: {e}")

def _transcribe_sync(file_path):
    segments, info = stt_model.transcribe(
        file_path, 
        language="ko",
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500)
    )
    
    results = []
    for segment in segments:
        results.append({
            "start": round(segment.start, 2),
            "end": round(segment.end, 2),
            "text": segment.text.strip()
        })
    return {
        "language": info.language,
        "duration": round(info.duration, 2),
        "segments": results
    }

async def process_stt(file_path: str, task_id: str, original_filename: str):
    try:
        async with gpu_lock:
            result_data = await asyncio.to_thread(_transcribe_sync, file_path)            
        final_output = {
            "id": task_id,
            "type": "stt",
            "filename": original_filename,
            "status": "completed",
            "data": result_data
        }
        await redis_client.rpush(REDIS_STT_QUEUE_KEY, json.dumps(final_output, ensure_ascii=False))        
        if os.path.exists(file_path):
            os.remove(file_path)
        
    except Exception as e:
        print(f"STT Error: {e}")
        error_output = {
            "id": task_id,
            "type": "stt",
            "status": "failed",
            "error": str(e)
        }
        await redis_client.rpush(REDIS_STT_QUEUE_KEY, json.dumps(error_output, ensure_ascii=False))

@app.route("/translate", methods=["POST"])
async def translate():
    data = await request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "text field required"}), 400
    
    korean_text = data["text"]
    task_id = str(uuid.uuid4())
    response = jsonify({
        "message": "will do",
        "id": task_id
    })

    asyncio.create_task(process_translation(korean_text, task_id))
    return response, 202

@app.route("/results", methods=["GET"])
async def get_results():
    count = await redis_client.llen(REDIS_QUEUE_KEY)
    if count == 0:
        return jsonify({"results": [], "count": 0})
    
    results = await redis_client.lrange(REDIS_QUEUE_KEY, 0, -1)
    parsed_results = [json.loads(r) for r in results]
    
    return jsonify({
        "results": parsed_results,
        "count": count
    })

@app.route("/results/pop", methods=["GET"])
async def pop_result():
    result = await redis_client.lpop(REDIS_QUEUE_KEY)
    if result is None:
        return jsonify({"result": None, "message": "queue empty"})
    
    return jsonify({
        "result": json.loads(result)
    })

@app.route("/stt", methods=["POST"])
async def stt_endpoint():
    data = await request.get_json()
    
    if not data or "path" not in data:
        return jsonify({"error": "json body with 'path' field required"}), 400
    
    file_path = data["path"]    
    if not os.path.exists(file_path):
        return jsonify({"error": f"File not found on server: {file_path}"}), 404

    task_id = str(uuid.uuid4())
    original_filename = os.path.basename(file_path)
    
    response = jsonify({
        "message": "stt processing started", 
        "id": task_id,
        "target_file": file_path
    })    
    asyncio.create_task(process_stt(file_path, task_id, original_filename))
    return response, 202

@app.route("/stt/results", methods=["GET"])
async def get_stt_results():
    count = await redis_client.llen(REDIS_STT_QUEUE_KEY)
    if count == 0: return jsonify({"results": [], "count": 0})
    results = await redis_client.lrange(REDIS_STT_QUEUE_KEY, 0, -1)
    return jsonify({"results": [json.loads(r) for r in results], "count": count})

@app.route("/stt/results/pop", methods=["GET"])
async def pop_stt_result():
    result = await redis_client.lpop(REDIS_STT_QUEUE_KEY)
    if result is None: return jsonify({"result": None, "message": "queue empty"})
    return jsonify({"result": json.loads(result)})

@app.route("/health", methods=["GET"])
async def health():
    return jsonify({
        "status": "ok",
        "model_loaded": translator is not None,
        "whisper_loaded": stt_model is not None,
        "redis_connected": redis_client is not None
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=61888, debug=False)
