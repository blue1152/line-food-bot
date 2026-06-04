import os
import json
import logging
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from urllib.error import HTTPError, URLError
from urllib.parse import quote_plus, urlencode
from urllib.request import Request as UrlRequest, urlopen
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import (
    Configuration,
    ApiException,
    ApiClient,
    MessagingApi,
    PushMessageRequest,
    ReplyMessageRequest,
    TextMessage
)
from linebot.v3.webhooks import MessageEvent, TextMessageContent, LocationMessageContent, GroupSource, RoomSource, UserSource
from google import genai
from google.genai import types

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI()

# --- 1. 初始化與環境變數 ---
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
GEMINI_FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-3.5-flash")
GEMINI_MAPS_MODEL = os.getenv("GEMINI_MAPS_MODEL") or GEMINI_MODEL
GEMINI_MAPS_FALLBACK_MODEL = os.getenv("GEMINI_MAPS_FALLBACK_MODEL") or GEMINI_FALLBACK_MODEL
MAPS_TOOLS_API_KEY = os.getenv("MAPS_GROUNDING_API_KEY") or os.getenv("GOOGLE_MAPS_API_KEY") or GEMINI_API_KEY
LINE_BOT_USER_ID = os.getenv("LINE_BOT_USER_ID")

missing_envs = [
    name for name, value in {
        "LINE_CHANNEL_SECRET": LINE_CHANNEL_SECRET,
        "LINE_CHANNEL_ACCESS_TOKEN": LINE_CHANNEL_ACCESS_TOKEN,
        "GEMINI_API_KEY": GEMINI_API_KEY,
        "GEMINI_MODEL": GEMINI_MODEL,
        "GEMINI_FALLBACK_MODEL": GEMINI_FALLBACK_MODEL,
        "GEMINI_MAPS_MODEL": GEMINI_MAPS_MODEL,
        "GEMINI_MAPS_FALLBACK_MODEL": GEMINI_MAPS_FALLBACK_MODEL
    }.items()
    if not value
]
if missing_envs:
    raise RuntimeError(f"Missing required environment variables: {', '.join(missing_envs)}")

configuration = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)
gemini_client = genai.Client(
    api_key=GEMINI_API_KEY,
    http_options=types.HttpOptions(
        timeout=45000,
        retry_options=types.HttpRetryOptions(attempts=1)
    )
)
gemini_executor = ThreadPoolExecutor(max_workers=4)

class GroundingUnavailableError(Exception):
    pass

# --- 2. 記憶庫：暫存最後發送的位置資訊 ---
group_location_cache = {}
user_location_cache = {}
processed_event_cache = {}
resolved_place_id_cache = {}
event_cache_lock = threading.Lock()
bot_user_id_lock = threading.Lock()
resolution_cache_lock = threading.Lock()
LOCATION_CACHE_TTL_SECONDS = 2 * 60 * 60
PROCESSED_EVENT_TTL_SECONDS = 10 * 60
LOCATION_REQUIRED_MESSAGE = (
    "請告訴我地點資訊（例如城市、區域、店名、景點或地址），或先用 LINE 傳送「位置資訊」。\n"
    "沒有地點的話，我沒辦法精準回答。"
)
AI_REPLY_TIMEOUT_SECONDS = 5
AI_TOTAL_TIMEOUT_SECONDS = 45

LOCATION_CONTEXT_KEYWORDS = ("附近", "周邊", "這裡", "這附近", "周圍", "nearby", "near me", "around me")
RELATIVE_LOCATION_VALUES = LOCATION_CONTEXT_KEYWORDS + ("here", "current location", "my location")
PLACE_QUERY_KEYWORDS = (
    "餐廳", "美食", "咖啡", "咖啡廳", "小吃", "宵夜", "早餐", "午餐", "晚餐", "甜點",
    "酒吧", "景點", "旅遊", "行程", "住宿", "飯店", "旅館", "夜市", "伴手禮",
    "restaurant", "food", "cafe", "coffee", "bar", "hotel", "attraction"
)
FOOD_QUERY_KEYWORDS = (
    "牛肉麵", "拉麵", "火鍋", "燒肉", "壽司", "咖哩", "便當", "滷肉飯", "雞排",
    "甜點", "下午茶", "早餐", "午餐", "晚餐", "宵夜", "飲料", "咖啡", "小吃",
    "noodle", "ramen", "hotpot", "sushi", "dessert", "breakfast", "lunch", "dinner",
    "bbq", "barbecue", "pizza", "burger", "taco", "brunch", "bakery", "bistro"
)
MAPS_KEYWORDS = LOCATION_CONTEXT_KEYWORDS + PLACE_QUERY_KEYWORDS + FOOD_QUERY_KEYWORDS
SEARCH_KEYWORDS = (
    "最新", "活動", "展覽", "演唱會", "市集", "施工", "營業異動", "臨時休息", "新聞", "票價", "時間表",
    "latest", "news", "event", "exhibition", "concert", "market", "construction", "schedule"
)
RECIPE_KEYWORDS = ("食譜", "做法", "怎麼做", "料理", "煮法", "recipe", "cook", "cooking", "how to make")
BOT_KEYWORDS = ("@美食家", "吃什麼")
GENERIC_PLACE_QUERY_WORDS = (
    "推薦", "找", "查", "有什麼", "哪裡", "哪家", "好吃", "好喝", "適合", "附近",
    "幫我", "請問", "想吃", "想喝", "一下", "的", "和", "跟",
    "recommend", "find", "search", "good", "best", "nearby", "near me"
)

# --- 3. 修正版角色設定（依 grounding 類型指定來源格式） ---
FOODIE_SYSTEM_INSTRUCTION = (
    "你是一位兼具旅遊專家、城市探險家與頂級美食家的嚮導；你熟悉景點、活動、路線、在地文化與臨場探索，但美食判斷是你的最強項，說話風趣刁鑽。\n\n"
    "【重要時空與真實性限制】：\n"
    "1. 若系統提供 Google Maps 或 Google Search grounding 資料，請優先依據 grounding 資料回答。\n"
    "2. 嚴格禁止虛構、捏造、想像任何不存在的店家、景點或活動名稱。\n"
    "3. 推薦店家或景點時，必須以可從 grounding 資料、使用者提供的位置、或明確地名合理查證的真實地點為準。\n"
    "4. 若 grounding 資料不足、地點不明確，或無法確認店家仍在營業，請直接說明資料不足，寧可少推薦，也不要湊數。\n"
    "5. 不要宣稱你已查到即時營業狀態，除非 grounding 資料中明確支持。\n\n"
    "【回覆格式約束】：\n"
    "0. 回覆使用者的時候，針對一般性回答，像個朋友，溫和而不帶侮辱字眼。只有針對美食/地點下評語時，才要毒舌。\n"
    "1. 前言必須極其簡短（嚴格限制在 30 字以內），直接切入主題，拒絕任何廢話。\n"
    "2. 使用點排列（Bullet points）呈現推薦項目；在 grounding 資料充足時，請盡量多給結果，優先提供 8 到 12 個真實可查證項目，不要只給 3 到 5 個。\n"
    "2-1. 每個推薦項目之間必須空一行，方便系統分段傳送；同一個項目的店名、評語、連結和指數必須放在同一段。\n"
    "3. 如果推薦的是店名、餐廳、景點或明確地點，每個推薦項目的結構必須嚴格遵守：\n"
    "店名/景點名\n"
    "【評語】：只能用一句話（20字內）精闢點出靈魂美味或特色，語氣要毒舌風趣。\n"
    "【Google 地圖】：請讓系統自動為每家店生成對應的 Google 搜尋聯網腳註標記（Attribution Sources）。\n"
    "【推薦指數】：滿分5顆星（如：★★★★☆）。\n"
    "4. 如果推薦的是活動、展覽、演唱會、市集、新聞、施工、營業異動、票價或時間表等 Google Search 查到的結果，嚴禁提供 Google 地圖連結，改用此格式：\n"
    "活動/結果名稱\n"
    "【評語】：只能用一句話（20字內）點出重點，語氣可犀利但不要侮辱。\n"
    "【資料來源】：必須使用 Google Search grounding 查到的原始網址。\n"
    "【推薦指數】：滿分5顆星（如：★★★★☆）。"
)

LOCATION_DETECTION_SYSTEM_INSTRUCTION = (
    "你是地點資訊抽取器。只判斷使用者文字中是否包含可判定為地點的資訊。\n"
    "地點可以是城市、國家、區域、街道、地址、店名、餐廳名、景點、地標、場館、Google Maps URL 或經緯度。\n"
    "食物品項、料理類型、活動類型、抽象需求不算地點，例如：拉麵、咖啡、展覽、市集、推薦美食。\n"
    "如果文字只有「附近、這裡、周邊、near me」這類相對位置，且沒有實際地點名稱，has_location 必須是 false。\n"
    "只回傳 JSON，格式必須是：{\"has_location\": boolean, \"locations\": [string]}。"
)

# --- 4. Webhook 入口 ---
@app.get("/")
async def health_check():
    return {"status": "ok"}

@app.post("/webhook")
async def webhook(request: Request, background_tasks: BackgroundTasks):
    signature = request.headers.get("X-Line-Signature")
    if not signature:
        raise HTTPException(status_code=400, detail="Missing Signature")
    
    body = await request.body()
    body_str = body.decode("utf-8")

    background_tasks.add_task(handle_line_request, body_str, signature)
    return "OK"

def handle_line_request(body: str, signature: str):
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        logger.warning("Invalid LINE signature.")
    except Exception:
        logger.exception("Unhandled error while handling LINE webhook.")

def cleanup_expired_processed_events(now: float):
    expired_event_ids = [
        event_id
        for event_id, expires_at in processed_event_cache.items()
        if expires_at <= now
    ]
    for event_id in expired_event_ids:
        processed_event_cache.pop(event_id, None)

def get_webhook_event_id(event: MessageEvent) -> str | None:
    for attr_name in ("webhook_event_id", "webhookEventId"):
        event_id = getattr(event, attr_name, None)
        if event_id:
            return event_id

    for serializer_name in ("to_dict", "model_dump"):
        serializer = getattr(event, serializer_name, None)
        if not serializer:
            continue
        try:
            event_data = serializer()
        except Exception:
            continue
        event_id = event_data.get("webhookEventId") or event_data.get("webhook_event_id")
        if event_id:
            return event_id

    return None

def should_skip_event(event: MessageEvent) -> bool:
    event_id = get_webhook_event_id(event)
    if not event_id:
        return False

    now = time.time()
    with event_cache_lock:
        cleanup_expired_processed_events(now)

        if event_id in processed_event_cache:
            logger.info("Skip duplicated webhook event. webhook_event_id=%s", event_id)
            return True

        processed_event_cache[event_id] = now + PROCESSED_EVENT_TTL_SECONDS
    return False

def store_location(cache: dict, cache_key: str, location: dict):
    cache[cache_key] = {
        "location": location,
        "expires_at": time.time() + LOCATION_CACHE_TTL_SECONDS
    }

def get_cached_location(cache: dict, cache_key: str):
    cached = cache.get(cache_key)
    if not cached:
        return None

    if cached["expires_at"] <= time.time():
        cache.pop(cache_key, None)
        return None

    return cached["location"]

def clear_cached_location(cache: dict, cache_key: str | None):
    if cache_key:
        cache.pop(cache_key, None)

def detect_locations_in_text(user_query: str) -> dict:
    query = user_query.strip()
    if not query:
        return {"has_location": False, "locations": []}

    prompt = (
        "從以下文字提取地點，回傳 JSON {\"has_location\": boolean, \"locations\": string[]}。\n"
        "只把可作為 Google Maps / Google Search 地理查詢依據的資訊放進 locations。\n\n"
        f"文字：{query}"
    )
    config = types.GenerateContentConfig(
        system_instruction=LOCATION_DETECTION_SYSTEM_INSTRUCTION,
        temperature=0.0,
        max_output_tokens=300,
        response_mime_type="application/json"
    )

    try:
        response = generate_location_detection_content(prompt, config, query)
        detection = parse_location_detection_response(response.text)
        logger.info("Location detection. query=%s detection=%s", query, detection)
        return detection
    except Exception:
        logger.exception("Location detection failed. query=%s", query)
        return {"has_location": False, "locations": []}

def generate_location_detection_content(prompt: str, config, user_query: str):
    models = [GEMINI_MODEL]
    if GEMINI_FALLBACK_MODEL != GEMINI_MODEL:
        models.append(GEMINI_FALLBACK_MODEL)

    last_error = None
    for model in models:
        try:
            return gemini_client.models.generate_content(
                model=model,
                contents=prompt,
                config=config
            )
        except Exception as e:
            last_error = e
            logger.warning("Location detection model failed. model=%s query=%s error=%s", model, user_query, e)

    raise last_error

def parse_location_detection_response(response_text: str | None) -> dict:
    if not response_text:
        return {"has_location": False, "locations": []}

    text = response_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    if not text.startswith("{"):
        json_start = text.find("{")
        json_end = text.rfind("}")
        if json_start >= 0 and json_end > json_start:
            text = text[json_start:json_end + 1]

    data = json.loads(text)
    if not isinstance(data, dict):
        return {"has_location": False, "locations": []}
    locations = data.get("locations") or []
    if not isinstance(locations, list):
        locations = []

    clean_locations = [
        str(location).strip()
        for location in locations
        if str(location).strip() and str(location).strip().lower() not in RELATIVE_LOCATION_VALUES
    ]
    return {
        "has_location": bool(data.get("has_location")) and bool(clean_locations),
        "locations": clean_locations
    }

def has_detected_location(location_detection: dict | None) -> bool:
    return bool(location_detection and location_detection.get("has_location") and location_detection.get("locations"))

def build_prompt_with_detected_locations(user_query: str, location_detection: dict) -> str:
    locations = "、".join(location_detection.get("locations", []))
    return (
        f"使用者詢問：『{user_query}』\n"
        f"NLP 偵測到的地點資訊：{locations}\n"
        "請優先以這些地點資訊作為查詢與回答依據。"
    )

def needs_location(user_query: str) -> bool:
    query = user_query.lower()
    return any(keyword.lower() in query for keyword in LOCATION_CONTEXT_KEYWORDS)

def get_conversation_id(source) -> str | None:
    if isinstance(source, GroupSource):
        return source.group_id
    if isinstance(source, RoomSource):
        return source.room_id
    return None

def get_message_target_id(source) -> str | None:
    if isinstance(source, UserSource):
        return source.user_id
    return get_conversation_id(source)

def get_bot_user_id() -> str | None:
    global LINE_BOT_USER_ID

    if LINE_BOT_USER_ID:
        return LINE_BOT_USER_ID

    with bot_user_id_lock:
        if LINE_BOT_USER_ID:
            return LINE_BOT_USER_ID

        try:
            with ApiClient(configuration) as api_client:
                line_bot_api = MessagingApi(api_client)
                bot_info = line_bot_api.get_bot_info()
                LINE_BOT_USER_ID = bot_info.user_id
                logger.info("Fetched LINE bot user id.")
                return LINE_BOT_USER_ID
        except Exception:
            logger.exception("Failed to fetch LINE bot user id.")
            return None

def get_grounding_kind(user_query: str) -> str | None:
    query = user_query.lower()
    if any(keyword.lower() in query for keyword in RECIPE_KEYWORDS):
        return None
    if any(keyword.lower() in query for keyword in SEARCH_KEYWORDS):
        return "search"
    if any(keyword.lower() in query for keyword in MAPS_KEYWORDS):
        return "maps"
    return None

def uses_grounding(user_query: str) -> bool:
    return get_grounding_kind(user_query) is not None

def mention_to_dict(mention) -> dict | None:
    if not mention:
        return None

    if isinstance(mention, dict):
        return mention
    if hasattr(mention, "to_dict"):
        return mention.to_dict()
    if hasattr(mention, "dict"):
        return mention.dict()
    if hasattr(mention, "model_dump"):
        return mention.model_dump()
    return None

def extract_prompt_from_bot_mention(event: MessageEvent) -> tuple[bool, str]:
    bot_user_id = get_bot_user_id()
    if not bot_user_id:
        return False, event.message.text.strip()

    mention_data = mention_to_dict(getattr(event.message, "mention", None))
    mentionees = mention_data.get("mentionees", []) if mention_data else []
    bot_mentions = [
        mentionee for mentionee in mentionees
        if mentionee.get("userId") == bot_user_id or mentionee.get("user_id") == bot_user_id
    ]
    if not bot_mentions:
        return False, event.message.text.strip()

    clean_prompt = remove_mention_ranges(event.message.text, bot_mentions)
    return True, clean_prompt.strip()

def remove_mention_ranges(text: str, mentionees: list[dict]) -> str:
    clean_text = text
    ranges = []
    for mentionee in mentionees:
        index = mentionee.get("index")
        length = mentionee.get("length")
        if isinstance(index, int) and isinstance(length, int):
            ranges.append((index, index + length))

    for start, end in sorted(ranges, reverse=True):
        clean_text = clean_text[:start] + clean_text[end:]

    return clean_text

def extract_group_prompt(user_message: str, event: MessageEvent | None = None) -> tuple[bool, str]:
    if event:
        is_mentioned, clean_prompt = extract_prompt_from_bot_mention(event)
        if is_mentioned:
            return True, clean_prompt

    for keyword in BOT_KEYWORDS:
        if keyword in user_message:
            return True, user_message.replace(keyword, "", 1).strip()
    return False, user_message.strip()

def log_mention_metadata(event: MessageEvent):
    mention = getattr(event.message, "mention", None)
    if not mention:
        return

    mention_data = mention_to_dict(mention) or repr(mention)

    logger.info("LINE mention metadata: %s", mention_data)

# --- 5. 處理「位置」訊息事件 (只紀錄，不回應) ---
@handler.add(MessageEvent, message=LocationMessageContent)
def handle_location_message(event: MessageEvent):
    if should_skip_event(event):
        return

    is_in_group = isinstance(event.source, (GroupSource, RoomSource))
    location = {
        "title": event.message.title if event.message.title else "未知起點",
        "address": event.message.address,
        "latitude": event.message.latitude,
        "longitude": event.message.longitude
    }
    
    if is_in_group:
        group_id = get_conversation_id(event.source)
        store_location(group_location_cache, group_id, location)
        logger.info("Stored group location. group_id=%s", group_id)
    elif isinstance(event.source, UserSource):
        store_location(user_location_cache, event.source.user_id, location)
        logger.info("Stored user location. user_id=%s", event.source.user_id)

# --- 6. 處理「文字」訊息事件 (被 Tag 時觸發) ---
@handler.add(MessageEvent, message=TextMessageContent)
def handle_text_message(event: MessageEvent):
    if should_skip_event(event):
        return

    user_message = event.message.text
    reply_token = event.reply_token
    log_mention_metadata(event)
    
    is_in_group = isinstance(event.source, (GroupSource, RoomSource))
    target_id = get_message_target_id(event.source)

    if is_in_group:
        is_triggered, clean_prompt = extract_group_prompt(user_message, event)
        if not is_triggered:
            return 
        
        group_id = get_conversation_id(event.source)
        cached_location = get_cached_location(group_location_cache, group_id)
        location_detection = detect_locations_in_text(clean_prompt)
        if cached_location and has_detected_location(location_detection) and not needs_location(clean_prompt):
            clear_cached_location(group_location_cache, group_id)
            cached_location = None
            logger.info("Cleared group location because query has explicit place. group_id=%s", group_id)
        
        if cached_location:
            final_prompt = (
                f"使用者在群組中標記了你，並詢問：『{clean_prompt}』\n"
                f"請結合該群組最後提供的定位資訊與周邊進行回答：\n"
                f"定位名稱: {cached_location['title']}\n"
                f"地址: {cached_location['address']}\n"
                f"座標: 緯度 {cached_location['latitude']}, 經度 {cached_location['longitude']}"
            )
        else:
            if not has_detected_location(location_detection):
                send_line_reply_or_push(reply_token, LOCATION_REQUIRED_MESSAGE, target_id)
                return
            final_prompt = build_prompt_with_detected_locations(clean_prompt, location_detection)
    else:
        user_id = event.source.user_id if isinstance(event.source, UserSource) else None
        cached_location = get_cached_location(user_location_cache, user_id) if user_id else None
        location_detection = detect_locations_in_text(user_message)
        if cached_location and has_detected_location(location_detection) and not needs_location(user_message):
            clear_cached_location(user_location_cache, user_id)
            cached_location = None
            logger.info("Cleared user location because query has explicit place. user_id=%s", user_id)
        
        if cached_location:
            final_prompt = (
                f"使用者在一對一聊天室詢問：『{user_message}』\n"
                f"請結合使用者最後提供的定位資訊與周邊進行回答：\n"
                f"定位名稱: {cached_location['title']}\n"
                f"地址: {cached_location['address']}\n"
                f"座標: 緯度 {cached_location['latitude']}, 經度 {cached_location['longitude']}"
            )
        else:
            if not has_detected_location(location_detection):
                send_line_reply_or_push(reply_token, LOCATION_REQUIRED_MESSAGE, target_id)
                return
            final_prompt = build_prompt_with_detected_locations(user_message, location_detection)

    user_query = clean_prompt if is_in_group else user_message
    ai_response = ask_gemini_with_optional_wait(
        final_prompt,
        user_query,
        cached_location,
        reply_token,
        target_id
    )
    if ai_response:
        send_line_reply_or_push(reply_token, ai_response, target_id)

# --- 7. Gemini API 呼叫邏輯 ---
def ask_gemini_with_optional_wait(
    prompt: str,
    user_query: str,
    location: dict | None,
    reply_token: str,
    target_id: str | None
) -> str | None:
    if not target_id or not uses_grounding(user_query):
        return ask_gemini_foodie(prompt, user_query, location)

    future = gemini_executor.submit(ask_gemini_foodie, prompt, user_query, location)
    try:
        return future.result(timeout=AI_REPLY_TIMEOUT_SECONDS)
    except TimeoutError:
        send_line_reply_or_push(reply_token, f"{user_query}, 還在查詢中...", target_id)
        try:
            ai_response = future.result(timeout=AI_TOTAL_TIMEOUT_SECONDS - AI_REPLY_TIMEOUT_SECONDS)
            send_line_push(target_id, ai_response)
        except TimeoutError:
            logger.warning("Gemini query timed out. query=%s", user_query)
            send_line_push(target_id, "查詢失敗，請再試一次。")
        except Exception:
            logger.exception("Gemini async query failed.")
            send_line_push(target_id, "查詢失敗，請再試一次。")
        return None

def ask_gemini_foodie(prompt: str, user_query: str, location: dict | None = None) -> str:
    try:
        config = build_gemini_config(user_query, location)
        response = generate_gemini_content(prompt, config, user_query)
        return format_gemini_response(response, user_query)
    except GroundingUnavailableError:
        logger.exception("Grounding unavailable.")
        return "查不到可驗證的 Google 來源，請換個更明確的地點或查詢再試一次。"
    except Exception:
        logger.exception("Gemini error.")
        return "查詢失敗，請再試一次。"

def generate_gemini_content(prompt: str, config, user_query: str):
    grounding_kind = get_grounding_kind(user_query)
    models = get_candidate_models(grounding_kind)

    last_error = None
    for model in models:
        try:
            response = gemini_client.models.generate_content(
                model=model,
                contents=prompt,
                config=config
            )
            if grounding_kind and config.tools and not has_grounding_chunks(response, grounding_kind):
                logger.warning("Gemini response missing grounding chunks. model=%s kind=%s query=%s", model, grounding_kind, user_query)
                continue
            return response
        except Exception as e:
            last_error = e
            logger.warning("Gemini model failed. model=%s query=%s error=%s", model, user_query, e)

    if grounding_kind and config.tools:
        raise GroundingUnavailableError(f"No verified {grounding_kind} grounding result.")

    raise last_error

def get_candidate_models(grounding_kind: str | None) -> list[str]:
    if grounding_kind == "maps":
        return unique_models([GEMINI_MAPS_MODEL, GEMINI_MAPS_FALLBACK_MODEL])
    return unique_models([GEMINI_MODEL, GEMINI_FALLBACK_MODEL])

def unique_models(models: list[str]) -> list[str]:
    unique = []
    for model in models:
        if model and model not in unique:
            unique.append(model)
    return unique

def has_grounding_chunks(response, grounding_kind: str) -> bool:
    try:
        chunks = response.candidates[0].grounding_metadata.grounding_chunks
    except (AttributeError, IndexError, TypeError):
        return False

    if not chunks:
        return False

    for chunk in chunks:
        if grounding_kind == "maps" and getattr(chunk, "maps", None):
            return True
        if grounding_kind == "search" and getattr(chunk, "web", None):
            return True

    return False

def build_gemini_config(user_query: str, location: dict | None = None):
    grounding_kind = get_grounding_kind(user_query)
    tools = []
    tool_config = None

    if grounding_kind == "search":
        tools.append(build_google_search_tool())
    elif grounding_kind == "maps":
        tools.append(build_google_search_tool())
        if location:
            tool_config = types.ToolConfig(
                retrieval_config=types.RetrievalConfig(
                    lat_lng=types.LatLng(
                        latitude=location["latitude"],
                        longitude=location["longitude"]
                    )
                )
            )

    return types.GenerateContentConfig(
        system_instruction=FOODIE_SYSTEM_INSTRUCTION,
        temperature=0.0,  # 降低隨機性，讓推薦更穩定並減少虛構
        max_output_tokens=6000,
        tools=tools or None,
        tool_config=tool_config
    )

def build_google_search_tool():
    return types.Tool(
        google_search=types.GoogleSearch()
    )

def build_plain_gemini_config():
    return types.GenerateContentConfig(
        system_instruction=FOODIE_SYSTEM_INSTRUCTION,
        temperature=0.0,
        max_output_tokens=6000
    )

def format_gemini_response(response, user_query: str) -> str:
    text = response.text or "本美食家突然失語，換個問法再來。"
    grounding_kind = get_grounding_kind(user_query)
    if grounding_kind == "search":
        text = remove_google_maps_lines(text)
    elif grounding_kind == "maps":
        text = replace_google_maps_lines(text, build_grounded_maps_urls(response))

    sources = collect_grounding_sources(response, grounding_kind)
    if sources:
        text = f"{text}\n\n資料來源：\n" + "\n".join(sources)
    return text

def remove_google_maps_lines(text: str) -> str:
    lines = [
        line for line in text.splitlines()
        if "【Google 地圖】" not in line and "【Google Maps】" not in line
    ]
    return "\n".join(lines).strip()

def replace_google_maps_lines(text: str, maps_urls: list[str]) -> str:
    if not maps_urls:
        return remove_google_maps_lines(text)

    url_index = 0
    lines = []
    for line in text.splitlines():
        if "【Google 地圖】" not in line and "【Google Maps】" not in line:
            lines.append(line)
            continue

        maps_url = maps_urls[min(url_index, len(maps_urls) - 1)]
        prefix = line[:len(line) - len(line.lstrip())]
        lines.append(f"{prefix}【Google 地圖】：{maps_url}")
        url_index += 1

    return "\n".join(lines).strip()

def build_grounded_maps_urls(response) -> list[str]:
    try:
        grounding = response.candidates[0].grounding_metadata
    except (AttributeError, IndexError, TypeError):
        return []

    if not grounding or not grounding.grounding_chunks:
        return []

    maps_urls = []
    seen = set()
    for chunk in grounding.grounding_chunks:
        maps_data = getattr(chunk, "maps", None)
        if not maps_data:
            continue

        maps_url = build_grounded_maps_url(maps_data)
        if not maps_url or maps_url in seen:
            continue

        seen.add(maps_url)
        maps_urls.append(maps_url)

    return maps_urls

def build_grounded_maps_url(maps_data) -> str | None:
    title = get_maps_title(maps_data)
    official_url = get_official_maps_url(maps_data)
    place_id = get_maps_place_id(maps_data)
    if not place_id and official_url:
        place_id = resolve_place_id_by_maps_url(official_url)
    if not place_id and title:
        place_id = resolve_place_id_by_name(title)
    if place_id and title:
        return (
            "https://www.google.com/maps/search/?api=1"
            f"&query={quote_plus(title)}"
            f"&query_place_id={quote_plus(place_id)}"
        )

    return official_url

def get_official_maps_url(maps_data) -> str | None:
    for attr_name in ("google_maps_uri", "googleMapsUri", "place_url", "placeUrl"):
        official_url = getattr(maps_data, attr_name, None)
        if official_url:
            return official_url

    maps_links = (
        getattr(maps_data, "google_maps_links", None)
        or getattr(maps_data, "googleMapsLinks", None)
    )
    if isinstance(maps_links, dict):
        return maps_links.get("placeUrl") or maps_links.get("place_url")
    if maps_links:
        return (
            getattr(maps_links, "place_url", None)
            or getattr(maps_links, "placeUrl", None)
        )

    return getattr(maps_data, "uri", None)

def get_maps_title(maps_data) -> str | None:
    return getattr(maps_data, "title", None) or getattr(maps_data, "text", None)

def get_maps_place_id(maps_data) -> str | None:
    place_id = (
        getattr(maps_data, "place_id", None)
        or getattr(maps_data, "placeId", None)
        or getattr(maps_data, "id", None)
    )
    if not place_id:
        return None

    return normalize_place_id(str(place_id))

def normalize_place_id(place_id: str) -> str:
    return place_id.removeprefix("places/")

def resolve_place_id_by_name(place_name: str) -> str | None:
    if not MAPS_TOOLS_API_KEY:
        return None

    normalized_name = place_name.strip().lower()
    if not normalized_name:
        return None
    cache_key = f"name:{normalized_name}"

    with resolution_cache_lock:
        if cache_key in resolved_place_id_cache:
            return resolved_place_id_cache[cache_key]

    try:
        response_data = call_resolve_names_api(place_name)
        result = (response_data.get("results") or [{}])[0]
        entity = result.get("entity") or {}
        place_id = entity.get("place")
        if not place_id:
            failed_requests = response_data.get("failedRequests") or {}
            if failed_requests:
                logger.info("Maps ResolveNames failed. place=%s detail=%s", place_name, failed_requests.get("0"))
            return None

        normalized_place_id = normalize_place_id(place_id)
        with resolution_cache_lock:
            resolved_place_id_cache[cache_key] = normalized_place_id
        return normalized_place_id
    except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError):
        logger.exception("Maps ResolveNames request failed. place=%s", place_name)
        return None

def resolve_place_id_by_maps_url(maps_url: str) -> str | None:
    if not MAPS_TOOLS_API_KEY:
        return None

    normalized_url = maps_url.strip()
    if not normalized_url:
        return None
    cache_key = f"url:{normalized_url}"

    with resolution_cache_lock:
        if cache_key in resolved_place_id_cache:
            return resolved_place_id_cache[cache_key]

    try:
        response_data = call_resolve_maps_urls_api(maps_url)
        entity = (response_data.get("entities") or [{}])[0]
        place_id = entity.get("place")
        if not place_id:
            failed_requests = response_data.get("failedRequests") or {}
            if failed_requests:
                logger.info("Maps ResolveMapsUrls failed. url=%s detail=%s", maps_url, failed_requests.get("0"))
            return None

        normalized_place_id = normalize_place_id(place_id)
        with resolution_cache_lock:
            resolved_place_id_cache[cache_key] = normalized_place_id
        return normalized_place_id
    except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError):
        logger.exception("Maps ResolveMapsUrls request failed. url=%s", maps_url)
        return None

def call_resolve_names_api(place_name: str) -> dict:
    endpoint = "https://mapstools.googleapis.com/v1alpha:resolveNames"
    url = f"{endpoint}?{urlencode({'key': MAPS_TOOLS_API_KEY})}"
    payload = json.dumps({"queries": [{"text": place_name}]}).encode("utf-8")
    request = UrlRequest(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urlopen(request, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))

def call_resolve_maps_urls_api(maps_url: str) -> dict:
    endpoint = "https://mapstools.googleapis.com/v1alpha:resolveMapsUrls"
    url = f"{endpoint}?{urlencode({'key': MAPS_TOOLS_API_KEY})}"
    payload = json.dumps({"urls": [maps_url]}).encode("utf-8")
    request = UrlRequest(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urlopen(request, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))

def collect_grounding_sources(response, grounding_kind: str | None = None) -> list[str]:
    try:
        grounding = response.candidates[0].grounding_metadata
    except (AttributeError, IndexError, TypeError):
        return []

    if not grounding or not grounding.grounding_chunks:
        return []

    sources = []
    seen = set()
    for chunk in grounding.grounding_chunks:
        source = None
        if getattr(chunk, "maps", None):
            if grounding_kind == "search":
                continue
            source = ("Google Maps", chunk.maps.title, build_grounded_maps_url(chunk.maps))
        elif getattr(chunk, "web", None):
            source = ("Web", chunk.web.title, chunk.web.uri)

        if not source:
            continue

        label, title, uri = source
        if not title or not uri or uri in seen:
            continue

        seen.add(uri)
        sources.append(f"- {label}：{title} {uri}")
        if len(sources) >= 5:
            break

    return sources

TRUNCATED_MESSAGE_SUFFIX = "\n...（內容過長，後續已省略）"

def split_line_text(text: str, max_length: int = 4500, max_messages: int = 5) -> list[str]:
    if len(text) <= max_length:
        return [text]

    units = split_message_units(text)
    return pack_message_units(units, max_length, max_messages)

def split_message_units(text: str) -> list[str]:
    normalized_text = text.replace("\r\n", "\n").strip()
    if not normalized_text:
        return []

    paragraph_units = [
        unit.strip()
        for unit in re.split(r"\n\s*\n", normalized_text)
        if unit.strip()
    ]
    if len(paragraph_units) > 1:
        units = []
        for paragraph_unit in paragraph_units:
            units.extend(split_lines_into_message_units(paragraph_unit.splitlines()))
        return units

    return split_lines_into_message_units(normalized_text.splitlines())

def split_lines_into_message_units(lines: list[str]) -> list[str]:
    units = []
    current_lines = []
    for index, line in enumerate(lines):
        next_line = lines[index + 1] if index + 1 < len(lines) else ""
        if is_message_unit_start(line, next_line) and current_lines:
            units.append("\n".join(current_lines).strip())
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines:
        units.append("\n".join(current_lines).strip())

    return units

def is_message_unit_start(line: str, next_line: str = "") -> bool:
    return (
        is_item_start_line(line)
        or is_plain_item_title_line(line, next_line)
        or is_source_section_start(line)
    )

def is_item_start_line(line: str) -> bool:
    stripped_line = line.strip()
    return bool(
        re.match(r"^[-*•]\s+", stripped_line)
        or re.match(r"^\d+[.、]\s+", stripped_line)
    )

def is_plain_item_title_line(line: str, next_line: str) -> bool:
    stripped_line = line.strip()
    if not stripped_line or stripped_line.startswith("【"):
        return False
    return next_line.strip().startswith("【評語】")

def is_source_section_start(line: str) -> bool:
    return line.strip() in ("資料來源：", "資料來源:")

def pack_message_units(units: list[str], max_length: int, max_messages: int) -> list[str]:
    messages = []
    unit_index = 0
    while unit_index < len(units) and len(messages) < max_messages:
        is_last_message = len(messages) == max_messages - 1
        message = ""
        limit = max_length - len(TRUNCATED_MESSAGE_SUFFIX) if is_last_message else max_length

        while unit_index < len(units):
            unit = units[unit_index]
            separator = "\n\n" if message else ""
            candidate = f"{message}{separator}{unit}"
            if len(candidate) <= limit:
                message = candidate
                unit_index += 1
                continue

            if not message:
                message = split_oversized_unit(unit, limit)
                unit_index += 1
            break

        if not message:
            break

        if is_last_message and unit_index < len(units):
            message = f"{message.rstrip()}{TRUNCATED_MESSAGE_SUFFIX}"
        messages.append(message.rstrip())

    return messages

def split_oversized_unit(unit: str, max_length: int) -> str:
    if len(unit) <= max_length:
        return unit

    split_at = unit.rfind("\n", 0, max_length)
    if split_at < max_length * 0.5:
        split_at = max_length
    return unit[:split_at].rstrip()

def build_text_messages(text: str) -> list[TextMessage]:
    return [TextMessage(text=chunk) for chunk in split_line_text(text)]

def send_line_reply(reply_token: str, text: str):
    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        line_bot_api.reply_message(
            ReplyMessageRequest(
                reply_token=reply_token,
                messages=build_text_messages(text)
            )
        )

def send_line_reply_or_push(reply_token: str, text: str, target_id: str | None = None):
    try:
        send_line_reply(reply_token, text)
    except ApiException as e:
        if e.status == 400:
            logger.warning("LINE reply token may be expired or invalid. Falling back to push.")
        else:
            logger.exception("LINE API error while sending reply. status=%s", e.status)
        if not target_id:
            return
        send_line_push(target_id, text)
    except Exception:
        logger.exception("Unexpected error sending LINE reply.")
        if not target_id:
            return
        send_line_push(target_id, text)

def send_line_push(target_id: str, text: str):
    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        line_bot_api.push_message(
            PushMessageRequest(
                to=target_id,
                messages=build_text_messages(text)
            )
        )
