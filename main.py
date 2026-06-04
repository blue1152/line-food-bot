import os
import base64
import hashlib
import hmac
import json
import logging
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError
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
    TextMessage,
)
from linebot.v3.webhooks import (
    MessageEvent,
    TextMessageContent,
    LocationMessageContent,
    GroupSource,
    RoomSource,
    UserSource,
)
from google import genai
from google.genai import types

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI()

# --- 1. 初始化與環境變數 ---
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
GEMINI_FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-3.5-flash")
LINE_BOT_USER_ID = os.getenv("LINE_BOT_USER_ID")

missing_envs = [
    name
    for name, value in {
        "LINE_CHANNEL_SECRET": LINE_CHANNEL_SECRET,
        "LINE_CHANNEL_ACCESS_TOKEN": LINE_CHANNEL_ACCESS_TOKEN,
        "GEMINI_API_KEY": GEMINI_API_KEY,
        "GEMINI_MODEL": GEMINI_MODEL,
        "GEMINI_FALLBACK_MODEL": GEMINI_FALLBACK_MODEL,
    }.items()
    if not value
]
if missing_envs:
    raise RuntimeError(
        f"Missing required environment variables: {', '.join(missing_envs)}"
    )

configuration = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)
gemini_client = genai.Client(
    api_key=GEMINI_API_KEY,
    http_options=types.HttpOptions(
        timeout=45000, retry_options=types.HttpRetryOptions(attempts=1)
    ),
)
gemini_executor = ThreadPoolExecutor(max_workers=4)

# --- 2. 記憶庫：暫存最後發送的位置資訊 ---
group_location_cache = {}
user_location_cache = {}
processed_event_cache = {}
event_cache_lock = threading.Lock()
bot_user_id_lock = threading.Lock()
LOCATION_CACHE_TTL_SECONDS = 2 * 60 * 60
PROCESSED_EVENT_TTL_SECONDS = 10 * 60
LOCATION_REQUIRED_MESSAGE = (
    "請告訴我地點資訊（例如城市、區域、店名、景點或地址），或先用 LINE 傳送「位置資訊」。\n"
    "沒有地點的話，我沒辦法精準回答。"
)
AI_REPLY_TIMEOUT_SECONDS = 5
AI_TOTAL_TIMEOUT_SECONDS = 45
RENDER_SLEEP_NOTICE_MESSAGE = "休眠中, 一分鐘後起床"
RENDER_WAKE_RETRY_DELAY_SECONDS = 60
RENDER_IDLE_SECONDS = int(os.getenv("RENDER_IDLE_SECONDS", "900"))
last_webhook_received_at = None
render_wake_event_ids = set()
render_wake_notified_event_ids = set()
render_wake_retry_lock = threading.Lock()

LOCATION_CONTEXT_KEYWORDS = (
    "附近",
    "周邊",
    "這裡",
    "這附近",
    "周圍",
    "nearby",
    "near me",
    "around me",
)
RELATIVE_LOCATION_VALUES = LOCATION_CONTEXT_KEYWORDS + (
    "here",
    "current location",
    "my location",
)
PLACE_QUERY_KEYWORDS = (
    "餐廳",
    "美食",
    "咖啡",
    "咖啡廳",
    "小吃",
    "宵夜",
    "早餐",
    "午餐",
    "晚餐",
    "甜點",
    "酒吧",
    "景點",
    "旅遊",
    "行程",
    "住宿",
    "飯店",
    "旅館",
    "夜市",
    "伴手禮",
    "restaurant",
    "food",
    "cafe",
    "coffee",
    "bar",
    "hotel",
    "attraction",
)
FOOD_QUERY_KEYWORDS = (
    "牛肉麵",
    "拉麵",
    "火鍋",
    "燒肉",
    "壽司",
    "咖哩",
    "便當",
    "滷肉飯",
    "雞排",
    "甜點",
    "下午茶",
    "早餐",
    "午餐",
    "晚餐",
    "宵夜",
    "飲料",
    "咖啡",
    "小吃",
    "noodle",
    "ramen",
    "hotpot",
    "sushi",
    "dessert",
    "breakfast",
    "lunch",
    "dinner",
    "bbq",
    "barbecue",
    "pizza",
    "burger",
    "taco",
    "brunch",
    "bakery",
    "bistro",
)
MAPS_KEYWORDS = LOCATION_CONTEXT_KEYWORDS + PLACE_QUERY_KEYWORDS + FOOD_QUERY_KEYWORDS
SEARCH_KEYWORDS = (
    "最新",
    "活動",
    "展覽",
    "演唱會",
    "市集",
    "施工",
    "營業異動",
    "臨時休息",
    "新聞",
    "票價",
    "時間表",
    "latest",
    "news",
    "event",
    "exhibition",
    "concert",
    "market",
    "construction",
    "schedule",
)
RECIPE_KEYWORDS = (
    "食譜",
    "做法",
    "怎麼做",
    "料理",
    "煮法",
    "recipe",
    "cook",
    "cooking",
    "how to make",
)
BOT_KEYWORDS = ("@美食家", "吃什麼")
GENERIC_PLACE_QUERY_WORDS = (
    "推薦",
    "找",
    "查",
    "有什麼",
    "哪裡",
    "哪家",
    "好吃",
    "好喝",
    "適合",
    "附近",
    "幫我",
    "請問",
    "想吃",
    "想喝",
    "一下",
    "的",
    "和",
    "跟",
    "recommend",
    "find",
    "search",
    "good",
    "best",
    "nearby",
    "near me",
)

# --- 3. 系統角色設定（由模型依真實店名生成標準 Google 地圖搜尋連結） ---
FOODIE_SYSTEM_INSTRUCTION = (
    "你是一位兼具旅遊專家、城市探險家與頂級美食家的嚮導；你熟悉景點、活動、路線、在地文化與臨場探索，但美食判斷是你的最強項，說話風趣刁鑽。\n\n"
    "【重要限制】：\n"
    "1. 你必須搭配內建的 Google Search 聯網工具（Grounding）來驗證店家、景點、活動與時效資訊，絕對不允許虛構。\n"
    "2. 永久歇業、無法查證、或 grounding 資料不足的店家一律不准推薦。\n"
    "3. 前言必須極其簡短（嚴格限制在 30 字以內），直接切入主題。\n"
    "4. 在 grounding 資料充足時，請盡量多給結果；若使用者或系統提示指定 8 到 12 個熱門項目，請以 8 到 12 個為準。\n\n"
    "【回覆格式約束】：\n"
    "1. 使用點排列（Bullet points）呈現推薦項目。\n"
    "2. 每個推薦項目之間必須空一行，方便系統分段傳送；同一個項目的名稱、碎碎念、地圖連結和指數必須放在同一段。\n"
    "3. 每個推薦項目的結構必須嚴格遵守：\n"
    "   * 店名/景點名/活動名\n"
    "   * 【評語】：只能用一句話（20字內）精闢點出特色，語氣要毒舌風趣但不要侮辱。\n"
    "   * 【Google 地圖】：請依據你聯網查到的真實名稱與區域，將其 URL 編碼並嚴格以下列格式自行生成標準地圖搜尋連結：https://www.google.com/maps/search/?api=1&query=店名+區域（例如：https://www.google.com/maps/search/?api=1&query=綠河+新店）。\n"
    "   * 【推薦指數】：滿分5顆星（如：★★★★☆）。"
)

LOCATION_DETECTION_SYSTEM_INSTRUCTION = (
    "你是地點資訊抽取器。只判斷使用者文字中是否包含可判定為地點的資訊。\n"
    "地點可以是城市、國家、區域、街道、地址、店名、餐廳名、景點、地標、場館、Google Maps URL 或經緯度。\n"
    "食物品項、料理類型、活動類型、抽象需求不算地點，例如：拉麵、咖啡、展覽、市集、推薦美食。\n"
    "如果文字只有「附近、這裡、周邊、near me」這類相對位置，且沒有實際地點名稱，has_location 必須是 false。\n"
    '只回傳 JSON，格式必須是：{"has_location": boolean, "locations": [string]}。'
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
    if not is_valid_line_signature(body, signature):
        raise HTTPException(status_code=400, detail="Invalid Signature")

    body_str = body.decode("utf-8")
    events = parse_line_webhook_events(body_str)
    log_line_webhook_delivery(events)
    wake_notice_events = mark_render_wake_events(events)
    for event_id, target_id in wake_notice_events:
        try:
            send_line_push(target_id, RENDER_SLEEP_NOTICE_MESSAGE)
            mark_render_wake_notice_sent(event_id)
            logger.info(
                "Sent immediate Render wake notice. webhook_event_id=%s target_id=%s",
                event_id,
                target_id,
            )
        except Exception:
            logger.exception(
                "Failed to send immediate Render wake notice. webhook_event_id=%s target_id=%s",
                event_id,
                target_id,
            )

    background_tasks.add_task(handle_line_request, body_str, signature)
    return "OK"


def handle_line_request(body: str, signature: str):
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        logger.warning("Invalid LINE signature.")
    except Exception:
        logger.exception("Unhandled error while handling LINE webhook.")


def is_valid_line_signature(body: bytes, signature: str) -> bool:
    digest = hmac.new(
        LINE_CHANNEL_SECRET.encode("utf-8"), body, hashlib.sha256
    ).digest()
    expected_signature = base64.b64encode(digest).decode("utf-8")
    return hmac.compare_digest(expected_signature, signature)


def parse_line_webhook_events(body: str) -> list[dict]:
    try:
        events = json.loads(body).get("events", [])
        return events if isinstance(events, list) else []
    except Exception:
        logger.exception("Failed to parse LINE webhook body.")
        return []


def log_line_webhook_delivery(events: list[dict]):
    redelivery_count = sum(1 for event in events if is_redelivered_event(event))
    logger.info(
        "Received LINE webhook events. event_count=%s redelivery_count=%s",
        len(events),
        redelivery_count,
    )

    for event in events:
        logger.info(
            "LINE webhook event delivery. webhook_event_id=%s event_type=%s message_type=%s source_type=%s is_redelivery=%s",
            get_raw_webhook_event_id(event),
            event.get("type"),
            (event.get("message") or {}).get("type"),
            (event.get("source") or {}).get("type"),
            is_redelivered_event(event),
        )


def get_raw_webhook_event_id(event: dict) -> str | None:
    return event.get("webhookEventId") or event.get("webhook_event_id")


def is_redelivered_event(event: dict) -> bool:
    delivery_context = event.get("deliveryContext") or event.get("delivery_context")
    if not isinstance(delivery_context, dict):
        return False
    return bool(
        delivery_context.get("isRedelivery")
        or delivery_context.get("is_redelivery")
    )


def mark_render_wake_events(events: list[dict]) -> list[tuple[str, str]]:
    global last_webhook_received_at

    now = time.time()
    with render_wake_retry_lock:
        previous_webhook_received_at = last_webhook_received_at
        idle_seconds = (
            None
            if previous_webhook_received_at is None
            else now - previous_webhook_received_at
        )
        is_wake_request = (
            previous_webhook_received_at is None or idle_seconds >= RENDER_IDLE_SECONDS
        )
        last_webhook_received_at = now

        if not is_wake_request:
            return []

        wake_notice_events = []
        for event in events:
            event_id = get_raw_webhook_event_id(event)
            if event_id:
                render_wake_event_ids.add(event_id)
                target_id = get_render_wake_notice_target(event)
                if target_id:
                    wake_notice_events.append((event_id, target_id))

        logger.info(
            "Marked Render wake webhook events. event_count=%s notice_count=%s idle_seconds=%s threshold_seconds=%s",
            len(events),
            len(wake_notice_events),
            "unknown" if idle_seconds is None else round(idle_seconds, 3),
            RENDER_IDLE_SECONDS,
        )
        return wake_notice_events


def mark_render_wake_notice_sent(event_id: str):
    with render_wake_retry_lock:
        render_wake_notified_event_ids.add(event_id)


def get_render_wake_notice_target(event: dict) -> str | None:
    message = event.get("message") or {}
    if message.get("type") != "text":
        return None

    source = event.get("source") or {}
    source_type = source.get("type")
    if source_type == "user":
        return source.get("userId") or source.get("user_id")

    if source_type == "group":
        if not raw_group_text_triggers_bot(message):
            return None
        return source.get("groupId") or source.get("group_id")

    if source_type == "room":
        if not raw_group_text_triggers_bot(message):
            return None
        return source.get("roomId") or source.get("room_id")

    return None


def raw_group_text_triggers_bot(message: dict) -> bool:
    text = message.get("text") or ""
    if any(keyword in text for keyword in BOT_KEYWORDS):
        return True

    if not LINE_BOT_USER_ID:
        return False

    mention = message.get("mention") or {}
    mentionees = mention.get("mentionees") or []
    return any(
        mentionee.get("userId") == LINE_BOT_USER_ID
        or mentionee.get("user_id") == LINE_BOT_USER_ID
        for mentionee in mentionees
    )


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
        event_id = event_data.get("webhookEventId") or event_data.get(
            "webhook_event_id"
        )
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
        "expires_at": time.time() + LOCATION_CACHE_TTL_SECONDS,
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
        '從以下文字提取地點，回傳 JSON {"has_location": boolean, "locations": string[]}。\n'
        "只把可作為 Google Maps / Google Search 地理查詢依據的資訊放進 locations。\n\n"
        f"文字：{query}"
    )
    config = types.GenerateContentConfig(
        system_instruction=LOCATION_DETECTION_SYSTEM_INSTRUCTION,
        temperature=0.0,
        max_output_tokens=300,
        response_mime_type="application/json",
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
                model=model, contents=prompt, config=config
            )
        except Exception as e:
            last_error = e
            logger.warning(
                "Location detection model failed. model=%s query=%s error=%s",
                model,
                user_query,
                e,
            )

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
            text = text[json_start : json_end + 1]

    data = json.loads(text)
    if not isinstance(data, dict):
        return {"has_location": False, "locations": []}
    locations = data.get("locations") or []
    if not isinstance(locations, list):
        locations = []

    clean_locations = [
        str(location).strip()
        for location in locations
        if str(location).strip()
        and str(location).strip().lower() not in RELATIVE_LOCATION_VALUES
    ]
    return {
        "has_location": bool(data.get("has_location")) and bool(clean_locations),
        "locations": clean_locations,
    }


def has_detected_location(location_detection: dict | None) -> bool:
    return bool(
        location_detection
        and location_detection.get("has_location")
        and location_detection.get("locations")
    )


def build_prompt_with_detected_locations(
    user_query: str, location_detection: dict
) -> str:
    locations = "、".join(location_detection.get("locations", []))
    return (
        "請使用 Google Search 聯網工具，精準抓出符合使用者需求的真實店家、景點或活動。\n"
        f"【使用者查詢】：『{user_query}』\n"
        f"【NLP 偵測到的地點資訊】：{locations}\n\n"
        "【檢索引導指令】：\n"
        "1. 請以偵測到的地點資訊作為搜尋範圍，不要把範圍擴大到其他城市或行政區。\n"
        "2. 如果查詢是「縣市/區域 + 概括品項」（例如咖啡廳、甜點、餐廳、市集、展覽），請直接挑選該區域目前最知名的 8 到 12 個熱門且可查證項目。\n"
        "3. 嚴格禁止因為結果太多而拒絕回答；請直接篩選代表性最高、資料最可信的結果。\n"
        "4. 若沒有足夠可查證資料，寧可少給，但至少嘗試提供已查證的代表性結果。"
    )


def build_prompt_with_cached_location(
    user_query: str, location: dict, chat_context: str
) -> str:
    return (
        f"使用者在{chat_context}詢問：『{user_query}』\n"
        "請使用 Google Search 聯網工具，以以下精準經緯度座標為中心，"
        "搜尋周邊半徑 1 公里內符合使用者需求的真實店家、景點或活動。\n"
        f"【使用者需求】：{user_query}\n"
        f"【中心點名稱】：{location['title']}\n"
        f"【中心點地址】：{location['address']}\n"
        f"【精準座標】：緯度 {location['latitude']}, 經度 {location['longitude']}\n\n"
        "【檢索引導指令】：\n"
        "1. 請直接挑選該座標周邊目前最知名的 8 到 12 個熱門且可查證項目。\n"
        "2. 嚴格禁止因為結果太多而拒絕回答；請直接篩選代表性最高、距離與資料可信度最好的結果。\n"
        "3. 請務必以 Google Search grounding 查到的資料為準，優先排除明顯歇業、永久停業或資料不足的結果。"
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
        mentionee
        for mentionee in mentionees
        if mentionee.get("userId") == bot_user_id
        or mentionee.get("user_id") == bot_user_id
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


def extract_group_prompt(
    user_message: str, event: MessageEvent | None = None
) -> tuple[bool, str]:
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
def consume_render_wake_delay(event: MessageEvent) -> tuple[bool, bool]:
    event_id = get_webhook_event_id(event)
    if not event_id:
        return False, False

    with render_wake_retry_lock:
        if event_id not in render_wake_event_ids:
            return False, False
        notice_sent = event_id in render_wake_notified_event_ids
        render_wake_event_ids.discard(event_id)
        render_wake_notified_event_ids.discard(event_id)
        logger.info(
            "Consumed Render wake delay. webhook_event_id=%s notice_sent=%s",
            event_id,
            notice_sent,
        )
        return True, notice_sent


def should_delay_for_render_wake(event: MessageEvent) -> bool:
    should_delay, _notice_sent = consume_render_wake_delay(event)
    return should_delay


def retry_text_message_after_render_wake(event: MessageEvent):
    try:
        logger.info("Retrying text message after Render wake delay.")
        handle_text_message(event, skip_duplicate_check=False)
    except Exception:
        logger.exception("Failed to retry text message after Render wake delay.")


def schedule_render_wake_retry(event: MessageEvent):
    event_id = get_webhook_event_id(event)
    timer = threading.Timer(
        RENDER_WAKE_RETRY_DELAY_SECONDS,
        retry_text_message_after_render_wake,
        args=(event,),
    )
    timer.daemon = True
    timer.start()
    logger.info(
        "Scheduled Render wake retry. webhook_event_id=%s delay_seconds=%s",
        event_id,
        RENDER_WAKE_RETRY_DELAY_SECONDS,
    )


@handler.add(MessageEvent, message=LocationMessageContent)
def handle_location_message(event: MessageEvent):
    if should_skip_event(event):
        return

    is_in_group = isinstance(event.source, (GroupSource, RoomSource))
    location = {
        "title": event.message.title if event.message.title else "未知起點",
        "address": event.message.address,
        "latitude": event.message.latitude,
        "longitude": event.message.longitude,
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
def handle_text_message(event: MessageEvent, skip_duplicate_check: bool = True):
    if skip_duplicate_check and should_skip_event(event):
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

        should_delay, notice_sent = consume_render_wake_delay(event)
        if should_delay:
            if not notice_sent:
                send_line_reply_or_push(
                    reply_token, RENDER_SLEEP_NOTICE_MESSAGE, target_id
                )
            schedule_render_wake_retry(event)
            return

        group_id = get_conversation_id(event.source)
        cached_location = get_cached_location(group_location_cache, group_id)
        location_detection = detect_locations_in_text(clean_prompt)
        if (
            cached_location
            and has_detected_location(location_detection)
            and not needs_location(clean_prompt)
        ):
            clear_cached_location(group_location_cache, group_id)
            cached_location = None
            logger.info(
                "Cleared group location because query has explicit place. group_id=%s",
                group_id,
            )

        if cached_location:
            final_prompt = build_prompt_with_cached_location(
                clean_prompt, cached_location, "群組中標記了你，並"
            )
        else:
            if not has_detected_location(location_detection):
                send_line_reply_or_push(
                    reply_token, LOCATION_REQUIRED_MESSAGE, target_id
                )
                return
            final_prompt = build_prompt_with_detected_locations(
                clean_prompt, location_detection
            )
    else:
        should_delay, notice_sent = consume_render_wake_delay(event)
        if should_delay:
            if not notice_sent:
                send_line_reply_or_push(
                    reply_token, RENDER_SLEEP_NOTICE_MESSAGE, target_id
                )
            schedule_render_wake_retry(event)
            return

        user_id = event.source.user_id if isinstance(event.source, UserSource) else None
        cached_location = (
            get_cached_location(user_location_cache, user_id) if user_id else None
        )
        location_detection = detect_locations_in_text(user_message)
        if (
            cached_location
            and has_detected_location(location_detection)
            and not needs_location(user_message)
        ):
            clear_cached_location(user_location_cache, user_id)
            cached_location = None
            logger.info(
                "Cleared user location because query has explicit place. user_id=%s",
                user_id,
            )

        if cached_location:
            final_prompt = build_prompt_with_cached_location(
                user_message, cached_location, "一對一聊天室"
            )
        else:
            if not has_detected_location(location_detection):
                send_line_reply_or_push(
                    reply_token, LOCATION_REQUIRED_MESSAGE, target_id
                )
                return
            final_prompt = build_prompt_with_detected_locations(
                user_message, location_detection
            )

    user_query = clean_prompt if is_in_group else user_message
    ai_response = ask_gemini_with_optional_wait(
        final_prompt, user_query, cached_location, reply_token, target_id
    )
    if ai_response:
        send_line_reply_or_push(reply_token, ai_response, target_id)


# --- 7. Gemini API 呼叫邏輯 ---
def ask_gemini_with_optional_wait(
    prompt: str,
    user_query: str,
    location: dict | None,
    reply_token: str,
    target_id: str | None,
) -> str | None:
    if not target_id or not uses_grounding(user_query):
        return ask_gemini_foodie(prompt, user_query, location)

    future = gemini_executor.submit(ask_gemini_foodie, prompt, user_query, location)
    try:
        return future.result(timeout=AI_REPLY_TIMEOUT_SECONDS)
    except TimeoutError:
        send_line_reply_or_push(reply_token, f"{user_query}, 還在查詢中...", target_id)
        try:
            ai_response = future.result(
                timeout=AI_TOTAL_TIMEOUT_SECONDS - AI_REPLY_TIMEOUT_SECONDS
            )
            send_line_push(target_id, ai_response)
        except TimeoutError:
            logger.warning("Gemini query timed out. query=%s", user_query)
            send_line_push(target_id, "查詢失敗，請再試一次。")
        except Exception:
            logger.exception("Gemini async query failed.")
            send_line_push(target_id, "查詢失敗，請再試一次。")
        return None


def ask_gemini_foodie(
    prompt: str, user_query: str, location: dict | None = None
) -> str:
    try:
        logger.info("Sending Gemini Grounding request. query=%s", user_query)
        config = build_gemini_config(user_query, location)
        response = generate_gemini_content(prompt, config, user_query)
        return response.text or "本美食家連網查了老半天突然失語，換個方式問問看？"
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
                model=model, contents=prompt, config=config
            )
            if (
                grounding_kind
                and config.tools
                and not has_grounding_chunks(response, grounding_kind)
            ):
                logger.info(
                    "Gemini response has no explicit grounding chunks; using model text. model=%s kind=%s query=%s chunk_types=%s",
                    model,
                    grounding_kind,
                    user_query,
                    get_grounding_chunk_types(response),
                )
            return response
        except Exception as e:
            last_error = e
            logger.warning(
                "Gemini model failed. model=%s query=%s error=%s", model, user_query, e
            )

    raise last_error


def get_candidate_models(grounding_kind: str | None) -> list[str]:
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
        if getattr(chunk, "web", None) or getattr(chunk, "maps", None):
            return True

    return False


def get_grounding_chunk_types(response) -> list[str]:
    try:
        chunks = response.candidates[0].grounding_metadata.grounding_chunks
    except (AttributeError, IndexError, TypeError):
        return []

    chunk_types = []
    for chunk in chunks or []:
        for chunk_type in ("web", "maps", "retrieved_context"):
            if getattr(chunk, chunk_type, None):
                chunk_types.append(chunk_type)
    return chunk_types


def build_gemini_config(user_query: str, location: dict | None = None):
    grounding_kind = get_grounding_kind(user_query)
    tools = []

    if grounding_kind == "search":
        tools.append(build_google_search_tool())
    elif grounding_kind == "maps":
        tools.append(build_google_search_tool())

    return types.GenerateContentConfig(
        system_instruction=FOODIE_SYSTEM_INSTRUCTION,
        temperature=0.0,  # 降低隨機性，讓推薦更穩定並減少虛構
        max_output_tokens=6000,
        safety_settings=build_safety_settings(),
        tools=tools or None,
    )


def build_google_search_tool():
    return types.Tool(google_search=types.GoogleSearch())


def build_plain_gemini_config():
    return types.GenerateContentConfig(
        system_instruction=FOODIE_SYSTEM_INSTRUCTION,
        temperature=0.0,
        max_output_tokens=6000,
        safety_settings=build_safety_settings(),
    )


def build_safety_settings():
    return [
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            threshold=types.HarmBlockThreshold.BLOCK_NONE,
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
            threshold=types.HarmBlockThreshold.BLOCK_NONE,
        ),
    ]


TRUNCATED_MESSAGE_SUFFIX = "\n...（內容過長，後續已省略）"


def split_line_text(
    text: str, max_length: int = 4500, max_messages: int = 5
) -> list[str]:
    if len(text) <= max_length:
        return [text]

    units = split_message_units(text)
    return pack_message_units(units, max_length, max_messages)


def split_message_units(text: str) -> list[str]:
    normalized_text = text.replace("\r\n", "\n").strip()
    if not normalized_text:
        return []

    paragraph_units = [
        unit.strip() for unit in re.split(r"\n\s*\n", normalized_text) if unit.strip()
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
    stripped_next_line = next_line.strip()
    return stripped_next_line.startswith("【評語】")


def is_source_section_start(line: str) -> bool:
    return line.strip() in ("資料來源：", "資料來源:")


def pack_message_units(
    units: list[str], max_length: int, max_messages: int
) -> list[str]:
    messages = []
    unit_index = 0
    while unit_index < len(units) and len(messages) < max_messages:
        is_last_message = len(messages) == max_messages - 1
        message = ""
        limit = (
            max_length - len(TRUNCATED_MESSAGE_SUFFIX)
            if is_last_message
            else max_length
        )

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
                reply_token=reply_token, messages=build_text_messages(text)
            )
        )


def send_line_reply_or_push(reply_token: str, text: str, target_id: str | None = None):
    try:
        send_line_reply(reply_token, text)
    except ApiException as e:
        if e.status == 400:
            logger.warning(
                "LINE reply token may be expired or invalid. Falling back to push."
            )
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
            PushMessageRequest(to=target_id, messages=build_text_messages(text))
        )
