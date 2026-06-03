import os
import logging
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

missing_envs = [
    name for name, value in {
        "LINE_CHANNEL_SECRET": LINE_CHANNEL_SECRET,
        "LINE_CHANNEL_ACCESS_TOKEN": LINE_CHANNEL_ACCESS_TOKEN,
        "GEMINI_API_KEY": GEMINI_API_KEY
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

# --- 2. 記憶庫：暫存最後發送的位置資訊 ---
group_location_cache = {}
user_location_cache = {}
processed_event_cache = {}
event_cache_lock = threading.Lock()
LOCATION_CACHE_TTL_SECONDS = 2 * 60 * 60
PROCESSED_EVENT_TTL_SECONDS = 10 * 60
LOCATION_REQUIRED_MESSAGE = (
    "我目前沒有可用定位，可能是服務剛喚醒、重新部署，或尚未收到位置。\n"
    "請先用 LINE 傳送「位置資訊」，再問我附近美食。"
)
AI_REPLY_TIMEOUT_SECONDS = 5
AI_TOTAL_TIMEOUT_SECONDS = 45

LOCATION_CONTEXT_KEYWORDS = ("附近", "周邊", "這裡", "這附近", "周圍", "nearby", "near me", "around me")
PLACE_QUERY_KEYWORDS = (
    "餐廳", "美食", "咖啡", "咖啡廳", "小吃", "宵夜", "早餐", "午餐", "晚餐", "甜點",
    "酒吧", "景點", "旅遊", "行程", "住宿", "飯店", "旅館", "夜市", "伴手禮",
    "restaurant", "food", "cafe", "coffee", "bar", "hotel", "attraction"
)
MAPS_KEYWORDS = LOCATION_CONTEXT_KEYWORDS + PLACE_QUERY_KEYWORDS
SEARCH_KEYWORDS = (
    "最新", "活動", "展覽", "演唱會", "市集", "施工", "營業異動", "臨時休息", "新聞", "票價", "時間表",
    "latest", "news", "event", "exhibition", "concert", "market", "construction", "schedule"
)
RECIPE_KEYWORDS = ("食譜", "做法", "怎麼做", "料理", "煮法", "recipe", "cook", "cooking", "how to make")
BOT_KEYWORDS = ("@美食家", "美食家", "@吃什麼", "吃什麼")

# --- 3. 修正版角色設定（嚴格指定 Google 地圖 URL 格式） ---
FOODIE_SYSTEM_INSTRUCTION = (
    "你是一位精通全球美食、說話風趣刁鑽的頂級美食家。\n\n"
    "【重要時空與真實性限制】：\n"
    "1. 若系統提供 Google Maps 或 Google Search grounding 資料，請優先依據 grounding 資料回答。\n"
    "2. 嚴格禁止虛構、捏造、想像任何不存在的店家、景點或活動名稱。\n"
    "3. 推薦店家或景點時，必須以可從 grounding 資料、使用者提供的位置、或明確地名合理查證的真實地點為準。\n"
    "4. 若 grounding 資料不足、地點不明確，或無法確認店家仍在營業，請直接說明資料不足，寧可少推薦，也不要湊數。\n"
    "5. 不要宣稱你已查到即時營業狀態，除非 grounding 資料中明確支持。\n\n"
    "【回覆格式約束】：\n"
    "1. 前言必須極其簡短（嚴格限制在 30 字以內），直接切入主題，拒絕任何廢話。\n"
    "2. 使用點排列（Bullet points）呈現推薦店家。\n"
    "3. 每個推薦項目的結構必須嚴格遵守：\n"
    "   * 店名/景點名\n"
    "   * 【老饕碎碎念】：只能用一句話（20字內）精闢點出靈魂美味或特色，語氣要毒舌風趣。\n"
    "   * 【Google 地圖】：必須嚴格、一字不差地使用此格式提供搜尋連結：https://www.google.com/maps/search/?api=1&query=店家名稱+地點（嚴禁將前綴修改為其他任何網址形式）。\n"
    "   * 【刁嘴指數】：滿分5顆星（如：★★★★☆）。"
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

def extract_group_prompt(user_message: str) -> tuple[bool, str]:
    for keyword in BOT_KEYWORDS:
        if keyword in user_message:
            return True, user_message.replace(keyword, "", 1).strip()
    return False, user_message.strip()

def log_mention_metadata(event: MessageEvent):
    mention = getattr(event.message, "mention", None)
    if not mention:
        return

    if hasattr(mention, "to_dict"):
        mention_data = mention.to_dict()
    elif hasattr(mention, "model_dump"):
        mention_data = mention.model_dump()
    else:
        mention_data = repr(mention)

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
        is_triggered, clean_prompt = extract_group_prompt(user_message)
        if not is_triggered:
            return 
        
        group_id = get_conversation_id(event.source)
        cached_location = get_cached_location(group_location_cache, group_id)
        
        if cached_location:
            final_prompt = (
                f"使用者在群組中標記了你，並詢問：『{clean_prompt}』\n"
                f"請結合該群組最後提供的定位資訊與周邊進行回答：\n"
                f"定位名稱: {cached_location['title']}\n"
                f"地址: {cached_location['address']}\n"
                f"座標: 緯度 {cached_location['latitude']}, 經度 {cached_location['longitude']}"
            )
        else:
            if not clean_prompt or needs_location(clean_prompt):
                send_line_reply_or_push(reply_token, LOCATION_REQUIRED_MESSAGE, target_id)
                return
            final_prompt = clean_prompt
    else:
        cached_location = get_cached_location(user_location_cache, event.source.user_id) if isinstance(event.source, UserSource) else None
        
        if cached_location:
            final_prompt = (
                f"使用者在一對一聊天室詢問：『{user_message}』\n"
                f"請結合使用者最後提供的定位資訊與周邊進行回答：\n"
                f"定位名稱: {cached_location['title']}\n"
                f"地址: {cached_location['address']}\n"
                f"座標: 緯度 {cached_location['latitude']}, 經度 {cached_location['longitude']}"
            )
        else:
            if needs_location(user_message):
                send_line_reply_or_push(reply_token, LOCATION_REQUIRED_MESSAGE, target_id)
                return
            final_prompt = user_message

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
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=config
        )
        return format_gemini_response(response)
    except Exception:
        logger.exception("Gemini error.")
        return "本美食家大腦低血糖打結中，等我吃口飯！"

def build_gemini_config(user_query: str, location: dict | None = None):
    grounding_kind = get_grounding_kind(user_query)
    tools = []
    tool_config = None

    if grounding_kind == "search":
        tools.append(types.Tool(google_search=types.GoogleSearch()))
    elif grounding_kind == "maps":
        tools.append(types.Tool(google_maps=types.GoogleMaps()))
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
        max_output_tokens=1200,
        tools=tools or None,
        tool_config=tool_config
    )

def format_gemini_response(response) -> str:
    text = response.text or "本美食家突然失語，換個問法再來。"
    sources = collect_grounding_sources(response)
    if sources:
        text = f"{text}\n\n資料來源：\n" + "\n".join(sources)
    return limit_line_text(text)

def collect_grounding_sources(response) -> list[str]:
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
            source = ("Google Maps", chunk.maps.title, chunk.maps.uri)
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

def limit_line_text(text: str, max_length: int = 4500) -> str:
    if len(text) <= max_length:
        return text
    return text[:max_length - 20].rstrip() + "\n...（回覆過長已截斷）"

def send_line_reply(reply_token: str, text: str):
    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        line_bot_api.reply_message(
            ReplyMessageRequest(
                reply_token=reply_token,
                messages=[TextMessage(text=text)]
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
                messages=[TextMessage(text=text)]
            )
        )
