import json
import os
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv

load_dotenv()

PORT = int(os.getenv("PORT", "8001"))
LLM_PROVIDER = "groq"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")


def get_meal_bucket(hour: int) -> str:
    if 6 <= hour < 11:
        return "morning"
    if 11 <= hour < 15:
        return "lunch"
    return "dinner"


def get_fallback_reason(restaurant: dict, hour: int) -> str:
    category = restaurant.get("categoryKey") or restaurant.get("category") or "기타"
    bucket = get_meal_bucket(hour)

    if bucket == "morning":
        if category == "카페":
            return "아침 시간에는 가볍게 마실 수 있는 카페류가 오전 분위기와 잘 맞습니다."
        if category == "샌드위치":
            return "아침 식사로는 샌드위치가 빠르게 해결하기 좋아서 추천합니다."
        return "이 시간대에는 가볍고 빠르게 먹을 수 있는 조합이 더 자연스럽습니다."

    if bucket == "lunch":
        if category == "한식":
            return "점심에는 속도도 챙기면서 포만감까지 잡아주는 한식이 더 잘 어울립니다."
        if category == "일식":
            return "점심시간엔 가볍게 먹되 만족감이 높은 일식 계열이 무난하게 추천됩니다."
        if category == "중식":
            return "점심에는 면 요리 중심의 중식이 한 끼로 충분한 편입니다."
        return "점심시간대에 무난하게 선택할 수 있는 메뉴 구성이 좋아 보입니다."

    if category == "치킨":
        return "저녁에는 간단하게 해결하면서도 만족감이 높은 치킨류가 잘 맞습니다."
    if category == "양식":
        return "저녁엔 든든한 양식 메뉴가 식사 흐름을 살리기 좋습니다."
    if category == "일식":
        return "저녁에는 가볍게 먹되 만족감이 높은 일식 계열이 특히 무난합니다."
    return "저녁 시간대에는 포만감이 높은 조합이 더 자연스럽게 느껴집니다."


def build_prompt(payload: dict) -> str:
    if not isinstance(payload, dict):
        payload = {}

    restaurant = payload.get("restaurant", {})
    if not isinstance(restaurant, dict):
        restaurant = {}

    name = restaurant.get("name", "이 식당")
    category = restaurant.get("categoryKey") or restaurant.get("category") or "기타"
    distance = restaurant.get("distance", "500m 이내")
    rating = restaurant.get("rating", 4.0)
    reviewCount = restaurant.get("reviewCount", 0)
    hour = int(payload.get("hour", 12) or 12)
    history = payload.get("history", [])
    if not isinstance(history, list):
        history = []
    meal = get_meal_bucket(hour)

    history_text = ", ".join(history) if history else "최근에 특별한 선호 정보가 아직 없습니다"

    return (
        f"너는 맛집을 잘 알고 친절하게 추천해주는 미식 가이드다. "
        f"아래 식당을 {meal} 시간대에 맞춰 자연스럽고 센스 있게 추천해줘.\n"
        f"- 식당명: {name}\n"
        f"- 카테고리: {category}\n"
        f"- 거리: {distance}\n"
        f"- 평점: {rating}\n"
        f"조건:\n"
        f"1. '저녁 시간에 ~를 추천합니다' 같은 로봇 같은 고정 틀을 절대 쓰지 마라, 또한 정중하게 존대를 써서 추천이유를 말해라.\n"
        f"2. 거리나 음식 종류, 시간대 특징을 살려 당장 가고 싶게 만들어라.\n"
        f"3. 절대 가중치나 수치를 언급하지 마라.\n"
        f"4. 한국어 2문장 이내로 자연스럽게 써라."
    )


def call_gemini(prompt: str) -> str | None:
    if not GEMINI_API_KEY:
        return None

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }

    request = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(request, timeout=12) as res:
            body = json.loads(res.read().decode("utf-8"))
            candidates = body.get("candidates") or []
            for candidate in candidates:
                parts = candidate.get("content", {}).get("parts") or []
                for part in parts:
                    text = part.get("text")
                    if text:
                        return text.strip()
    except (urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError, TimeoutError):
        return None

    return None


def call_groq(prompt: str) -> str | None:
    if not GROQ_API_KEY:
        print("GROQ_CALL_SKIPPED: missing key")
        return None

    # 가장 빠른 표준 Groq 모델인 llama-3.1-8b-instant 적용
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {
                "role": "system",
                "content": "너는 미식가 큐레이터다. 로봇처럼 상투적인 문구를 쓰지 말고, 존대를 쓰고 친근하게 합리적인 이유를 제시하며 매력적으로 한국어 2문장 이내로 작성해라."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.7
    }

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {GROQ_API_KEY.strip()}"
            },
            json=payload,
            timeout=12
        )
        response.raise_for_status()
        body = response.json()
        choices = body.get("choices") or []
        for choice in choices:
            message = choice.get("message") or {}
            text = message.get("content")
            if text:
                print("★ GROQ API 호출 성공!")
                return text.strip()
    except requests.RequestException as exc:
        print("GROQ_REQUEST_ERROR:", exc)
        if hasattr(exc, 'response') and exc.response is not None:
            print("Groq Response Body:", exc.response.text)
        return None

    return None


def call_llm(prompt: str) -> str | None:
    if LLM_PROVIDER == "groq":
        return call_groq(prompt)
    return call_gemini(prompt)


class Handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _send_json(self, status: int, body: dict):
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/recommend":
            self._send_json(404, {"error": "not found"})
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(content_length) if content_length > 0 else b"{}"
            payload = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json(400, {"error": "invalid json"})
            return

        prompt = build_prompt(payload)
        reason = call_llm(prompt)

        if not reason:
            print("★ Groq 호출 실패로 Fallback 대체 문구가 출력되었습니다.")
            reason = get_fallback_reason(payload.get("restaurant", {}), payload.get("hour", 12))

        self._send_json(200, {"reason": reason})

    def log_message(self, format, *args):
        return


if __name__ == "__main__":
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"LLM proxy server started at http://127.0.0.1:{PORT}")
    print(f"provider={LLM_PROVIDER} groq_key_loaded={bool(GROQ_API_KEY)} gemini_key_loaded={bool(GEMINI_API_KEY)}")
    server.serve_forever()