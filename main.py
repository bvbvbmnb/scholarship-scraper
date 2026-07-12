import os
import json
import hashlib
import logging
import re
from dotenv import load_dotenv
import requests
from google import genai
from google.genai import types

# Load configuration values from the .env file (no-op on GitHub Actions, where
# these are injected as real environment variables from repo secrets instead)
load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ADMIN_USER_ID = os.getenv("ADMIN_USER_ID")
DESTINATION_CHANNEL = os.getenv("DESTINATION_CHANNEL")

# Optional proxy support for regions where t.me is blocked at the network level.
# Leave PROXY_URL empty/unset in .env to make this a no-op (direct connection).
# Not needed on GitHub Actions runners, but kept for local runs.
PROXY_URL = os.getenv("PROXY_URL", "").strip()
PROXIES = {"http": PROXY_URL, "https": PROXY_URL} if PROXY_URL else None

# Where processed-post fingerprints are persisted between runs.
# The GitHub Actions workflow commits this file back to the repo after each run.
STATE_FILE = os.getenv("STATE_FILE", "processed_posts.json")

# Cap on how many fingerprints we keep, so the state file doesn't grow forever.
MAX_STATE_ENTRIES = 3000

# Channels to monitor automatically
CHANNELS_TO_WATCH = [
    "Scholarship_holding_pen",
    "edugrandsuz", "kukukakaaa", "grantscholar",
    "opportunities_zula", "opcorners", "scholarshipscorner",
    "BrightScholarship", "studyqa", "nucleus_borziyon", "studygrants"
]

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize the official Google GenAI Client
ai_client = genai.Client(api_key=GEMINI_API_KEY)

# Initialize a web browser session simulation
session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5'
})
if PROXIES:
    session.proxies.update(PROXIES)
    logging.info(f"Routing outbound requests through proxy: {PROXY_URL.split('@')[-1]}")

AI_SYSTEM_PROMPT = """
You are an expert scholarship database coordinator. Your job is to extract information from the text or link provided and format it into a Telegram card.

You must output ONLY the raw formatted text block. Do not include markdown code wrappers, intro lines, or greetings.

📢 type: [lowercase level: bs, ms, phd, fellowships, or others]
🎓 program: [Program Name] ([University/City, Country])
⏳ deadline: [Day Month Year]
📝 ielts required: [yes / no / not stated]
💸 app fee: [yes / no]
🔗 link: [URL to apply]

[A 2-to-3 sentence detailed paragraph summarizing what the scholarship covers, eligibility, value, and specific target groups. Use universal, simple language.]

STRICT RULES:
1. Translate any source text in Uzbek, Russian, or other languages into clean, professional English.
2. USE THE GOOGLE SEARCH TOOL to look up the specific program name or link. Verify if IELTS/TOEFL is required and if an application fee exists. Update the fields accurately based on your search results.
3. Current year is 2026. If the deadline has passed relative to 2026, or if the program is explicitly not open yet, output exactly: "SKIP: Deadline passed or not open yet."
"""


def stable_hash(text: str) -> str:
    """
    A hash that's the same across separate process runs.
    NOTE: Python's built-in hash() is randomized per-process for strings, so it
    cannot be used to compare against fingerprints saved in a previous run
    (which is exactly what we need now that state persists across GitHub
    Actions runs instead of living in one long-lived process).
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_processed_posts() -> set:
    if not os.path.exists(STATE_FILE):
        return set()
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return set(data.get("processed", []))
    except Exception as e:
        logging.error(f"Could not read state file, starting fresh: {e}")
        return set()


def save_processed_posts(processed: set):
    # Keep only the most recent entries so the file doesn't grow unbounded.
    # (We don't have per-item timestamps, so this just caps total size;
    # order isn't meaningful here, it's a simple ring-buffer-style trim.)
    trimmed = list(processed)[-MAX_STATE_ENTRIES:]
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump({"processed": trimmed}, f)
    except Exception as e:
        logging.error(f"Failed to write state file: {e}")


def send_telegram_message(text: str):
    """Sends a message to the Admin user using standard web requests."""
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": ADMIN_USER_ID,
        "text": text,
        "parse_mode": "Markdown"
    }
    try:
        session.post(url, data=payload, timeout=15)
    except Exception as e:
        logging.error(f"Failed to communicate with Telegram API: {e}")


def check_channels_once(processed_posts: set) -> set:
    """
    Does a single pass over every watched channel, sends approval cards for
    anything new, and returns the updated set of processed-post fingerprints.
    """
    is_first_run = len(processed_posts) == 0

    for tg_chan in CHANNELS_TO_WATCH:
        try:
            url = f"https://t.me/{tg_chan}"
            res = session.get(url, timeout=15)

            # Extract text container blocks via regex patterns
            msg_blocks = re.findall(r'<div class="tgme_widget_message_text[^"]*"[^>]*>(.*?)</div>', res.text, re.DOTALL)
            if not msg_blocks:
                continue

            latest_raw_msg = msg_blocks[-1]
            clean_text = re.sub(r'<[^>]*>', '', latest_raw_msg).strip()

            post_hash = stable_hash(clean_text)
            if post_hash in processed_posts:
                continue

            processed_posts.add(post_hash)

            if is_first_run:
                # First-ever run: just cache the current baseline so we don't
                # spam approval cards for posts that already existed.
                continue

            logging.info(f"New update found on public channel: @{tg_chan}")

            # Send details to Gemini with real-time web verification tools active
            response = ai_client.models.generate_content(
                model='gemini-1.5-flash',
                contents=clean_text,
                config=types.GenerateContentConfig(
                    system_instruction=AI_SYSTEM_PROMPT,
                    tools=[types.Tool(google_search=types.GoogleSearch())]
                )
            )
            generated_card = response.text.strip()

            if "SKIP:" in generated_card:
                logging.info(f"Skipped expired or unopened post from @{tg_chan}")
                continue

            approval_instructions = (
                f"📋 **NEW SCHOLARSHIP CARD GENERATED**\n"
                f"**Source:** @{tg_chan}\n\n"
                f"{generated_card}\n\n"
                f"📌 **HOW TO APPROVE:**\n"
                f"To publish this card to your public channel, copy the raw card layout block above, "
                f"send it directly to your bot chat or forward it directly to your target channel."
            )

            send_telegram_message(approval_instructions)

        except Exception as e:
            logging.error(f"Error checking channel @{tg_chan}: {e}")

    if is_first_run:
        print("Baseline cached on first run. Future runs will report new posts only.")

    return processed_posts


if __name__ == "__main__":
    print("Scraper run starting (single pass, triggered by scheduler)...")
    processed_posts = load_processed_posts()
    processed_posts = check_channels_once(processed_posts)
    save_processed_posts(processed_posts)
    print("Scraper run complete.")
