"""
Image-based plant diagnosis using Groq's vision-capable chat completions API.

Groq's multimodal lineup changes fairly often, so the model id is kept in one
place (VISION_MODEL below) rather than hard-coded across pages. If this stops
working, check https://console.groq.com/docs/vision for the current
production-ready vision model id and update the constant.
"""

import base64
import re

from groq import Groq

VISION_MODEL = "qwen/qwen3.6-27b"  # preview vision model, per Groq docs as of mid-2026


def encode_image_bytes(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")


def _mime_from_filename(filename: str) -> str:
    ext = (filename or "").lower().rsplit(".", 1)[-1]
    return {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "webp": "image/webp",
    }.get(ext, "image/jpeg")


def diagnose_image(api_key: str, image_bytes: bytes, filename: str, known_issues: list[str]) -> dict:
    """
    Sends the uploaded photo to a Groq vision model and asks it to identify
    the most likely plant health issue, plus general care/cure guidance.

    Returns a dict: {"matched_issue": str | None, "raw_response": str, "error": str | None}
    `matched_issue` is set only if the model's first line names one of our
    known issues closely enough to reuse our structured care data for it;
    otherwise the caller should fall back to showing `raw_response` as-is.
    """
    try:
        if not api_key:
            raise ValueError("GROQ_API_KEY is missing. Add it to the project-root .env file.")
        client = Groq(api_key=api_key)
        b64 = encode_image_bytes(image_bytes)
        mime = _mime_from_filename(filename)

        issue_list_str = ", ".join(known_issues)

        prompt = f"""You are a plant-care assistant looking at a photo of a plant.

Known issue categories this app can give detailed structured care plans for:
{issue_list_str}

Look at the photo and:
1. On the very first line, write "MATCH: <issue name>" using the EXACT name from
   the list above if the photo clearly resembles one of those issues, or
   "MATCH: none" if it doesn't clearly match any of them (e.g. it looks healthy,
   or shows something not on the list).
2. Then, in a few short paragraphs, describe what you observe in the photo, your
   best guess at what's going on (even if it's not an exact match to the list),
   and general care/cure suggestions.
3. Keep the tone friendly and clear. Do not claim certainty you don't have —
   photos alone can't always distinguish similar-looking issues.
"""

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                ],
            }
        ]

        # Vision requests can occasionally fail transiently (429/5xx/timeouts).
        # Retry a few times so the first click does not randomly fail while a
        # second click succeeds. Do not retry permanent 4xx errors such as an
        # invalid API key or an unsupported request.
        completion = None
        last_error = None
        for attempt in range(4):
            try:
                completion = client.chat.completions.create(
                    model=VISION_MODEL,
                    reasoning_format="hidden",
                    reasoning_effort="none",
                    messages=messages,
                    temperature=0.4,
                    max_completion_tokens=700,
                )
                break
            except Exception as exc:
                last_error = exc
                text = str(exc).lower()
                retryable = any(token in text for token in (
                    "429", "rate limit", "too many requests", "500", "502",
                    "503", "504", "timeout", "timed out", "temporarily",
                    "connection reset", "connection error",
                ))
                if not retryable or attempt == 3:
                    raise
                import time
                time.sleep(1.5 * (2 ** attempt))

        if completion is None:
            raise last_error or RuntimeError("Vision model returned no response.")

        raw = completion.choices[0].message.content or ""

        # Defensive cleanup: never expose model reasoning if a provider/model
        # returns it despite the hidden reasoning request.
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.IGNORECASE | re.DOTALL)
        raw = re.sub(r"</?think(?:ing)?>", "", raw, flags=re.IGNORECASE).strip()

        matched_issue = None
        first_line_match = re.match(r"\s*MATCH:\s*(.+)", raw.strip(), re.IGNORECASE)
        if first_line_match:
            candidate = first_line_match.group(1).strip().strip(".")
            for issue in known_issues:
                if candidate.lower() == issue.lower():
                    matched_issue = issue
                    break
            # strip the MATCH line out of the display text either way
            raw = raw.strip().split("\n", 1)[1].strip() if "\n" in raw.strip() else ""

        return {"matched_issue": matched_issue, "raw_response": raw, "error": None}

    except Exception as e:
        return {"matched_issue": None, "raw_response": "", "error": str(e)}
