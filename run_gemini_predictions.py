"""
HLE Evaluation: Gemini 2.5 Flash Predictions
Adapted from https://github.com/centerforaisafety/hle

Uses Google's Gemini API (via google-genai SDK) to generate responses,
then formats them identically to the original HLE pipeline so the
o3-mini judge script can be run unchanged.

Usage:
    python run_gemini_predictions.py \
        --google_api_key YOUR_GOOGLE_KEY \
        --dataset cais/hle \
        --max_completion_tokens 8192 \
        --num_workers 20
        # --max_samples 10   ← add this to do a quick smoke-test
"""

import os
import json
import argparse
import asyncio
import base64
from io import BytesIO

import httpx
from datasets import load_dataset
from tqdm.asyncio import tqdm_asyncio

# ---------------------------------------------------------------------------
# Prompt (identical to the upstream HLE repo)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "Your response should be in the following format:\n"
    "Explanation: {your explanation for your answer choice}\n"
    "Answer: {your chosen answer}\n"
    "Confidence: {your confidence score between 0% and 100% for your answer}"
)

MODEL = "gemini-2.5-flash-lite"


# ---------------------------------------------------------------------------
# Async Gemini client (thin wrapper around the REST API so we don't need
# the heavyweight google-generativeai SDK and can control concurrency easily)
# ---------------------------------------------------------------------------
class AsyncGeminiClient:
    """
    Minimal async wrapper around the Gemini generateContent REST endpoint.
    Supports text-only and image+text prompts.
    """

    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(self, api_key: str, timeout: float = 600.0):
        self.api_key = api_key
        self._http = httpx.AsyncClient(timeout=timeout)

    async def close(self):
        await self._http.aclose()

    async def generate(
        self,
        model: str,
        system_prompt: str,
        user_text: str,
        image_url: str | None = None,
        max_output_tokens: int | None = None,
        temperature: float = 0.0,
    ) -> tuple[str, dict]:
        parts = []

        if image_url:
            if image_url.startswith("data:image"):
                # Base64 data URI — extract directly, no HTTP fetch needed
                header, b64data = image_url.split(",", 1)
                mime = header.split(":")[1].split(";")[0]  # e.g. "image/png"
                parts.append({
                    "inline_data": {
                        "mime_type": mime,
                        "data": b64data,  # already base64 encoded
                    }
                })
            else:
                # Regular URL — fetch and encode as before
                img_bytes = await self._fetch_image(image_url)
                mime = self._guess_mime(image_url)
                parts.append({
                    "inline_data": {
                        "mime_type": mime,
                        "data": base64.b64encode(img_bytes).decode(),
                    }
                })

        parts.append({"text": user_text})

        body: dict = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": temperature,
            },
        }
        if max_output_tokens:
            body["generationConfig"]["maxOutputTokens"] = max_output_tokens

        url = f"{self.BASE_URL}/{model}:generateContent?key={self.api_key}"
        resp = await self._http.post(url, json=body)
        if resp.status_code == 400:
            print(f"400 error body: {resp.text}")
        resp.raise_for_status()
        data = resp.json()

        # Extract text
        candidates = data.get("candidates", [])
        if not candidates:
            raise ValueError(f"No candidates in response: {data}")
        content_text = candidates[0]["content"]["parts"][0]["text"]

        # Usage metadata (mirrored to match original pipeline's structure)
        usage_meta = data.get("usageMetadata", {})
        usage = {
            "prompt_tokens": usage_meta.get("promptTokenCount", 0),
            "completion_tokens": usage_meta.get("candidatesTokenCount", 0),
            "total_tokens": usage_meta.get("totalTokenCount", 0),
        }
        return content_text, usage

    async def _fetch_image(self, url: str) -> bytes:
        r = await self._http.get(url)
        r.raise_for_status()
        return r.content

    @staticmethod
    def _guess_mime(url: str) -> str:
        url_lower = url.lower()
        if url_lower.endswith(".png"):
            return "image/png"
        if url_lower.endswith(".gif"):
            return "image/gif"
        if url_lower.endswith(".webp"):
            return "image/webp"
        return "image/jpeg"  # safe default


# ---------------------------------------------------------------------------
# Core eval logic
# ---------------------------------------------------------------------------
client: AsyncGeminiClient  # set in main()


async def attempt_question(question, args) -> tuple[str, str, dict] | None:
    """Returns (question_id, response_text, usage) or None on failure."""
    try:
        response_text, usage = await client.generate(
            model=MODEL,
            system_prompt=SYSTEM_PROMPT,
            user_text=question["question"],
            image_url=question["image"] if question["image"] else None,
            max_output_tokens=args.max_completion_tokens,
            temperature=args.temperature,
        )
    except Exception as e:
        print(f"Error on question {question['id']}: {e}")
        return None

    if response_text is None:
        return None

    return question["id"], response_text, usage


async def attempt_all(questions, args) -> list:
    semaphore = asyncio.Semaphore(args.num_workers)

    async def bound_func(q):
        async with semaphore:
            return await attempt_question(q, args)

    tasks = [bound_func(q) for q in questions]
    return await tqdm_asyncio.gather(*tasks)


def main(args):
    global client

    assert args.num_workers > 1, "num_workers must be 2 or greater"

    client = AsyncGeminiClient(api_key=args.google_api_key)

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    print(f"Loading dataset: {args.dataset}")
    dataset = load_dataset(args.dataset, split="test").to_dict()
    questions = [
        dict(zip(dataset.keys(), values)) for values in zip(*dataset.values())
    ]

    if args.max_samples:
        questions = questions[: args.max_samples]
        print(f"Limiting to {args.max_samples} samples")

    # ------------------------------------------------------------------
    # Resume from existing predictions file
    # ------------------------------------------------------------------
    model_slug = MODEL.replace("/", "_")
    output_filepath = f"{model_slug}_results.json"

    if os.path.exists(output_filepath):
        with open(output_filepath, "r") as f:
            predictions = json.load(f)
        already_done = len(predictions)
        questions = [q for q in questions if q["id"] not in predictions]
        print(
            f"Resuming: {already_done} already answered, {len(questions)} remaining"
        )
    else:
        predictions = {}

    if not questions:
        print("All questions already answered. Skipping prediction step.")
    else:
        # ------------------------------------------------------------------
        # Run predictions
        # ------------------------------------------------------------------
        print(
            f"\nRunning {len(questions)} questions with {MODEL} "
            f"(num_workers={args.num_workers}) …"
        )
        results = asyncio.run(attempt_all(questions, args))

        for result in results:
            if result is None:
                continue
            unique_id, response, usage = result
            predictions[unique_id] = {
                "model": MODEL,
                "response": response,
                "usage": usage,
            }

        # Cache to disk after every run so it's safe to resume
        with open(output_filepath, "w") as f:
            json.dump(predictions, f, indent=4)
        print(f"\nPredictions saved → {output_filepath}")

    asyncio.run(client.close())
    return output_filepath


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run HLE evaluation with Gemini 2.5 Flash"
    )
    parser.add_argument(
        "--google_api_key", type=str, required=True, help="Google AI API key"
    )
    parser.add_argument(
        "--dataset", type=str, default="cais/hle", help="HuggingFace dataset path"
    )
    parser.add_argument(
        "--max_completion_tokens",
        type=int,
        default=8192,
        help="Max output tokens for Gemini",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="Sampling temperature"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=20,
        help="Concurrent request limit (tune to your quota)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit to first N questions (useful for testing)",
    )
    args = parser.parse_args()
    main(args)
