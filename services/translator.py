"""Translation Engine Supporting Multi-Provider Architecture (Gemini, Groq, NLLB-200, Qwen)"""
import asyncio
import copy
import gc
import json
import logging
import re
from types import SimpleNamespace
from typing import List, Optional, Dict, Any, Tuple
import httpx
import torch

from core.config import settings

logger = logging.getLogger(__name__)

def validate_provider_response(
    requested_batch: List[SimpleNamespace],
    raw_payload: Any
) -> Tuple[Dict[int, str], Dict[int, List[str]], List[str]]:
    """Strictly validates provider response against requested batch indices.
    
    Checks:
    - Enforces expected ID set: any index not in requested batch is flagged as UNEXPECTED_PROVIDER_ID.
    - Duplicate detection: duplicate occurrences of the same index are flagged as DUPLICATE_PROVIDER_ID.
    - Missing translation: any requested index not returned or returned empty is flagged as PROVIDER_MISSING_TRANSLATION.
    - Value type validation: non-string translations are flagged as INVALID_TRANSLATION_TYPE.
    
    Returns:
    - text_map: Dict[int, str] mapping valid index to clean translated text.
    - cue_warnings: Dict[int, List[str]] mapping index to cue-specific warnings.
    - batch_warnings: List[str] batch-level warnings.
    """
    expected_indices = {s.index for s in requested_batch}
    ans_map: Dict[int, str] = {}
    cue_warnings: Dict[int, List[str]] = {s.index: [] for s in requested_batch}
    batch_warnings: List[str] = []
    seen_indices = set()

    items = raw_payload
    if isinstance(items, dict):
        for val in items.values():
            if isinstance(val, list):
                items = val
                break

    if not isinstance(items, list):
        batch_warnings.append("MALFORMED_PROVIDER_PAYLOAD")
        items = []

    for item in items:
        if not isinstance(item, dict):
            batch_warnings.append("MALFORMED_PROVIDER_ITEM")
            continue

        if "index" not in item:
            batch_warnings.append("MALFORMED_PROVIDER_ITEM")
            continue

        raw_idx = item["index"]
        if type(raw_idx) is not int or isinstance(raw_idx, bool):
            batch_warnings.append("MALFORMED_PROVIDER_ID_TYPE")
            continue

        if raw_idx not in expected_indices:
            batch_warnings.append("UNEXPECTED_PROVIDER_ID")
            continue

        if raw_idx in seen_indices:
            cue_warnings[raw_idx].append("DUPLICATE_PROVIDER_ID")
            continue

        seen_indices.add(raw_idx)

        raw_text = item.get("arabic") if "arabic" in item else item.get("text", "")
        if not isinstance(raw_text, str):
            cue_warnings[raw_idx].append("INVALID_TRANSLATION_TYPE")
            continue

        clean_text = raw_text.strip()
        if not clean_text:
            cue_warnings[raw_idx].append("PROVIDER_EMPTY_TRANSLATION")
        else:
            ans_map[raw_idx] = clean_text

    for s in requested_batch:
        if s.index not in ans_map:
            if "PROVIDER_MISSING_TRANSLATION" not in cue_warnings[s.index]:
                cue_warnings[s.index].append("PROVIDER_MISSING_TRANSLATION")

    return ans_map, cue_warnings, batch_warnings

def reattach_segment(
    source_seg: SimpleNamespace,
    translated_text: Optional[str] = None,
    extra_warnings: Optional[List[str]] = None
) -> SimpleNamespace:
    """Clones source segment preserving 100% of event metadata (layer, style, margins, effect, speaker)."""
    new_seg = copy.copy(source_seg)
    if translated_text:
        new_seg.text = translated_text
    else:
        new_seg.text = source_seg.text

    merged_warnings = list(getattr(source_seg, "warnings", ()))
    if extra_warnings:
        for w in extra_warnings:
            if w not in merged_warnings:
                merged_warnings.append(w)
    new_seg.warnings = tuple(merged_warnings)
    return new_seg

# Specialized System Prompt for Cinematic Arabic Subtitles
CINEMATIC_PROMPT = """أنت مترجم سينمائي محترف ومبدع في ترجمة الأفلام والمسلسلات التلفزيونية العالمية إلى اللغة العربية الفصحى المعاصرة.
مهمتك ترجمة قائمة نصوص الحوار المرفقة ترجمة سينمائية دقيقة وطبيعية ودرامية تناسب شاشات السينما والمنصات العالمية (Netflix, HBO, Apple TV).

القواعد الإلزامية:
1. حافظ على السياق الدرامي والشعوري للمشهد، وابتعد تماماً عن الترجمة الحرفية الركيكة.
2. الحفاظ الصارم على أسماء الشخصيات، والأماكن، والمصطلحات، والأرقام دون تشويه.
3. الترجمة يجب أن تكون موجزة ومركزة تناسب سرعة قراءة المشاهد (لا تتجاوز 42 حرفاً للسطر).
4. أعد النتيجة بنفس أرقام الفهارس (index) المحددة تماماً بصيغة JSON حصراً، على شكل قائمة كائنات:
[{"index": 1, "arabic": "النص المترجم هنا"}, ...]
لا تضف أي شرح أو نصوص خارج بنية الـ JSON المطلوبة."""

NLLB_LANG_MAP = {
    "en": "eng_Latn",
    "ar": "arb_Arab",
    "fr": "fra_Latn",
    "es": "spa_Latn",
    "de": "deu_Latn",
    "it": "ita_Latn",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "zh": "zho_Hans",
    "ru": "rus_Cyrl",
    "tr": "tur_Latn",
    "pt": "por_Latn",
    "nl": "nld_Latn",
    "sv": "swe_Latn",
    "pl": "pol_Latn",
    "hi": "hin_Deva"
}

def resolve_nllb_source_language(source_lang: Optional[str]) -> Tuple[str, Tuple[str, ...]]:
    """Resolve the app's source map; warn when it assumes English for a code."""
    clean_lang = (source_lang or "en").lower().split("-")[0].split("_")[0]
    lang_code = NLLB_LANG_MAP.get(clean_lang, "eng_Latn")
    warnings = ()
    if source_lang and source_lang.strip() and clean_lang not in NLLB_LANG_MAP:
        warnings = ("UNSUPPORTED_LANGUAGE_FALLBACK",)
    return lang_code, warnings


class UnifiedTranslator:
    """Multi-provider neural translation engine with isolated VRAM lifecycles."""

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

    async def translate(
        self,
        segments: List[SimpleNamespace],
        source_language: str = "en",
        provider: str = "local",
        progress_cb=None,
        device: Optional[str] = None,
    ) -> List[SimpleNamespace]:
        """Translates segments using the requested provider backend."""
        if not segments:
            return []

        # If source is already Arabic, return as-is
        clean_lang = (source_language or "en").lower().split("-")[0].split("_")[0]
        if clean_lang == "ar":
            logger.info("Source is already Arabic, skipping translation.")
            return segments

        logger.info("Translating %d segments using provider='%s' (source_lang='%s')...", len(segments), provider, source_language)

        if provider == "gemini" and settings.GEMINI_API_KEY:
            return await self._translate_gemini(segments, source_language, progress_cb)
        elif provider == "groq" and settings.GROQ_API_KEY:
            return await self._translate_groq(segments, source_language, progress_cb)
        elif provider == "qwen":
            return await self._translate_local_qwen(segments, source_language, progress_cb)
        else:
            return await self._translate_nllb_local(segments, source_language, progress_cb, device=device)

    # =========================================================================
    # 1. Google Gemini API (Recommended Cinematic Option)
    # =========================================================================
    async def _translate_gemini(
        self,
        segments: List[SimpleNamespace],
        source_lang: str,
        progress_cb=None
    ) -> List[SimpleNamespace]:
        """Translates in bounded batches using Google Gemini API for Hollywood-standard dialogue."""
        logger.info("[Translator] Using Google Gemini API...")
        api_key = settings.GEMINI_API_KEY
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"

        batch_size = 20
        total = len(segments)
        translated_segments = []

        async with httpx.AsyncClient(timeout=90.0) as client:
            for start in range(0, total, batch_size):
                batch = segments[start:start + batch_size]
                items_payload = [{"index": s.index, "text": s.text} for s in batch]

                prompt_content = f"{CINEMATIC_PROMPT}\n\nنصوص الحوار المطلوبة:\n{json.dumps(items_payload, ensure_ascii=False)}"
                req_body = {
                    "contents": [{"parts": [{"text": prompt_content}]}],
                    "generationConfig": {"responseMimeType": "application/json"}
                }

                try:
                    resp = await client.post(url, json=req_body)
                    if resp.status_code == 200:
                        raw_ans = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
                        ans_list = json.loads(raw_ans)
                        ans_map, cue_warns, batch_warns = validate_provider_response(batch, ans_list)

                        for s in batch:
                            ar_text = ans_map.get(s.index)
                            warns = list(cue_warns.get(s.index, []))
                            if batch_warns:
                                warns.extend(batch_warns)
                            new_seg = reattach_segment(s, ar_text, extra_warnings=warns)
                            translated_segments.append(new_seg)
                    else:
                        logger.warning("Gemini API error (%d): %s. Falling back to local.", resp.status_code, resp.text)
                        fb = await self._translate_nllb_local(batch, source_lang)
                        translated_segments.extend(fb)
                except Exception as e:
                    logger.error("Gemini call exception: %s. Falling back to local.", e)
                    fb = await self._translate_nllb_local(batch, source_lang)
                    translated_segments.extend(fb)

                if progress_cb:
                    progress_cb(int((len(translated_segments) / total) * 100))

        return translated_segments

    # =========================================================================
    # 2. Groq Cloud API (Llama 3.3 70B Versatile)
    # =========================================================================
    async def _translate_groq(
        self,
        segments: List[SimpleNamespace],
        source_lang: str,
        progress_cb=None
    ) -> List[SimpleNamespace]:
        """Translates via Groq Llama-3.3-70b-versatile."""
        logger.info("[Translator] Using Groq Llama 3.3 API...")
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {settings.GROQ_API_KEY}"}

        batch_size = 20
        total = len(segments)
        translated_segments = []

        async with httpx.AsyncClient(timeout=90.0) as client:
            for start in range(0, total, batch_size):
                batch = segments[start:start + batch_size]
                items_payload = [{"index": s.index, "text": s.text} for s in batch]

                messages = [
                    {"role": "system", "content": CINEMATIC_PROMPT},
                    {"role": "user", "content": json.dumps(items_payload, ensure_ascii=False)}
                ]
                req_body = {
                    "model": "llama-3.3-70b-versatile",
                    "messages": messages,
                    "response_format": {"type": "json_object"}
                }

                try:
                    resp = await client.post(url, headers=headers, json=req_body)
                    if resp.status_code == 200:
                        content = resp.json()["choices"][0]["message"]["content"]
                        ans_data = json.loads(content)
                        items = ans_data if isinstance(ans_data, list) else list(ans_data.values())[0] if isinstance(ans_data, dict) and ans_data else []
                        ans_map, cue_warns, batch_warns = validate_provider_response(batch, items)

                        for s in batch:
                            ar_text = ans_map.get(s.index)
                            warns = list(cue_warns.get(s.index, []))
                            if batch_warns:
                                warns.extend(batch_warns)
                            new_seg = reattach_segment(s, ar_text, extra_warnings=warns)
                            translated_segments.append(new_seg)
                    else:
                        logger.warning("Groq API error (%d). Falling back to local.", resp.status_code)
                        fb = await self._translate_nllb_local(batch, source_lang)
                        translated_segments.extend(fb)
                except Exception as e:
                    logger.error("Groq call exception: %s. Falling back to local.", e)
                    fb = await self._translate_nllb_local(batch, source_lang)
                    translated_segments.extend(fb)

                if progress_cb:
                    progress_cb(int((len(translated_segments) / total) * 100))

        return translated_segments

    # =========================================================================
    # 3. Local Qwen via llama.cpp Server
    # =========================================================================
    async def _translate_local_qwen(
        self,
        segments: List[SimpleNamespace],
        source_lang: str,
        progress_cb=None
    ) -> List[SimpleNamespace]:
        """Translates via local OpenAI-compatible endpoint (e.g. Qwen 2.5 on llama.cpp)."""
        logger.info("[Translator] Connecting to Local Qwen server at %s...", settings.LOCAL_QWEN_URL)
        url = f"{settings.LOCAL_QWEN_URL}/chat/completions"
        batch_size = 15
        total = len(segments)
        translated_segments = []

        async with httpx.AsyncClient(timeout=120.0) as client:
            for start in range(0, total, batch_size):
                batch = segments[start:start + batch_size]
                items_payload = [{"index": s.index, "text": s.text} for s in batch]

                messages = [
                    {"role": "system", "content": CINEMATIC_PROMPT},
                    {"role": "user", "content": json.dumps(items_payload, ensure_ascii=False)}
                ]
                req_body = {
                    "model": "qwen",
                    "messages": messages,
                    "temperature": 0.2
                }

                try:
                    resp = await client.post(url, json=req_body)
                    if resp.status_code == 200:
                        content = resp.json()["choices"][0]["message"]["content"]
                        match = re.search(r"\[.*\]", content, re.DOTALL)
                        ans_list = json.loads(match.group(0)) if match else []
                        ans_map, cue_warns, batch_warns = validate_provider_response(batch, ans_list)

                        for s in batch:
                            ar_text = ans_map.get(s.index)
                            warns = list(cue_warns.get(s.index, []))
                            if batch_warns:
                                warns.extend(batch_warns)
                            new_seg = reattach_segment(s, ar_text, extra_warnings=warns)
                            translated_segments.append(new_seg)
                    else:
                        fb = await self._translate_nllb_local(batch, source_lang)
                        translated_segments.extend(fb)
                except Exception as e:
                    logger.warning("Local Qwen unavailable: %s. Falling back to NLLB-200.", e)
                    fb = await self._translate_nllb_local(batch, source_lang)
                    translated_segments.extend(fb)

                if progress_cb:
                    progress_cb(int((len(translated_segments) / total) * 100))

        return translated_segments

    # =========================================================================
    # 4. Local Meta NLLB-200 (Offline 100% On-Device)
    # =========================================================================
    async def _translate_nllb_local(
        self,
        segments: List[SimpleNamespace],
        source_lang: str,
        progress_cb=None,
        device: Optional[str] = None,
    ) -> List[SimpleNamespace]:
        """Translates locally using Meta NLLB-200 distilled 600M with lazy load & unload."""
        import copy
        import os

        from huggingface_hub import try_to_load_from_cache
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        model_name = "facebook/nllb-200-distilled-600M"
        pinned_revision = "a3e77be725cf30383f1faeb8d2f0b0ea98ac554e"

        def is_cached(filename, revision="main"):
            hit = try_to_load_from_cache(model_name, filename, revision=revision)
            return isinstance(hit, (str, os.PathLike)) and os.path.isfile(hit)
        saved_device = self.device
        saved_dtype = self.dtype
        if device is not None:
            requested_device = device.strip().lower()
            if requested_device == "cuda" and not torch.cuda.is_available():
                requested_device = "cpu"
            self.device = requested_device
            if requested_device != saved_device:
                self.dtype = getattr(
                    torch,
                    "float16" if requested_device == "cuda" else "float32",
                    saved_dtype,
                )

        logger.info("[NLLB Local] Lazily loading NLLB-200 onto %s (%s)...", self.device, self.dtype)

        model = None
        tokenizer = None
        try:
            model_options = {}
            if not (is_cached("model.safetensors") or is_cached("model.safetensors.index.json")):
                if not is_cached("model.safetensors", pinned_revision):
                    raise RuntimeError("Missing local NLLB safetensors weights. Complete local NLLB setup first.")

                from transformers import AutoConfig, GenerationConfig

                pinned_config = copy.deepcopy(AutoConfig.from_pretrained(model_name, local_files_only=True))
                pinned_config._commit_hash = pinned_revision
                if is_cached("generation_config.json"):
                    generation_config = GenerationConfig.from_pretrained(model_name, local_files_only=True)
                else:
                    generation_config = GenerationConfig.from_model_config(pinned_config)
                model_options.update(revision=pinned_revision, config=pinned_config,
                                     generation_config=generation_config)

            tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                torch_dtype=self.dtype,
                use_safetensors=True,
                local_files_only=True,
                **model_options,
            ).to(self.device)
            model.eval()

            arabic_id = tokenizer.convert_tokens_to_ids("arb_Arab")
            lang_code, language_warnings = resolve_nllb_source_language(source_lang)
            tokenizer.src_lang = lang_code

            batch_size = 6 if self.device == "cuda" else 2
            total = len(segments)
            translated = []

            def _translate_batch_sync(texts):
                max_input_tokens = 256

                def _integer_rows(value, label):
                    rows = value.tolist() if hasattr(value, "tolist") else value
                    if not isinstance(rows, (list, tuple)) or any(
                        not isinstance(row, (list, tuple))
                        or any(type(token) is not int or token < 0 for token in row)
                        for row in rows
                    ):
                        raise ValueError(f"NLLB {label} must contain integer rows")
                    return [list(row) for row in rows]

                def _resolve_id(key, *sources):
                    for src in sources:
                        if src is not None and hasattr(src, key):
                            val = getattr(src, key)
                            if val is not None:
                                return val
                        elif isinstance(src, dict) and key in src:
                            val = src[key]
                            if val is not None:
                                return val
                    return None

                gen_config = getattr(model, "generation_config", None)
                model_config = getattr(model, "config", None)

                raw_eos = _resolve_id("eos_token_id", gen_config, model_config, tokenizer)
                if raw_eos is None:
                    raise ValueError("NLLB generation configuration missing eos_token_id")
                if type(raw_eos) is int and raw_eos >= 0:
                    eos_ids = {raw_eos}
                elif (
                    isinstance(raw_eos, (list, tuple, set))
                    and raw_eos
                    and all(type(x) is int and x >= 0 for x in raw_eos)
                ):
                    eos_ids = set(raw_eos)
                else:
                    raise ValueError("NLLB generation configuration contains invalid eos_token_id")

                decoder_start_id = _resolve_id("decoder_start_token_id", gen_config, model_config, tokenizer)
                if decoder_start_id is None or type(decoder_start_id) is not int or decoder_start_id < 0:
                    raise ValueError("NLLB generation configuration missing or invalid decoder_start_token_id")

                pad_id = _resolve_id("pad_token_id", gen_config, model_config, tokenizer)
                if pad_id is not None and (type(pad_id) is not int or pad_id < 0):
                    raise ValueError("NLLB generation configuration contains invalid pad_token_id")

                admitted_ids, admitted_masks, ordinals = [], [], []
                results = [
                    {"action": "RETAIN_SOURCE", "text": None,
                     "warnings": ("INPUT_TOKEN_BUDGET_EXCEEDED", "SOURCE_RETAINED_FOR_REVIEW")}
                    for _ in texts
                ]
                for ordinal, text in enumerate(texts):
                    encoded = tokenizer(
                        [text], add_special_tokens=True, padding=False, truncation=False,
                        return_attention_mask=True, return_token_type_ids=False
                    )
                    if not hasattr(encoded, "get"):
                        raise ValueError("NLLB preflight encoding is malformed")
                    ids_rows = _integer_rows(encoded.get("input_ids"), "preflight IDs")
                    mask_rows = _integer_rows(encoded.get("attention_mask"), "preflight mask")
                    if len(ids_rows) != 1 or len(mask_rows) != 1:
                        raise ValueError("NLLB preflight must return exactly one cue")
                    ids, mask = ids_rows[0], mask_rows[0]
                    if not ids or len(ids) != len(mask) or any(bit != 1 for bit in mask):
                        raise ValueError("NLLB preflight requires an unpadded cue and matching mask")
                    if len(ids) <= max_input_tokens:
                        admitted_ids.append(ids)
                        admitted_masks.append(mask)
                        ordinals.append(ordinal)
                    # Never retain multiple oversized encodings or regenerate from a clipped prefix.
                    del encoded, ids_rows, mask_rows, ids, mask

                if not admitted_ids:
                    return results
                inputs = tokenizer.pad(
                    {"input_ids": [list(row) for row in admitted_ids],
                     "attention_mask": [list(row) for row in admitted_masks]},
                    padding=True, return_tensors="pt"
                )
                if not hasattr(inputs, "get"):
                    raise ValueError("NLLB padded encoding is malformed")
                padded_ids = _integer_rows(inputs.get("input_ids"), "padded IDs")
                padded_masks = _integer_rows(inputs.get("attention_mask"), "padded mask")
                if len(padded_ids) != len(ordinals) or len(padded_masks) != len(ordinals):
                    raise ValueError("NLLB padding changed cue cardinality")
                width = len(padded_ids[0])
                if not width or width > max_input_tokens:
                    raise ValueError("NLLB padded input exceeds the admitted budget")
                for ordinal, (ids, mask) in enumerate(zip(padded_ids, padded_masks)):
                    if (len(ids) != width or len(mask) != width
                            or any(bit not in (0, 1) for bit in mask)
                            or [token for token, bit in zip(ids, mask) if bit] != admitted_ids[ordinal]):
                        raise ValueError("NLLB padding changed prepared input tokens")
                inputs = inputs.to(self.device)
                with torch.no_grad():
                    generated = model.generate(
                        **inputs,
                        forced_bos_token_id=arabic_id,
                        max_new_tokens=140,
                        num_beams=2,
                        do_sample=False
                    )

                gen_rows = _integer_rows(generated, "generated tokens")
                if len(gen_rows) != len(ordinals):
                    raise ValueError("NLLB generation changed cue cardinality")

                for ordinal_idx, raw_row in enumerate(gen_rows):
                    ordinal = ordinals[ordinal_idx]
                    if not raw_row or raw_row[0] != decoder_start_id:
                        raise ValueError("NLLB generated sequence missing expected decoder start token")

                    gen_tokens = raw_row[1:]
                    first_eos_idx = next((i for i, tok in enumerate(gen_tokens) if tok in eos_ids), None)

                    if first_eos_idx is not None:
                        gen_step = first_eos_idx + 1
                        suffix = gen_tokens[first_eos_idx + 1:]
                        if pad_id is None:
                            if suffix:
                                raise ValueError("NLLB generated tokens follow EOS without pad token ID configured")
                        else:
                            if any(tok != pad_id for tok in suffix):
                                raise ValueError("NLLB generated tokens contain non-padding following EOS")
                        if gen_step > 140:
                            raise ValueError("NLLB active generated count exceeds maximum allowed generation ceiling")
                        if gen_step == 140:
                            status = "BUDGET_EXHAUSTED"
                        else:
                            status = "COMPLETE"
                    else:
                        active_tokens = list(gen_tokens)
                        if pad_id is not None and pad_id not in eos_ids:
                            while active_tokens and active_tokens[-1] == pad_id:
                                active_tokens.pop()
                        active_count = len(active_tokens)
                        if active_count > 140:
                            raise ValueError("NLLB active generated count exceeds maximum allowed generation ceiling")
                        if active_count == 140:
                            status = "BUDGET_EXHAUSTED"
                        else:
                            status = "INCOMPLETE"

                    raw_item = generated[ordinal_idx] if hasattr(generated, "__getitem__") else raw_row
                    decoded_text = tokenizer.decode(raw_item, skip_special_tokens=True)
                    if not isinstance(decoded_text, str):
                        raise ValueError("NLLB decoded output must be a string")
                    decoded_text = decoded_text.strip()
                    is_empty = (len(decoded_text) == 0)

                    if status == "COMPLETE":
                        if is_empty:
                            results[ordinal] = {
                                "action": "RETAIN_SOURCE",
                                "text": None,
                                "warnings": ("EMPTY_TRANSLATION_OUTPUT", "SOURCE_RETAINED_FOR_REVIEW"),
                            }
                        else:
                            results[ordinal] = {
                                "action": "TRANSLATED",
                                "text": decoded_text,
                                "warnings": tuple(language_warnings),
                            }
                    elif status == "BUDGET_EXHAUSTED":
                        warns = (
                            ("OUTPUT_TOKEN_BUDGET_EXCEEDED", "EMPTY_TRANSLATION_OUTPUT", "SOURCE_RETAINED_FOR_REVIEW")
                            if is_empty
                            else ("OUTPUT_TOKEN_BUDGET_EXCEEDED", "SOURCE_RETAINED_FOR_REVIEW")
                        )
                        results[ordinal] = {
                            "action": "RETAIN_SOURCE",
                            "text": None,
                            "warnings": warns,
                        }
                    else:  # INCOMPLETE
                        warns = (
                            ("OUTPUT_GENERATION_INCOMPLETE", "EMPTY_TRANSLATION_OUTPUT", "SOURCE_RETAINED_FOR_REVIEW")
                            if is_empty
                            else ("OUTPUT_GENERATION_INCOMPLETE", "SOURCE_RETAINED_FOR_REVIEW")
                        )
                        results[ordinal] = {
                            "action": "RETAIN_SOURCE",
                            "text": None,
                            "warnings": warns,
                        }

                return results

            for start in range(0, total, batch_size):
                batch = segments[start:start + batch_size]
                texts = [s.text for s in batch]
                batch_results = await asyncio.to_thread(_translate_batch_sync, texts)
                if len(batch_results) != len(batch):
                    raise ValueError("NLLB worker changed original batch cardinality")
                for s, res in zip(batch, batch_results):
                    new_seg = copy.copy(s)
                    if res["action"] == "RETAIN_SOURCE":
                        new_seg.text = s.text
                    else:
                        new_seg.text = res["text"]

                    orig_has_warnings = hasattr(s, "warnings")
                    orig_warnings = getattr(s, "warnings", ())
                    if not isinstance(orig_warnings, (list, tuple)):
                        orig_warnings = ()
                    result_warnings = res.get("warnings", ())
                    if not isinstance(result_warnings, (list, tuple)):
                        result_warnings = ()

                    combined = list(orig_warnings) + list(result_warnings)
                    deduped = []
                    for w in combined:
                        if w not in deduped:
                            deduped.append(w)

                    if deduped or orig_has_warnings:
                        new_seg.warnings = tuple(deduped)
                    translated.append(new_seg)

                if progress_cb:
                    progress_cb(int((len(translated) / total) * 100))

            return translated

        finally:
            self.device = saved_device
            self.dtype = saved_dtype
            # Guaranteed cleanup in all conditions
            if model is not None:
                del model
            if tokenizer is not None:
                del tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

translator = UnifiedTranslator()
