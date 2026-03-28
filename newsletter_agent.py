import argparse
import html
import json
import os
import platform
import re
import sqlite3
import ssl
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

from newsletter_schema import DB_PATH, initialize_database

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_ROOT = os.environ.get(
    "NEWSLETTER_AGENT_MODEL_ROOT",
    str((PROJECT_ROOT / "models").resolve()),
)

DEFAULT_MLX_MODEL_PATH = os.environ.get(
    "NEWSLETTER_AGENT_MODEL_MLX",
    str((Path(DEFAULT_MODEL_ROOT) / "mlx-community" / "gemma-3-4b-it-4bit").resolve()),
)
DEFAULT_TRANSFORMERS_MODEL_PATH = os.environ.get(
    "NEWSLETTER_AGENT_MODEL_TRANSFORMERS",
    str((Path(DEFAULT_MODEL_ROOT) / "google" / "gemma-2-2b-it").resolve()),
)
DEFAULT_MODEL_PATH = os.environ.get("NEWSLETTER_AGENT_MODEL", DEFAULT_MLX_MODEL_PATH)

DEFAULT_BROWSER_MODEL_ID = os.environ.get("NEWSLETTER_AGENT_BROWSER_MODEL_ID", "").strip()
DEFAULT_BROWSER_MODEL_CANDIDATES = tuple(
    candidate
    for candidate in (
        DEFAULT_BROWSER_MODEL_ID,
        "onnx-community/gemma-3n-E4B-it-ONNX",
        "gemma-3n-E4B-it-ONNX",
        "onnx-community/gemma-3n-E2B-it-ONNX",
        "gemma-3n-E2B-it-ONNX",
        "gemma-3-it",
    )
    if candidate
)

DEFAULT_DAYS = 7
DEFAULT_QUERIES = 4
DEFAULT_RESULTS_PER_QUERY = 3
MAX_ARTICLE_CHARS = 1600
REQUEST_TIMEOUT_SECONDS = 6
DEFAULT_RAM_RESERVE_GB = float(os.environ.get("NEWSLETTER_AGENT_RAM_RESERVE_GB", "4"))
MAX_NEWSLETTER_RUNTIME_SECONDS = int(
    float(os.environ.get("NEWSLETTER_AGENT_MAX_RUNTIME_SECONDS", "300"))
)
DEFAULT_WRITING_STYLE = (
    "Sharp, analytical, and premium. Write like a high-end newsletter editor, not a corporate content bot."
)

SEARCH_NOISE_WORDS = {
    "a",
    "about",
    "after",
    "analysis",
    "and",
    "around",
    "before",
    "change",
    "changed",
    "commentary",
    "day",
    "days",
    "development",
    "developments",
    "expert",
    "for",
    "from",
    "how",
    "in",
    "into",
    "key",
    "last",
    "latest",
    "month",
    "months",
    "new",
    "news",
    "of",
    "on",
    "or",
    "outlook",
    "recent",
    "show",
    "tell",
    "that",
    "the",
    "this",
    "today",
    "update",
    "updates",
    "week",
    "what",
    "with",
    "why",
    "yesterday",
}

TOPIC_ALIASES = {
    "elon": {
        "canonical": "Elon Musk",
        "title": "Elon Musk",
        "include_terms": ("elon", "musk", "tesla", "spacex", "x", "neuralink"),
        "exclude_terms": ("university", "college", "campus", "student", "athletics", "mentor"),
    },
    "musk": {
        "canonical": "Elon Musk",
        "title": "Elon Musk",
        "include_terms": ("elon", "musk", "tesla", "spacex", "x", "neuralink"),
        "exclude_terms": ("university", "college", "campus", "student", "athletics", "mentor"),
    },
    "trump": {
        "canonical": "Donald Trump",
        "title": "Donald Trump",
        "include_terms": ("donald", "trump", "president", "white house"),
        "exclude_terms": (),
    },
    "modi": {
        "canonical": "Narendra Modi",
        "title": "Narendra Modi",
        "include_terms": ("narendra", "modi", "india", "indian"),
        "exclude_terms": (),
    },
    "trudeau": {
        "canonical": "Justin Trudeau",
        "title": "Justin Trudeau",
        "include_terms": ("justin", "trudeau", "canada", "canadian"),
        "exclude_terms": (),
    },
    "carney": {
        "canonical": "Mark Carney",
        "title": "Mark Carney",
        "include_terms": ("mark", "carney", "canada", "canadian"),
        "exclude_terms": (),
    },
}

EXPLANATION_STYLE_GUIDANCE = {
    "concise": "Be tight, selective, and high-signal. Short paragraphs. No filler.",
    "feynman": "Explain clearly in plain language, as if teaching an intelligent beginner.",
    "soc": "Use a Socratic question-and-answer structure that guides the reader through the topic.",
}

DEPTH_PRESETS = {
    "low": {
        "query_limit": 2,
        "results_per_query": 2,
        "article_chars": 0,
        "newsletter_tokens": 520,
        "research_budget_seconds": 15,
    },
    "medium": {
        "query_limit": 4,
        "results_per_query": 3,
        "article_chars": 900,
        "newsletter_tokens": 680,
        "research_budget_seconds": 35,
    },
    "high": {
        "query_limit": 5,
        "results_per_query": 4,
        "article_chars": 1200,
        "newsletter_tokens": 820,
        "research_budget_seconds": 60,
    },
}

SLICE_MODEL_PATHS = {
    0.125: os.environ.get("NEWSLETTER_AGENT_MODEL_SLICE_12_5", DEFAULT_MODEL_PATH),
    0.25: os.environ.get("NEWSLETTER_AGENT_MODEL_SLICE_25", DEFAULT_MODEL_PATH),
    0.50: os.environ.get("NEWSLETTER_AGENT_MODEL_SLICE_50", DEFAULT_MODEL_PATH),
    0.75: os.environ.get("NEWSLETTER_AGENT_MODEL_SLICE_75", DEFAULT_MODEL_PATH),
    1.00: os.environ.get("NEWSLETTER_AGENT_MODEL_SLICE_100", DEFAULT_MODEL_PATH),
}

INSECURE_SSL_CONTEXT = ssl._create_unverified_context()

MODEL = None
TOKENIZER = None
MODEL_RUNTIME = None
DEVICE_CLASS_OVERRIDE = os.environ.get("NEWSLETTER_AGENT_DEVICE_CLASS", "").strip().lower() or None

initialize_database()


def main():
    parser = argparse.ArgumentParser(description="Chronicle local newsletter agent")
    parser.add_argument("--brief", help="Newsletter brief or topic")
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS)
    parser.add_argument("--depth", choices=("low", "medium", "high"), default="medium")
    parser.add_argument(
        "--device-class",
        choices=("macbook", "midrange_laptop", "gaming_laptop", "midrange_phone", "flagship_phone"),
    )
    parser.add_argument(
        "--explanation-style",
        choices=("concise", "feynman", "soc", "custom"),
        default="concise",
    )
    parser.add_argument("--style-instructions", default="")
    parser.add_argument("--queries", type=int)
    parser.add_argument("--results-per-query", type=int)
    parser.add_argument("--output-dir", default="output/newsletters")
    args = parser.parse_args()

    brief = clean_text(args.brief or input("Newsletter brief: "))
    if not brief:
        raise SystemExit("A newsletter brief is required.")

    explanation_style, custom_style = resolve_explanation_style(
        args.explanation_style,
        args.style_instructions,
    )
    settings = build_research_settings(args.depth, args.queries, args.results_per_query)
    initialize_model_runtime(args.device_class)
    result = run_newsletter_pipeline(
        brief=brief,
        days=args.days,
        depth=args.depth,
        explanation_style=explanation_style,
        custom_style_instructions=custom_style,
        settings=settings,
        output_dir=args.output_dir,
    )
    print(f"Run complete: {result['title']}")
    print(f"HTML: {result['output_files']['html_path']}")
    print(f"Markdown: {result['output_files']['markdown_path']}")


def detect_system_info():
    memory_total_gb, memory_available_gb = detect_memory_gb()
    system_name = platform.system()
    machine = platform.machine()
    hardware_model = ""
    chip = ""
    gpu_model = ""

    if system_name == "Darwin":
        hardware_model = read_sysctl_value("hw.model")
        chip = read_sysctl_value("machdep.cpu.brand_string") or read_sysctl_value("machdep.cpu.brand_string")
        if not chip and machine == "arm64":
            chip = "Apple Silicon"
        gpu_model = chip if chip.startswith("Apple") else chip

    return {
        "system_name": system_name,
        "platform": platform.platform(),
        "machine": machine,
        "hardware_model": hardware_model,
        "chip": chip,
        "gpu_model": gpu_model,
        "memory_total_gb": round(memory_total_gb, 2),
        "memory_available_gb": round(memory_available_gb, 2),
    }


def detect_memory_gb():
    gib = 1024 ** 3
    try:
        import psutil

        stats = psutil.virtual_memory()
        return stats.total / gib, stats.available / gib
    except Exception:
        pass

    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        total_pages = os.sysconf("SC_PHYS_PAGES")
        total = (page_size * total_pages) / gib
        return total, total
    except Exception:
        return 0.0, 0.0


def read_sysctl_value(key):
    try:
        completed = subprocess.run(
            ["sysctl", "-n", key],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return ""
    return clean_text(completed.stdout)


def choose_model_profile(system_info, device_class_override=None):
    device_class = (device_class_override or DEVICE_CLASS_OVERRIDE or infer_device_class(system_info)).strip().lower()
    slice_ratio = choose_slice_ratio(system_info, device_class)
    runtime_backend = choose_runtime_backend(system_info)
    model_path = normalize_model_reference(SLICE_MODEL_PATHS.get(slice_ratio, DEFAULT_MODEL_PATH))

    return {
        "device_class": device_class,
        "runtime_backend": runtime_backend,
        "slice_ratio": slice_ratio,
        "slice_label": format_slice_label(slice_ratio),
        "model_path": model_path,
        "draft_model_path": "",
        "lazy": True,
        "num_draft_tokens": 0,
    }


def infer_device_class(system_info):
    total = float(system_info.get("memory_total_gb", 0) or 0)
    available = float(system_info.get("memory_available_gb", 0) or 0)
    system_name = str(system_info.get("system_name", ""))
    machine = str(system_info.get("machine", "")).lower()

    if "iphone" in machine or "android" in machine:
        return "midrange_phone"
    if total >= 24 or available >= 18:
        return "gaming_laptop"
    if total >= 16:
        return "midrange_laptop"
    if system_name == "Darwin" and machine == "arm64":
        return "macbook"
    if total >= 8:
        return "midrange_laptop"
    return "midrange_phone"


def choose_slice_ratio(system_info, device_class):
    available = max(0.0, float(system_info.get("memory_available_gb", 0) or 0) - DEFAULT_RAM_RESERVE_GB)

    if device_class in {"midrange_phone", "flagship_phone"}:
        return 0.125 if available < 4 else 0.25
    if device_class == "macbook":
        if available < 3:
            return 0.125
        if available < 6:
            return 0.25
        return 0.50
    if device_class == "midrange_laptop":
        if available < 4:
            return 0.25
        if available < 8:
            return 0.50
        return 0.75
    if device_class == "gaming_laptop":
        if available < 6:
            return 0.50
        if available < 10:
            return 0.75
        return 1.00
    return 0.25


def format_slice_label(slice_ratio):
    percentage = slice_ratio * 100
    if float(percentage).is_integer():
        return f"{int(percentage)}%"
    return f"{percentage:.1f}%"


def choose_runtime_backend(system_info):
    system_name = str(system_info.get("system_name", ""))
    machine = str(system_info.get("machine", "")).lower()
    if system_name == "Darwin" and machine == "arm64":
        return "mlx"
    return "transformers"


def normalize_model_reference(model_path):
    value = str(model_path or "").strip()
    if not value:
        return ""

    path_candidate = Path(os.path.expanduser(value))
    if path_candidate.is_absolute() or value.startswith("."):
        return str(path_candidate.resolve())

    local_candidate = (Path(DEFAULT_MODEL_ROOT) / value).resolve()
    if local_candidate.exists():
        return str(local_candidate)

    return value


def is_local_model_reference(model_path):
    value = str(model_path or "").strip()
    if not value:
        return False
    if value.startswith("/") or value.startswith(".") or value.startswith("~"):
        return True
    return Path(value).exists()


def initialize_model_runtime(device_class_override=None):
    global MODEL
    global TOKENIZER
    global MODEL_RUNTIME

    system_info = detect_system_info()
    profile = choose_model_profile(system_info, device_class_override)
    model_path = normalize_model_reference(profile["model_path"])

    if (
        MODEL_RUNTIME
        and MODEL_RUNTIME.get("model_path") == model_path
        and MODEL_RUNTIME.get("runtime_backend") == profile["runtime_backend"]
    ):
        return MODEL_RUNTIME

    if not is_local_model_reference(model_path):
        raise SystemExit(f"Model reference must be local for Chronicle: {model_path}")
    if not Path(model_path).exists():
        raise SystemExit(f"Local model not found at {model_path}")

    if profile["runtime_backend"] == "mlx":
        MODEL_RUNTIME = initialize_mlx_runtime(model_path, profile)
        return MODEL_RUNTIME

    if profile["runtime_backend"] == "transformers":
        MODEL_RUNTIME = initialize_transformers_runtime(model_path, profile)
        return MODEL_RUNTIME

    raise SystemExit(f"Unsupported runtime backend: {profile['runtime_backend']}")


def initialize_mlx_runtime(model_path, profile):
    global MODEL
    global TOKENIZER

    try:
        from mlx_lm import generate, load
    except ModuleNotFoundError as exc:
        raise SystemExit(f"Missing dependency: {exc.name}") from exc

    MODEL, TOKENIZER = load(model_path)

    def mlx_generate_fn(model, tokenizer, prompt, max_tokens, draft_model=None, num_draft_tokens=0):
        kwargs = {
            "prompt": prompt,
            "max_tokens": max_tokens,
        }
        if draft_model is not None and num_draft_tokens > 0:
            kwargs["draft_model"] = draft_model
            kwargs["num_draft_tokens"] = num_draft_tokens
        return generate(model, tokenizer, **kwargs)

    return {
        **profile,
        "model_path": model_path,
        "generate_fn": mlx_generate_fn,
        "draft_model": None,
        "device": "mlx",
    }


def initialize_transformers_runtime(model_path, profile):
    global MODEL
    global TOKENIZER

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ModuleNotFoundError as exc:
        raise SystemExit(f"Missing dependency: {exc.name}") from exc

    TOKENIZER = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    MODEL = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype="auto",
    )

    if torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    MODEL = MODEL.to(device)
    MODEL.eval()

    if TOKENIZER.pad_token_id is None and TOKENIZER.eos_token_id is not None:
        TOKENIZER.pad_token = TOKENIZER.eos_token

    return {
        **profile,
        "model_path": model_path,
        "device": device,
    }


def describe_browser_model():
    for candidate in DEFAULT_BROWSER_MODEL_CANDIDATES:
        model_path = resolve_browser_model_location(candidate)
        if model_path is None or not model_path.exists():
            continue

        config = read_json_file(model_path / "config.json")
        model_type = clean_text(config.get("model_type", "")) or "unknown"
        architecture = clean_text(first_item(config.get("architectures")) or "")
        onnx_root = model_path / "onnx"
        preferred_precision, dtype_map = detect_browser_model_precision(onnx_root)
        supports_slicing = model_type == "gemma3n"
        model_id = derive_browser_model_id(candidate, model_path)

        return {
            "ready": True,
            "model_id": model_id,
            "model_path": str(model_path.resolve()),
            "model_type": model_type,
            "architecture": architecture,
            "supports_slicing": supports_slicing,
            "max_slices": 8 if supports_slicing else 1,
            "preferred_precision": preferred_precision,
            "dtype_map": dtype_map,
            "display_name": derive_browser_display_name(model_id, architecture, model_type),
        }

    fallback_id = DEFAULT_BROWSER_MODEL_CANDIDATES[0] if DEFAULT_BROWSER_MODEL_CANDIDATES else ""
    return {
        "ready": False,
        "model_id": fallback_id,
        "model_path": "",
        "model_type": "",
        "architecture": "",
        "supports_slicing": False,
        "max_slices": 1,
        "preferred_precision": "",
        "dtype_map": {},
        "display_name": derive_browser_display_name(fallback_id, "", ""),
    }


def resolve_browser_model_location(candidate):
    value = str(candidate or "").strip()
    if not value:
        return None

    raw_path = Path(os.path.expanduser(value))
    if raw_path.is_absolute() or value.startswith("."):
        return raw_path.resolve()

    return (Path(DEFAULT_MODEL_ROOT) / value).resolve()


def derive_browser_model_id(candidate, model_path):
    value = str(candidate or "").strip()
    if value and not value.startswith("/") and not value.startswith(".") and not value.startswith("~"):
        return value

    try:
        return model_path.resolve().relative_to(Path(DEFAULT_MODEL_ROOT).resolve()).as_posix()
    except Exception:
        return model_path.name


def derive_browser_display_name(model_id, architecture, model_type):
    lowered = f"{model_id} {architecture} {model_type}".lower()
    if "gemma-3n" in lowered or "gemma3n" in lowered:
        return "Gemma 3n adaptive"
    if "gemma-3" in lowered or "gemma3" in lowered:
        return "Gemma 3"
    return clean_text(architecture or model_id or "Local model")


def detect_browser_model_precision(onnx_root):
    if not onnx_root.exists():
        return "", {}

    precision_order = ("q4", "q4f16", "uint8", "fp16", "fp32")
    components = (
        "decoder_model_merged",
        "embed_tokens",
        "audio_encoder",
        "vision_encoder",
    )
    dtype_map = {}

    for component in components:
        precision = detect_onnx_component_precision(onnx_root, component, precision_order)
        if precision:
            dtype_map[component] = precision

    preferred_precision = dtype_map.get("decoder_model_merged") or next(iter(dtype_map.values()), "")
    return preferred_precision, dtype_map


def detect_onnx_component_precision(onnx_root, component_name, precision_order):
    for precision in precision_order:
        if precision == "fp32":
            filename = f"{component_name}.onnx"
        else:
            filename = f"{component_name}_{precision}.onnx"
        if (onnx_root / filename).exists():
            return precision
    return ""


def build_research_settings(depth, query_limit_override, results_per_query_override):
    settings = dict(DEPTH_PRESETS[depth])
    if query_limit_override is not None:
        settings["query_limit"] = max(1, min(int(query_limit_override), 8))
    if results_per_query_override is not None:
        settings["results_per_query"] = max(1, min(int(results_per_query_override), 8))
    return settings


def build_fallback_research_plan(brief, days, query_limit, depth):
    normalized_brief = clean_text(brief)
    topic_profile = resolve_topic_profile(normalized_brief)
    focus_phrase = topic_profile["search_focus"]
    query_candidates = [
        topic_profile["canonical"],
        focus_phrase,
        f"{focus_phrase} news" if focus_phrase else "",
        f"{focus_phrase} latest" if focus_phrase else "",
        f"{focus_phrase} this week" if focus_phrase and days <= 10 else "",
        f"{focus_phrase} last {days} days" if focus_phrase and days > 10 else "",
        f"{focus_phrase} analysis" if focus_phrase else "",
        normalized_brief,
        *build_search_query_variants(focus_phrase or normalized_brief),
    ]

    queries = []
    seen = set()
    for candidate in query_candidates:
        value = clean_text(candidate)
        if not value:
            continue
        lowered = value.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        queries.append(value)
        if len(queries) >= query_limit:
            break

    sections = [
        "What happened",
        "Why it matters",
        "Key developments",
        "What to watch next",
    ]

    return {
        "title": generate_fallback_title(normalized_brief),
        "audience": "General readers who want a useful weekly briefing",
        "tone": DEFAULT_WRITING_STYLE,
        "queries": queries or [normalized_brief],
        "sections": sections[: max(3, 4 if depth == "high" else 3)],
    }


def generate_fallback_title(brief):
    topic_profile = resolve_topic_profile(brief)
    focus_phrase = topic_profile["title"] or topic_profile["canonical"] or derive_search_focus_phrase(brief)
    words = [word for word in re.split(r"\s+", focus_phrase or brief) if word]
    compact = " ".join(words[:4]).strip()
    return compact.title() if compact else "Weekly Newsletter"


def derive_search_focus_phrase(text):
    topic_profile = resolve_topic_profile(text)
    if topic_profile["canonical"] and topic_profile["canonical"].lower() != clean_text(text).lower():
        return topic_profile["search_focus"]
    keywords = extract_search_keywords(text)
    if keywords:
        return " ".join(keywords[:4])
    stripped = re.sub(
        r"\b(what changed in|what changed|tell me about|show me|latest developments|key news|analysis and outlook)\b",
        " ",
        str(text or ""),
        flags=re.IGNORECASE,
    )
    return clean_text(stripped)


def extract_search_keywords(text):
    keywords = []
    seen = set()
    for word in re.findall(r"[A-Za-z0-9][A-Za-z0-9'.-]*", str(text or "").lower()):
        normalized = word.strip(".'-")
        if len(normalized) < 3 or normalized.isdigit():
            continue
        if normalized in SEARCH_NOISE_WORDS:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        keywords.append(normalized)
    return keywords


def simplify_search_query(query):
    simplified = re.sub(
        r"\b(latest developments?|key news|analysis and outlook|expert commentary|what changed this week|what changed in|what changed|latest|this week|last \d+ days?)\b",
        " ",
        str(query or ""),
        flags=re.IGNORECASE,
    )
    return clean_text(simplified)


def build_search_query_variants(query):
    original_query = clean_text(query)
    simplified_query = simplify_search_query(original_query)
    keywords = extract_search_keywords(simplified_query or original_query)
    focus_phrase = " ".join(keywords[:4]) if keywords else simplified_query or original_query
    last_keyword = keywords[-1] if keywords else ""

    variants = []
    seen = set()

    def add(candidate):
        value = clean_text(candidate)
        if not value:
            return
        lowered = value.lower()
        if lowered in seen:
            return
        seen.add(lowered)
        variants.append(value)

    add(original_query)
    add(simplified_query)
    if focus_phrase:
        add(f"{focus_phrase} news")
        add(f"{focus_phrase} latest")
        add(f"{focus_phrase} analysis")
        add(f"{focus_phrase} this week")
    if last_keyword:
        add(f"{last_keyword} latest news")

    return variants[:5]


def resolve_topic_profile(brief):
    normalized_brief = clean_text(brief)
    lowered = normalized_brief.lower()
    keywords = extract_search_keywords(normalized_brief)
    alias_key = lowered if lowered in TOPIC_ALIASES else (keywords[0] if len(keywords) == 1 and keywords[0] in TOPIC_ALIASES else "")
    alias = TOPIC_ALIASES.get(alias_key, {})

    canonical = clean_text(alias.get("canonical", "")) or normalized_brief
    title = clean_text(alias.get("title", "")) or canonical
    include_terms = tuple(alias.get("include_terms", ()) or tuple(keywords[:4]))
    exclude_terms = tuple(alias.get("exclude_terms", ()))

    return {
        "canonical": canonical,
        "title": title,
        "search_focus": canonical,
        "include_terms": include_terms,
        "exclude_terms": exclude_terms,
    }


def fetch_market_snapshot(brief):
    if not looks_like_crypto_brief(brief):
        return []

    url = (
        "https://api.coingecko.com/api/v3/coins/markets"
        "?vs_currency=usd&order=market_cap_desc&per_page=5&page=1"
        "&sparkline=false&price_change_percentage=7d"
    )

    try:
        payload = fetch_url(url)
        data = json.loads(payload)
    except Exception as exc:
        print(f"Structured market data fetch failed: {exc}")
        return []

    snapshot = []
    for item in data[:5]:
        snapshot.append(
            {
                "name": item.get("name", ""),
                "symbol": str(item.get("symbol", "")).upper(),
                "price_usd": item.get("current_price"),
                "market_cap_rank": item.get("market_cap_rank"),
                "change_24h_pct": item.get("price_change_percentage_24h"),
                "change_7d_pct": item.get("price_change_percentage_7d_in_currency"),
            }
        )
    return snapshot


def looks_like_crypto_brief(brief):
    lowered = str(brief or "").lower()
    keywords = (
        "crypto",
        "bitcoin",
        "ethereum",
        "blockchain",
        "defi",
        "token",
        "altcoin",
        "web3",
    )
    return any(keyword in lowered for keyword in keywords)


def search_web(query, max_results, deadline=None):
    if deadline and time.monotonic() >= deadline:
        print("  Search budget reached before Google search started.")
        return []

    query_variant = build_search_query_variants(query)[0]
    try:
        results = search_google_news_rss(query_variant, max_results)
    except Exception as exc:
        print(f"  Google search failed: {exc}")
        return []

    results = prioritize_sources(results)
    print(f"  Google search: {len(results)} result(s)")
    return results[:max_results]


def search_google_news_rss(query, max_results):
    rss_query = urllib.parse.quote(f"{query} when:7d")
    url = f"https://news.google.com/rss/search?q={rss_query}&hl=en-US&gl=US&ceid=US:en"
    xml_text = fetch_url(url)

    root = ET.fromstring(xml_text)
    results = []
    seen = set()

    for item in root.findall(".//item"):
        title = clean_source_title(item.findtext("title", ""))
        url = clean_text(item.findtext("link", ""))
        description = item.findtext("description", "") or ""
        snippet = clean_text(strip_tags(description).replace("\xa0", " "))
        if normalize_source_phrase(snippet) == normalize_source_phrase(title):
            snippet = ""

        if not title or not url:
            continue

        key = source_identity({"title": title, "url": url})
        if key in seen:
            continue
        seen.add(key)
        results.append(
            {
                "title": title,
                "url": url,
                "snippet": snippet,
            }
        )
        if len(results) >= max_results:
            break

    return results


def prioritize_sources(results):
    seen = set()
    prioritized = []
    for result in sorted(
        results,
        key=lambda item: (
            1 if is_indirect_source_url(item.get("url", "")) else 0,
            -len(str(item.get("snippet", "") or "")),
        ),
    ):
        key = source_identity(result)
        if key in seen:
            continue
        seen.add(key)
        prioritized.append(result)
    return prioritized


def source_identity(source):
    title_key = normalize_source_phrase(clean_source_title(source.get("title", "")))
    url = str(source.get("url", "") or "")
    if "news.google.com/" in url.lower():
        return f"title:{title_key}"
    parsed = urllib.parse.urlparse(url)
    return f"url:{parsed.netloc}{parsed.path}" if parsed.netloc else f"title:{title_key}"


def is_indirect_source_url(url):
    return "news.google.com/" in str(url or "").lower()


def fetch_url(url):
    body, _ = fetch_response(url)
    return body


def fetch_response(url):
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Cache-Control": "no-cache",
        },
    )

    with urllib.request.urlopen(
        request,
        timeout=REQUEST_TIMEOUT_SECONDS,
        context=INSECURE_SSL_CONTEXT,
    ) as response:
        return response.read().decode("utf-8", errors="ignore"), response.geturl()


def fetch_article_text(url, max_article_chars):
    if not max_article_chars:
        return ""

    try:
        raw_html, final_url = fetch_response(url)
    except Exception as exc:
        print(f"  Article fetch failed for {url}: {exc}")
        return ""

    text = extract_article_text(raw_html)
    if looks_like_placeholder_article_text(text):
        if final_url and final_url != url:
            try:
                raw_html, _ = fetch_response(final_url)
                text = extract_article_text(raw_html)
            except Exception:
                return ""

    if looks_like_placeholder_article_text(text):
        return ""

    return compact_evidence_text(text, max_article_chars)


def extract_article_text(raw_html):
    paragraph_matches = re.findall(r"(?is)<p\b[^>]*>(.*?)</p>", raw_html or "")
    paragraphs = []
    for match in paragraph_matches:
        cleaned = clean_text(strip_tags(match))
        if len(cleaned) >= 40:
            paragraphs.append(cleaned)

    if paragraphs:
        return " ".join(paragraphs[:8])

    text = re.sub(r"(?is)<script.*?>.*?</script>", " ", raw_html or "")
    text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
    text = re.sub(r"(?is)<noscript.*?>.*?</noscript>", " ", text)
    text = re.sub(r"(?is)<svg.*?>.*?</svg>", " ", text)
    text = strip_tags(text)
    return clean_text(text)


def build_source_text(result, article_text):
    title = clean_source_title(result.get("title", ""))
    snippet = clean_text(result.get("snippet", ""))
    article_evidence = compact_evidence_text(article_text, 360)
    if article_evidence:
        return article_evidence

    snippet_evidence = compact_evidence_text(snippet, 220)
    if snippet_evidence and normalize_source_phrase(snippet_evidence) != normalize_source_phrase(title):
        return snippet_evidence

    return title


def looks_like_placeholder_article_text(text):
    normalized = clean_text(text).lower()
    if not normalized:
        return True
    if normalized in {"google news", "bing", "duckduckgo"}:
        return True
    if len(normalized) < 120:
        return True
    repeated = normalized.count("cookie")
    return repeated >= 3


def build_mode_system_prompt(explanation_style, custom_style_instructions):
    base = (
        "You are Chronicle, a newsletter writer. "
        "Think silently, form one clear argument from the research notes, and write clean markdown. "
        "Do not copy source wording. Do not use raw URLs in the body. Do not repeat sentences."
    )

    if explanation_style == "feynman":
        return (
            f"{base}\n\n"
            "Mode: Feynman. Explain clearly in plain language, with simple causality and concrete examples."
        )

    if explanation_style == "soc":
        return (
            f"{base}\n\n"
            "Mode: Socratic. Structure sections as sharp questions followed by clear answers grounded in evidence."
        )

    if explanation_style == "custom" and custom_style_instructions:
        return (
            f"{base}\n\n"
            f"Mode: Custom. Follow these instructions exactly: {clean_text(custom_style_instructions)[:220]}"
        )

    return (
        f"{base}\n\n"
        "Mode: Concise. Be selective, high-signal, and analytical."
    )


def build_source_block(collected_sources):
    if not collected_sources:
        return "No web sources were collected."

    blocks = []
    for index, source in enumerate(collected_sources[:4], start=1):
        title = clean_source_title(source.get("title", ""))
        evidence = build_source_text(source, source.get("article_text", ""))
        evidence = compact_evidence_text(evidence, 240) or title
        blocks.append(f"[{index}] {title}: {evidence}")
    return "\n\n".join(blocks)


def compose_newsletter(
    brief,
    plan,
    collected_sources,
    days,
    market_snapshot,
    depth,
    explanation_style,
    custom_style_instructions,
    newsletter_tokens,
):
    system_prompt = build_mode_system_prompt(explanation_style, custom_style_instructions)
    source_block = build_source_block(collected_sources)
    market_block = json.dumps(market_snapshot, ensure_ascii=True) if market_snapshot else "None."
    sections_block = " | ".join(plan["sections"][:3])

    prompt = f"""<start_of_turn>user
{system_prompt}

Topic: {clean_text(brief)[:180]}
Title: {clean_text(plan["title"])[:100]}
Audience: {clean_text(plan["audience"])[:80]}
Coverage window: last {days} days
Depth: {depth}
Section plan: {sections_block}

Research notes:
{source_block}

Market data:
{market_block}

Instructions:
- First identify the strongest through-line across the research notes
- Then write 550 to 750 words
- Start with one H1 title and a strong opening paragraph
- Use H2 headings for the main sections
- Synthesize the evidence into a coherent argument, not a list of summaries
- Explain why the developments matter, not just what happened
- Cite sources inline as [1], [2] and use [M1] for market data if needed
- Do not include a Sources section; Chronicle will add the reference list after writing
- Write only the markdown newsletter
<end_of_turn>
<start_of_turn>model
"""

    return generate_text_from_prompt(prompt, newsletter_tokens).strip()


def render_deterministic_newsletter(plan, brief, collected_sources, explanation_style, market_snapshot):
    lead_sources = collected_sources[:3]
    title = plan["title"]
    sections = plan["sections"][:3]

    if lead_sources:
        lead_titles = [clean_source_title(source["title"]) for source in lead_sources[:2]]
        opening = (
            f"The clearest signal around {brief} is not any one isolated headline, "
            f"but the way {lead_titles[0]}{' and ' + lead_titles[1] if len(lead_titles) > 1 else ''} "
            "are pulling the story into a single frame."
        )
    else:
        opening = (
            f"External reporting on {brief} was thin in this run, so this issue stays close to the verified brief "
            "and avoids overclaiming."
        )

    body = [f"# {title}", "", opening, ""]
    for index, section in enumerate(sections):
        body.append(f"## {section}")
        body.append("")
        paired_sources = lead_sources[index:index + 2] or lead_sources[:1]
        if paired_sources:
            lines = []
            for source_index, source in enumerate(paired_sources, start=1):
                evidence = build_source_text(source, source.get("article_text", ""))
                citation = collected_sources.index(source) + 1
                lines.append(f"{evidence} [{citation}]")
            body.append(" ".join(lines))
        else:
            body.append(f"This section remains provisional because Chronicle did not collect enough usable evidence for {brief}.")
        body.append("")

    if market_snapshot:
        top_asset = market_snapshot[0]
        body.append("## Market context")
        body.append("")
        body.append(
            f"{top_asset['name']} ({top_asset['symbol']}) is trading near {top_asset['price_usd']} USD, "
            f"with 24h change of {top_asset['change_24h_pct']} and 7d change of {top_asset['change_7d_pct']} [M1]."
        )
        body.append("")

    if explanation_style == "soc":
        body.insert(4, "What is the real question here? It is whether the visible headlines add up to one durable direction or just a noisy burst.")
        body.insert(5, "")
    elif explanation_style == "feynman":
        body.insert(4, "Think of the reporting window like a map: each headline is a pin, and the pattern matters more than any single pin.")
        body.insert(5, "")

    return "\n".join(body).strip()


def finalize_newsletter_markdown(markdown, plan, collected_sources, market_snapshot):
    text = normalize_newsletter_markdown(markdown)
    if not text:
        text = f"# {plan['title']}\n"
    if not re.search(r"(?m)^#\s+", text):
        text = f"# {plan['title']}\n\n{text}"
    if not re.search(r"(?im)^##\s+sources\b", text):
        text = f"{text.rstrip()}\n\n{build_sources_section(collected_sources, market_snapshot)}"
    return text.strip()


def build_sources_section(collected_sources, market_snapshot):
    if collected_sources:
        lines = []
        for index, source in enumerate(collected_sources, start=1):
            title = clean_source_title(source.get("title", ""))
            url = clean_text(source.get("url", ""))
            lines.append(f"[{index}]: {title} - {url}")
    else:
        lines = ["No external sources were successfully collected."]

    if market_snapshot:
        lines.append("[M1]: CoinGecko Markets API - https://www.coingecko.com/")

    return "## Sources\n" + "\n".join(lines)


def run_newsletter_pipeline(
    brief,
    days,
    depth,
    explanation_style,
    custom_style_instructions,
    settings,
    output_dir,
):
    plan = build_fallback_research_plan(brief, days, settings["query_limit"], depth)
    run_id = save_run(plan, brief, depth, explanation_style, custom_style_instructions)
    market_snapshot = fetch_market_snapshot(brief)

    research_budget_seconds = min(
        int(settings.get("research_budget_seconds", 45)),
        max(30, MAX_NEWSLETTER_RUNTIME_SECONDS - 120),
    )
    research_deadline = time.monotonic() + research_budget_seconds
    article_fetches = 0
    max_article_fetches = max(2, min(6, settings["query_limit"] + 1))
    collected_sources = []
    seen_source_keys = set()

    print("Planning complete.")
    print(f"Title: {plan['title']}")
    print(f"Research depth: {depth}")
    print(f"Explanation style: {explanation_style}")
    print(f"Research budget: {research_budget_seconds}s")
    if market_snapshot:
        print(f"Structured market data collected for {len(market_snapshot)} assets.")

    for query in plan["queries"]:
        if time.monotonic() >= research_deadline:
            print("Research budget reached. Drafting with the material already collected.")
            break

        print(f"Searching: {query}")
        results = search_web(query, settings["results_per_query"], deadline=research_deadline)
        print(f"  Search results: {len(results)}")

        for rank_index, result in enumerate(results, start=1):
            if time.monotonic() >= research_deadline:
                print("  Research budget reached while processing results.")
                break

            key = source_identity(result)
            if key in seen_source_keys:
                continue

            article_text = ""
            should_fetch_article = (
                settings.get("article_chars", 0) > 0
                and article_fetches < max_article_fetches
                and rank_index <= 2
            )
            if should_fetch_article:
                print("  Fetching article text for richer evidence…")
                article_text = fetch_article_text(result["url"], settings["article_chars"])
                if article_text:
                    article_fetches += 1
                    print(f"  Article text captured: {len(article_text)} chars")
                else:
                    print("  Article fetch did not yield usable text. Falling back to snippet evidence.")

            source_text = build_source_text(result, article_text)
            if not source_text:
                print("  Skipped source: no article text or search snippet available.")
                continue

            seen_source_keys.add(key)
            record = {
                "query": query,
                "rank_index": rank_index,
                "title": clean_source_title(result["title"]),
                "url": result["url"],
                "snippet": clean_text(result.get("snippet", "")),
                "article_text": article_text,
                "source_summary": source_text,
                "source_text": source_text,
                "relevance_score": rank_source(result, article_text, rank_index),
            }
            save_source(run_id, record)
            collected_sources.append(record)
            print(f"  Source ready: {record['title']}")

    try:
        newsletter_markdown = compose_newsletter(
            brief,
            plan,
            collected_sources,
            days,
            market_snapshot,
            depth,
            explanation_style,
            custom_style_instructions,
            settings["newsletter_tokens"],
        )
    except Exception as exc:
        print(f"Model drafting failed: {exc}")
        newsletter_markdown = render_deterministic_newsletter(
            plan,
            brief,
            collected_sources,
            explanation_style,
            market_snapshot,
        )

    newsletter_markdown = finalize_newsletter_markdown(
        newsletter_markdown,
        plan,
        collected_sources,
        market_snapshot,
    )
    output_files = write_newsletter_files(output_dir, plan["title"], newsletter_markdown)
    update_run_output_path(run_id, output_files["html_path"])

    print("Newsletter generated.")
    print(f"Saved editable HTML to: {output_files['html_path']}")
    print(f"Saved markdown source to: {output_files['markdown_path']}")

    return {
        "run_id": run_id,
        "title": plan["title"],
        "output_files": output_files,
    }


def rank_source(result, article_text, rank_index):
    score = 10.0 - (rank_index * 0.5)
    if article_text:
        score += min(3.0, len(article_text) / 400.0)
    if result.get("snippet"):
        score += 1.0
    if is_indirect_source_url(result.get("url", "")):
        score -= 0.5
    return round(score, 2)


def generate_text_from_prompt(prompt, max_tokens):
    runtime = initialize_model_runtime()

    if runtime["runtime_backend"] == "mlx":
        return runtime["generate_fn"](
            MODEL,
            TOKENIZER,
            prompt=prompt,
            max_tokens=max_tokens,
            draft_model=runtime.get("draft_model"),
            num_draft_tokens=runtime.get("num_draft_tokens", 0),
        ).strip()

    if runtime["runtime_backend"] == "transformers":
        return generate_with_transformers(prompt, max_tokens, runtime)

    raise RuntimeError(f"Unsupported runtime backend: {runtime['runtime_backend']}")


def generate_with_transformers(prompt, max_tokens, runtime):
    import torch

    encoded = TOKENIZER(prompt, return_tensors="pt")
    device = runtime.get("device", "cpu")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    input_length = encoded["input_ids"].shape[-1]
    pad_token_id = TOKENIZER.pad_token_id or TOKENIZER.eos_token_id

    with torch.no_grad():
        output = MODEL.generate(
            **encoded,
            max_new_tokens=max_tokens,
            do_sample=False,
            repetition_penalty=1.12,
            pad_token_id=pad_token_id,
        )

    generated = output[0][input_length:]
    return TOKENIZER.decode(generated, skip_special_tokens=True).strip()


def save_run(plan, brief, depth, explanation_style, custom_style_instructions):
    initialize_database()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO newsletter_runs (
            brief,
            depth,
            explanation_style,
            custom_style_instructions,
            audience,
            tone,
            title,
            queries_json,
            sections_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            brief,
            depth,
            explanation_style,
            custom_style_instructions,
            plan["audience"],
            plan["tone"],
            plan["title"],
            json.dumps(plan["queries"]),
            json.dumps(plan["sections"]),
        ),
    )
    run_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return run_id


def save_source(run_id, source_record):
    initialize_database()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT OR REPLACE INTO source_items (
            run_id,
            query_text,
            rank_index,
            title,
            url,
            snippet,
            article_text,
            source_summary,
            relevance_score
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            clean_text(source_record.get("query", "")),
            int(source_record.get("rank_index", 0) or 0),
            clean_text(source_record.get("title", "")) or "Untitled source",
            clean_text(source_record.get("url", "")),
            clean_text(source_record.get("snippet", "")),
            clean_text(source_record.get("article_text", "")),
            clean_text(source_record.get("source_summary", "")),
            float(source_record.get("relevance_score", 0.0) or 0.0),
        ),
    )
    conn.commit()
    conn.close()


def update_run_output_path(run_id, output_path):
    initialize_database()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE newsletter_runs SET output_path = ? WHERE id = ?",
        (str(output_path or ""), run_id),
    )
    conn.commit()
    conn.close()


def write_newsletter_files(output_dir, title, markdown):
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now()
    timestamp = created_at.strftime("%Y%m%d_%H%M%S")
    slug = slugify(title)

    normalized_markdown = normalize_newsletter_markdown(markdown)
    markdown_path = output_root / f"{timestamp}_{slug}.md"
    html_path = output_root / f"{timestamp}_{slug}.html"

    markdown_path.write_text(normalized_markdown + "\n", encoding="utf-8")
    html_path.write_text(
        render_editable_newsletter_html(title, normalized_markdown, created_at),
        encoding="utf-8",
    )

    return {
        "markdown_path": str(markdown_path.resolve()),
        "html_path": str(html_path.resolve()),
    }


def normalize_newsletter_markdown(markdown):
    text = strip_code_fences(str(markdown or "")).replace("\r\n", "\n").strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def strip_code_fences(text):
    stripped = str(text or "").strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def render_editable_newsletter_html(fallback_title, markdown, created_at):
    title = extract_title_from_markdown(markdown, fallback_title)
    body_html = markdown_to_html(markdown)
    storage_key = f"chronicle::{slugify(title)}::{created_at.strftime('%Y%m%d%H%M%S')}"
    description = html.escape(extract_plain_text(markdown)[:160], quote=True)
    safe_title = html.escape(title)
    edition_label = created_at.strftime("%B %d, %Y")
    download_name = f"{slugify(title)}-editable.html"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{safe_title}</title>
  <meta name="description" content="{description}">
  <style>
    :root {{
      --bg: #0c1118;
      --panel: rgba(15, 22, 32, 0.92);
      --panel-soft: rgba(23, 32, 45, 0.88);
      --ink: #edf3fb;
      --muted: #8ca0b9;
      --line: rgba(151, 173, 202, 0.16);
      --accent: #d07b4a;
      --accent-strong: #ef9b65;
      --shadow: 0 24px 80px rgba(0, 0, 0, 0.35);
      --display: "Iowan Old Style", "Palatino Linotype", Georgia, serif;
      --ui: "Avenir Next", "Segoe UI", sans-serif;
      --mono: "SFMono-Regular", Menlo, monospace;
    }}

    * {{
      box-sizing: border-box;
    }}

    body {{
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      font-family: var(--ui);
      background:
        radial-gradient(circle at top left, rgba(208, 123, 74, 0.22), transparent 28%),
        radial-gradient(circle at 85% 10%, rgba(98, 132, 255, 0.16), transparent 22%),
        linear-gradient(180deg, #081017 0%, #0d1520 50%, #111b27 100%);
    }}

    .shell {{
      width: min(1200px, calc(100vw - 28px));
      margin: 0 auto;
      padding: 22px 0 48px;
    }}

    .toolbar {{
      position: sticky;
      top: 18px;
      z-index: 20;
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 14px 18px;
      border: 1px solid var(--line);
      border-radius: 20px;
      background: rgba(7, 12, 18, 0.8);
      backdrop-filter: blur(18px);
      box-shadow: var(--shadow);
    }}

    .toolbar-copy {{
      display: flex;
      flex-direction: column;
      gap: 4px;
    }}

    .toolbar-kicker {{
      font-size: 0.72rem;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      color: var(--muted);
    }}

    .toolbar-title {{
      font-size: 1rem;
      font-weight: 600;
    }}

    .toolbar-actions {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}

    button {{
      border: 0;
      border-radius: 999px;
      padding: 10px 14px;
      font: inherit;
      cursor: pointer;
      color: var(--ink);
      background: rgba(255, 255, 255, 0.08);
      transition: transform 160ms ease, background 160ms ease;
    }}

    button:hover {{
      transform: translateY(-1px);
      background: rgba(255, 255, 255, 0.14);
    }}

    button.primary {{
      background: linear-gradient(135deg, var(--accent), var(--accent-strong));
      color: #1b120c;
      font-weight: 700;
    }}

    .card {{
      margin-top: 22px;
      border: 1px solid var(--line);
      border-radius: 28px;
      background: linear-gradient(180deg, var(--panel), var(--panel-soft));
      box-shadow: var(--shadow);
      overflow: hidden;
    }}

    .meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      padding: 22px 26px 0;
      color: var(--muted);
      font-size: 0.86rem;
      letter-spacing: 0.06em;
      text-transform: uppercase;
    }}

    .editor {{
      min-height: 72vh;
      padding: 26px;
      outline: none;
      line-height: 1.72;
      font-size: 1.05rem;
    }}

    .editor h1,
    .editor h2,
    .editor h3 {{
      font-family: var(--display);
      line-height: 1.04;
      margin: 0 0 16px;
      letter-spacing: -0.03em;
    }}

    .editor h1 {{
      font-size: clamp(2.8rem, 7vw, 5rem);
      margin-top: 8px;
    }}

    .editor h2 {{
      font-size: clamp(1.5rem, 3vw, 2.25rem);
      margin-top: 38px;
    }}

    .editor p {{
      margin: 0 0 16px;
      color: rgba(237, 243, 251, 0.92);
    }}

    .editor ul {{
      margin: 0 0 18px 22px;
      padding: 0;
    }}

    .editor li {{
      margin: 0 0 10px;
    }}

    .editor code {{
      font-family: var(--mono);
      background: rgba(255, 255, 255, 0.08);
      padding: 2px 6px;
      border-radius: 6px;
    }}

    .foot {{
      padding: 0 26px 24px;
      color: var(--muted);
      font-size: 0.9rem;
    }}

    @media (max-width: 720px) {{
      .shell {{
        width: min(100vw - 16px, 1200px);
        padding-top: 12px;
      }}

      .editor {{
        padding: 20px;
      }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <div class="toolbar">
      <div class="toolbar-copy">
        <div class="toolbar-kicker">Chronicle HTML editor</div>
        <div class="toolbar-title">{safe_title}</div>
      </div>
      <div class="toolbar-actions">
        <button type="button" id="restore-button">Restore saved draft</button>
        <button type="button" id="download-button" class="primary">Download HTML</button>
      </div>
    </div>

    <div class="card">
      <div class="meta">
        <span>{html.escape(edition_label)}</span>
        <span>Editable issue</span>
      </div>
      <article id="editor" class="editor" contenteditable="true" spellcheck="true">{body_html}</article>
      <div class="foot">Changes are saved to this browser automatically while the page stays open.</div>
    </div>
  </div>

  <script>
    const storageKey = {json.dumps(storage_key)};
    const editor = document.getElementById("editor");
    const restoreButton = document.getElementById("restore-button");
    const downloadButton = document.getElementById("download-button");

    const savedHtml = window.localStorage.getItem(storageKey);
    if (savedHtml) {{
      editor.innerHTML = savedHtml;
    }}

    editor.addEventListener("input", () => {{
      window.localStorage.setItem(storageKey, editor.innerHTML);
    }});

    restoreButton.addEventListener("click", () => {{
      const cached = window.localStorage.getItem(storageKey);
      if (cached) {{
        editor.innerHTML = cached;
      }}
    }});

    downloadButton.addEventListener("click", () => {{
      const htmlText = document.documentElement.outerHTML;
      const blob = new Blob([htmlText], {{ type: "text/html;charset=utf-8" }});
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = {json.dumps(download_name)};
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
    }});
  </script>
</body>
</html>
"""


def markdown_to_html(markdown):
    lines = normalize_newsletter_markdown(markdown).splitlines()
    html_lines = []
    in_list = False

    def close_list():
        nonlocal in_list
        if in_list:
            html_lines.append("</ul>")
            in_list = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            close_list()
            continue

        if stripped.startswith("# "):
            close_list()
            html_lines.append(f"<h1>{format_inline_markdown(stripped[2:].strip())}</h1>")
            continue

        if stripped.startswith("## "):
            close_list()
            html_lines.append(f"<h2>{format_inline_markdown(stripped[3:].strip())}</h2>")
            continue

        if stripped.startswith("### "):
            close_list()
            html_lines.append(f"<h3>{format_inline_markdown(stripped[4:].strip())}</h3>")
            continue

        if stripped.startswith("- "):
            if not in_list:
                html_lines.append("<ul>")
                in_list = True
            html_lines.append(f"<li>{format_inline_markdown(stripped[2:].strip())}</li>")
            continue

        close_list()
        html_lines.append(f"<p>{format_inline_markdown(stripped)}</p>")

    close_list()
    return "\n".join(html_lines)


def format_inline_markdown(text):
    value = html.escape(str(text or ""))
    value = re.sub(r"`([^`]+)`", r"<code>\1</code>", value)
    value = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", value)
    value = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", value)
    return value


def extract_title_from_markdown(markdown, fallback_title):
    match = re.search(r"(?m)^#\s+(.+)$", normalize_newsletter_markdown(markdown))
    return clean_text(match.group(1)) if match else (clean_text(fallback_title) or "Newsletter")


def extract_plain_text(value):
    text = normalize_newsletter_markdown(value)
    text = re.sub(r"(?m)^#{1,6}\s+", "", text)
    text = text.replace("\n", " ")
    return clean_text(text)


def resolve_explanation_style(explanation_style, style_instructions):
    style = clean_text(explanation_style or "concise").lower()
    if style not in {"concise", "feynman", "soc", "custom"}:
        style = "concise"

    custom_instructions = clean_text(style_instructions or "")
    if style == "custom" and not custom_instructions:
        raise SystemExit("Custom style instructions are required when explanation style is custom.")

    return style, custom_instructions


def clean_source_title(title):
    value = clean_text(title)
    return re.sub(
        r"\s+-\s+(Reuters|AP News|Associated Press|MSN|AOL\.com|Yahoo!?\s*News|DW\.com|The New York Times|Al Jazeera|National Post|Toronto Star|Britannica|reuters\.com)$",
        "",
        value,
        flags=re.IGNORECASE,
    ).strip()


def normalize_source_phrase(value):
    normalized = re.sub(
        r"\b(reuters|ap news|associated press|msn|aol\.com|yahoo!?\s*news|dw\.com|the new york times|al jazeera|national post|toronto star|britannica|asia news network|the times of india)\b",
        " ",
        str(value or "").lower(),
    )
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return clean_text(normalized)


def compact_evidence_text(text, max_chars):
    value = clean_text(text or "")
    if not value or max_chars <= 0:
        return ""

    value = re.sub(r"\b(title|url|evidence)\s*:\s*", " ", value, flags=re.IGNORECASE)
    value = re.sub(r"https?://\S+", " ", value, flags=re.IGNORECASE)
    value = clean_text(value)
    if not value:
        return ""

    sentences = re.split(r"(?<=[.!?])\s+", value)
    selected = []
    seen = set()
    total_length = 0

    for sentence in sentences:
        cleaned_sentence = clean_text(sentence)
        if len(cleaned_sentence) < 20:
            continue
        key = re.sub(r"[^a-z0-9]+", " ", cleaned_sentence.lower()).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        selected.append(cleaned_sentence)
        total_length += len(cleaned_sentence) + 1
        if total_length >= max_chars:
            break

    if not selected:
        fallback = value[:max_chars].rsplit(" ", 1)[0].strip()
        return fallback or value[:max_chars].strip()

    compact = clean_text(" ".join(selected))
    if len(compact) <= max_chars:
        return compact

    trimmed = compact[:max_chars].rsplit(" ", 1)[0].strip()
    return trimmed or compact[:max_chars].strip()


def strip_tags(value):
    without_tags = re.sub(r"(?s)<[^>]+>", " ", str(value or ""))
    return html.unescape(clean_text(without_tags))


def read_json_file(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def first_item(value):
    if isinstance(value, list) and value:
        return value[0]
    return ""


def slugify(value):
    slug = re.sub(r"[^a-z0-9]+", "-", str(value or "").lower()).strip("-")
    return slug or "newsletter"


def clean_text(value):
    text = str(value or "")
    text = html.unescape(text)
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


if __name__ == "__main__":
    main()
