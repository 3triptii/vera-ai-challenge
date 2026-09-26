import hashlib
import json
import os
import re
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from google import genai

load_dotenv()

# ============================================================
# CONFIG
# ============================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found in .env")

client = genai.Client(api_key=GEMINI_API_KEY)

MODEL = "gemini-3.5-flash-lite"

app = FastAPI(title="Vera AI")


# ============================================================
# IN-MEMORY CONTEXT STORE
# ============================================================

categories: Dict[str, Dict[str, Any]] = {}
merchants: Dict[str, Dict[str, Any]] = {}
customers: Dict[str, Dict[str, Any]] = {}
triggers: Dict[str, Dict[str, Any]] = {}

conversations: Dict[str, list] = {}

# Exact-input cache: repeated identical compositions reuse the same generated
# message, improving replay determinism and reducing unnecessary API calls.
CACHE_FILE = "vera_message_cache.json"
message_cache: Dict[str, str] = {}

try:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            loaded = json.load(f)
            if isinstance(loaded, dict):
                message_cache = {str(k): str(v) for k, v in loaded.items()}
except Exception:
    message_cache = {}


def _cache_key(value: Any) -> str:
    canonical = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _cache_get(key: str) -> Optional[str]:
    return message_cache.get(key)


def _cache_put(key: str, value: str) -> None:
    message_cache[key] = value
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(message_cache, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# ============================================================
# REQUEST MODELS
# ============================================================

class ContextRequest(BaseModel):
    scope: str
    context_id: str
    version: int = 1
    payload: Dict[str, Any]
    delivered_at: Optional[str] = None


class TickRequest(BaseModel):
    available_triggers: list[str] = []
    now: Optional[str] = None


class ReplyRequest(BaseModel):
    conversation_id: str
    merchant_id: str
    customer_id: Optional[str] = None
    from_role: str = "merchant"
    message: str
    received_at: Optional[str] = None
    turn_number: int = 1


# ============================================================
# BASIC HELPERS & RESOLVERS
# ============================================================

def store_context(
    scope: str,
    context_id: str,
    version: int,
    payload: Dict[str, Any],
):
    target = {
        "category": categories,
        "merchant": merchants,
        "customer": customers,
        "trigger": triggers,
    }.get(scope)

    if target is None:
        return False

    existing = target.get(context_id)

    # Newer versions replace older versions.
    if existing is None or version >= existing.get("_version", 0):
        payload = dict(payload)
        payload["_version"] = version
        target[context_id] = payload

    return True


def get_category_for_merchant(merchant: Dict[str, Any]) -> Dict[str, Any]:
    category_slug = merchant.get("category_slug")

    if category_slug and category_slug in categories:
        return categories[category_slug]

    mid = merchant.get("merchant_id", "")
    for slug in categories:
        if slug in mid:
            return categories[slug]

    if categories:
        return next(iter(categories.values()))

    return {}


def get_customer(customer_id: Optional[str]) -> Optional[Dict[str, Any]]:
    if not customer_id:
        return None

    return customers.get(customer_id)


def resolve_salutation(merchant: Dict[str, Any], category: Dict[str, Any]) -> str:
    identity = merchant.get("identity", {})
    owner_name = identity.get("owner_first_name", "").strip()
    slug = category.get("slug", "")

    if slug == "dentists":
        return f"Dr. {owner_name}" if owner_name else "Dr."
    if slug in ("salons", "restaurants", "pharmacies"):
        return f"{owner_name} ji" if owner_name else "Partner"
    return owner_name if owner_name else "Partner"


def resolve_digest_item(trigger: Dict[str, Any], category: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    payload = trigger.get("payload", {})
    target_id = (
        payload.get("top_item_id")
        or payload.get("digest_item_id")
        or payload.get("alert_id")
        or payload.get("item_id")
    )
    if isinstance(payload.get("top_item"), dict):
        return payload["top_item"]

    if target_id:
        for item in category.get("digest", []):
            if item.get("id") == target_id:
                return item

    return None


def resolve_customer_name(trigger: Dict[str, Any], customer: Optional[Dict[str, Any]]) -> str:
    if customer:
        name = customer.get("identity", {}).get("name")
        if name:
            return name
    cid = trigger.get("customer_id") or ""
    if cid:
        parts = cid.split("_")
        if len(parts) >= 3 and parts[0] == "c":
            return parts[2].capitalize()
    return "your customer"


def get_trigger_strategy(
    kind: str,
    scope: str,
    customer_name: str,
    salutation: str,
) -> str:

    if kind == "active_planning_intent":
        return (
            f"ACTIVE PLANNING STRATEGY (merchant-facing):\n"
            f"- Treat the merchant's latest message as active planning intent, not a cold lead.\n"
            f"- Reflect the exact business idea or offer described in the trigger payload.\n"
            f"- Help turn the idea into one concrete next step Vera can prepare or execute.\n"
            f"- Use merchant-confirmed pricing/details only when supplied.\n"
            f"- Do not invent demand, expected orders, or projected results.\n"
            f"- End with one clear binary CTA."
        )

    if kind in ("research_digest", "category_research_digest_release"):
        return (
            f"RESEARCH DIGEST STRATEGY (merchant-facing):\n"
            f"- Lead with the exact digest title/source and the strongest finding supplied in context.\n"
            f"- Include at least two concrete evidence points when available (for example sample size, percentage, date, segment, or finding).\n"
            f"- Explain why the finding is relevant to this merchant/category using only supported context.\n"
            f"- Never invent local feedback, customer counts, clinical outcomes, or causal claims.\n"
            f"- Offer one practical asset or action Vera can prepare for the merchant.\n"
            f"- End with one binary CTA."
        )

    if kind in ("regulation_change", "compliance"):
        return (
            f"REGULATION / COMPLIANCE STRATEGY (merchant-facing):\n"
            f"- State the governing body, requirement, threshold, and deadline only when explicitly supplied.\n"
            f"- Connect the change to the merchant's operational context.\n"
            f"- Never infer whether the merchant is compliant or non-compliant.\n"
            f"- Offer one concrete checklist, audit, or preparation action Vera can provide.\n"
            f"- End with one binary CTA."
        )

    if kind in ("cde_opportunity", "cde"):
        return (
            f"CONTINUING EDUCATION STRATEGY (merchant-facing):\n"
            f"- State organizer, topic, date, speaker, credits, and fee only when supplied.\n"
            f"- Explain the practical relevance to the merchant's profession.\n"
            f"- Offer to share details, prepare a concise summary, or help with registration when supported.\n"
            f"- End with one binary CTA."
        )

    if kind in ("perf_dip", "seasonal_perf_dip"):
        return (
            f"PERFORMANCE DIP STRATEGY (merchant-facing):\n"
            f"- State the exact metric decline and timeframe from the merchant context.\n"
            f"- Use benchmark data only when explicitly supplied and label it as a benchmark.\n"
            f"- Connect the dip to one concrete recovery action.\n"
            f"- Use a merchant-confirmed offer only when relevant and supplied.\n"
            f"- Never invent a reason for the decline.\n"
            f"- End with one binary CTA."
        )

    if kind == "perf_spike":
        return (
            f"PERFORMANCE GAIN STRATEGY (merchant-facing):\n"
            f"- Acknowledge the exact positive metric change and timeframe when supplied.\n"
            f"- Explain the opportunity created by the improvement without guaranteeing continued growth.\n"
            f"- Suggest one concrete action to build on the momentum.\n"
            f"- End with one binary CTA."
        )

    if kind in ("renewal_due", "subscription_expiry"):
        return (
            f"RENEWAL STRATEGY (merchant-facing):\n"
            f"- State remaining days, current plan, and renewal amount only when supplied.\n"
            f"- Reference actual recent value such as views/calls only when supplied.\n"
            f"- Explain continuity benefits without guarantees.\n"
            f"- Offer the simplest available renewal next step.\n"
            f"- End with one binary CTA."
        )

    if kind == "competitor_opened":
        return (
            f"COMPETITOR OPENING STRATEGY (merchant-facing):\n"
            f"- Lead with the exact competitor name and distance from the payload.\n"
            f"- In the same message, state one verified merchant strength (rating, review count, calls/views, or active offer) when supplied.\n"
            f"- State the competitor offer only when supplied; never infer that customers will switch.\n"
            f"- Turn the facts into one specific response Vera can prepare now, such as a targeted visibility post or a merchant-confirmed offer draft.\n"
            f"- Avoid generic phrases such as 'stay competitive' or 'visibility opportunity'.\n"
            f"- End with one binary CTA."
        )

    if kind in ("festival_upcoming", "ipl_match_today", "local_news_event"):
        return (
            f"LOCAL EVENT / SEASONAL STRATEGY (merchant-facing):\n"
            f"- Reference the event, date/time, and local context only when supplied.\n"
            f"- Explain the business opportunity without guaranteeing demand or footfall.\n"
            f"- Suggest one relevant offer, post, or promotion using only supplied merchant data.\n"
            f"- End with one binary CTA."
        )

    if kind == "review_theme_emerged":
        return (
            f"REVIEW PATTERN STRATEGY (merchant-facing):\n"
            f"- State the exact review theme and occurrence count when supplied.\n"
            f"- Include one concrete review phrase or specific evidence detail when available.\n"
            f"- Translate the pattern into one practical operational or customer-communication action.\n"
            f"- Do not generalize beyond the supplied review evidence or invent sentiment.\n"
            f"- End with one binary CTA."
        )

    if kind == "milestone_reached":
        return (
            f"MILESTONE STRATEGY (merchant-facing):\n"
            f"- State the achieved milestone and next benchmark only when supplied.\n"
            f"- Celebrate briefly and keep it business-relevant.\n"
            f"- Offer one concrete way Vera can help with the next step.\n"
            f"- End with one binary CTA."
        )

    if kind == "supply_alert":
        return (
            f"SUPPLY ALERT STRATEGY (merchant-facing):\n"
            f"- State affected product, batch, manufacturer, and alert details only when supplied.\n"
            f"- Focus on the merchant's operational response.\n"
            f"- Avoid unsupported medical advice or outcome claims.\n"
            f"- Offer one concrete inventory or return-preparation action.\n"
            f"- End with one binary CTA."
        )

    if kind == "gbp_unverified":
        return (
            f"GBP VERIFICATION STRATEGY (merchant-facing):\n"
            f"- State that the profile is unverified when explicitly indicated.\n"
            f"- Mention an uplift estimate only when explicitly supplied.\n"
            f"- Offer guidance for the verification process.\n"
            f"- Do not guarantee visibility or lead increases.\n"
            f"- End with one binary CTA."
        )

    if kind == "recall_due":
        return (
            f"RECALL FOLLOW-UP STRATEGY (merchant-facing):\n"
            f"- Tell {salutation} that {customer_name} is due for the service stated in the trigger.\n"
            f"- Use last-visit date and available booking slots only when present in the payload.\n"
            f"- Frame this as a merchant opportunity to re-engage an existing customer.\n"
            f"- Use an active merchant offer only when explicitly supplied.\n"
            f"- Offer to prepare a customer follow-up for {salutation}'s approval.\n"
            f"- Never invent demand, scarcity, retention rates, or customer intent.\n"
            f"- End with one binary CTA."
        )

    if kind == "trial_followup":
        return (
            f"TRIAL FOLLOW-UP STRATEGY (merchant-facing):\n"
            f"- State the trial/event completed by {customer_name} only when explicitly present.\n"
            f"- Focus on the merchant's next follow-up or conversion action.\n"
            f"- Use service, offer, timing, or customer status only when supplied.\n"
            f"- Do not assume satisfaction, purchase intent, or readiness.\n"
            f"- Offer one concrete follow-up action Vera can prepare.\n"
            f"- End with one binary CTA."
        )

    if kind == "chronic_refill_due":
        return (
            f"REFILL CONTINUITY STRATEGY (merchant-facing):\n"
            f"- Address {salutation} as the pharmacy owner/operator; never frame the message as medical advice to the patient.\n"
            f"- Lead with the exact customer name plus medicine/product and due timing from the trigger when supplied.\n"
            f"- Explicitly connect the refill signal to one pharmacy action: prepare a reminder, check stock, or review the customer's refill request, using only supplied facts.\n"
            f"- If the trigger contains more than one medicine/item, mention the exact items rather than saying 'a refill'.\n"
            f"- Never invent adherence, diagnosis, health outcomes, or purchase intent.\n"
            f"- Avoid generic wording such as 'timely customer follow-up' without naming the actual refill event.\n"
            f"- End with one binary CTA."
        )

    if kind == "customer_lapsed_hard":
        return (
            f"LAPSED CUSTOMER WIN-BACK STRATEGY (merchant-facing):\n"
            f"- Lead with the exact inactivity signal: days since the last conversation/visit/event, plus the last known interaction detail when supplied.\n"
            f"- Add one verified merchant fact from performance or an active offer so the action is tied to this business.\n"
            f"- Make the next action explicit: draft a named win-back message, offer, or follow-up based on the supplied facts.\n"
            f"- Never invent why the customer lapsed, future intent, or likely return.\n"
            f"- Avoid vague wording like 're-engagement opportunity' unless followed by the exact action Vera will prepare.\n"
            f"- End with one binary CTA."
        )

    if kind == "wedding_package_followup":
        return (
            f"WEDDING / EVENT FOLLOW-UP STRATEGY (merchant-facing):\n"
            f"- State the exact event/package or trial context and timing when supplied.\n"
            f"- Include guest count, budget, service/package, or availability when supplied.\n"
            f"- Tie those facts directly to the merchant's next sales/follow-up action.\n"
            f"- Do not assume purchase intent or satisfaction.\n"
            f"- Offer one concrete follow-up or package-summary action Vera can prepare.\n"
            f"- End with one binary CTA."
        )

    if kind in ("curious_ask_due", "dormant_with_vera", "winback_eligible"):
        return (
            f"RE-ENGAGEMENT STRATEGY (merchant-facing):\n"
            f"- Put the exact trigger signal first: days inactive, last conversation date, views, calls, or another supplied metric.\n"
            f"- Pair it with one concrete merchant fact such as a named active offer or recent profile metric when supplied.\n"
            f"- The action must be explicit and executable: draft a check-in, win-back message, offer reminder, or profile post tied to those facts.\n"
            f"- Do not use empty phrases like 'we spotted an opportunity' or 'let's re-engage' without saying what Vera will prepare.\n"
            f"- Do not invent current demand, customer intent, or outcomes.\n"
            f"- End with one binary CTA."
        )

    if scope == "customer":
        return (
            f"CUSTOMER EVENT STRATEGY (merchant-facing):\n"
            f"- Explain the specific customer event to {salutation}.\n"
            f"- Translate it into one relevant business action for the merchant.\n"
            f"- Use customer details only when explicitly supplied and necessary.\n"
            f"- Never speak as if Vera is addressing the customer.\n"
            f"- Never invent customer intent or outcomes.\n"
            f"- End with one clear binary CTA."
        )

    return (
        f"BUSINESS OPPORTUNITY STRATEGY (merchant-facing):\n"
        f"- State the specific trigger event and why it matters now for {salutation}'s business.\n"
        f"- Use only verified facts from the supplied context.\n"
        f"- Propose one concrete action Vera can execute.\n"
        f"- Never invent metrics, demand, feedback, or outcomes.\n"
        f"- End with one binary CTA."
    )


def is_opt_out(message: str) -> bool:
    text = message.lower().strip()

    patterns = [
        r"\bstop\b",
        r"\bunsubscribe\b",
        r"\bdo not message\b",
        r"\bdon't message\b",
        r"\bdo not contact\b",
        r"\bdon't contact\b",
        r"\bremove me\b",
        r"\bstop messaging\b",
        r"\bstop contacting\b",
        r"\bspam\b",
        r"\bnot interested\b",
        r"\bleave me alone\b",
    ]

    return any(re.search(pattern, text) for pattern in patterns)


def is_auto_reply(message: str) -> bool:
    text = message.lower()

    phrases = [
        "thank you for contacting us",
        "our team will respond shortly",
        "we will respond shortly",
        "we'll respond shortly",
        "we will get back to you",
        "we'll get back to you",
        "someone will get back to you",
        "automated assistant",
        "auto-reply",
        "autoreply",
    ]

    return any(phrase in text for phrase in phrases)


def is_action_ready(message: str) -> bool:
    text = message.lower()

    phrases = [
        "lets do it",
        "let's do it",
        "do it",
        "go ahead",
        "let's proceed",
        "lets proceed",
        "proceed",
        "what's next",
        "whats next",
        "next?",
        "send it",
        "send me",
        "draft it",
        "draft this",
        "i'm in",
        "im in",
        "yes, do it",
        "yes do it",
        "ok lets do it",
        "ok let's do it",
        "sure, let's do it",
        "sure lets do it",
    ]

    return any(phrase in text for phrase in phrases)


def is_interested(message: str) -> bool:
    text = message.lower()

    phrases = [
        "yes",
        "sounds good",
        "good idea",
        "interested",
        "tell me more",
        "what would it look like",
        "how does it work",
        "how can we",
        "can we do",
    ]

    return any(phrase in text for phrase in phrases)


# ============================================================
# PROMPT BUILDER
# ============================================================

def build_prompt(
    merchant: Dict[str, Any],
    customer: Optional[Dict[str, Any]],
    trigger: Dict[str, Any],
    category: Dict[str, Any],
    conversation: Optional[list] = None,
) -> str:
    identity = merchant.get("identity", {})
    performance = merchant.get("performance", {})
    offers = merchant.get("offers", [])
    active_offers = [o.get("title") for o in offers if o.get("status") == "active"]
    signals = merchant.get("signals", [])
    customer_agg = merchant.get("customer_aggregate", {})
    review_themes = merchant.get("review_themes", [])

    voice = category.get("voice", {})
    peer_stats = category.get("peer_stats", {})

    salutation = resolve_salutation(merchant, category)
    digest_item = resolve_digest_item(trigger, category)
    customer_name = resolve_customer_name(trigger, customer)
    trigger_kind = trigger.get("kind", "")
    trigger_scope = trigger.get("scope", "merchant")
    strategy = get_trigger_strategy(trigger_kind, trigger_scope, customer_name, salutation)

    digest_block = ""
    if digest_item:
        digest_block = (
            f"\nRESOLVED INDUSTRY / REGULATORY / DIGEST ITEM:\n"
            f"Title: {digest_item.get('title', '')}\n"
            f"Source: {digest_item.get('source', '')}\n"
            f"Summary: {digest_item.get('summary', '')}\n"
            f"Actionable: {digest_item.get('actionable', '')}\n"
            f"Trial Sample Size: {digest_item.get('trial_n', '')}\n"
            f"Patient / Customer Segment: {digest_item.get('patient_segment', '')}\n"
        )

    customer_block = ""
    if trigger_scope == "customer" or trigger.get("customer_id"):
        customer_block = (
            f"\nCUSTOMER CONTEXT:\n"
            f"Customer Name: {customer_name}\n"
            f"Customer State: {customer.get('state', '') if customer else ''}\n"
            f"Relationship: {customer.get('relationship', {}) if customer else ''}\n"
        )

    prompt = f"""You are Vera, magicpin's elite AI business growth assistant.
You communicate directly TO THE MERCHANT to help them grow their business, manage customer retention, and take action.

CRITICAL ROLE & PERSPECTIVE:
- YOU ARE COMMUNICATING TO THE MERCHANT: {salutation}.
- Always address the merchant owner with: "{salutation}".
- NEVER communicate as if you are talking to a customer.
- NEVER write as a passive receptionist forwarding booking memos.
- For customer events (recalls, refills, follow-ups), present a proactive business opportunity: notify {salutation} of the customer's status and offer to reach out to the customer on the merchant's behalf.

GROUNDING & FACTUAL RULES (STRICT):
- Use ONLY facts, numbers, dates, prices, citations, and names explicitly present below.
- NEVER invent metrics, reviews, wait times, demand changes, outcomes, competitor names, or dates.
- Do NOT convert an implied possibility into a stated fact.
- Prefer exact numbers, percentages, dates, and prices when they are explicitly supplied; never invent a number just to make the message specific.
- If research or compliance is mentioned, cite the source clearly (e.g. "— JIDA Oct 2026, p.14").
- NEVER expose internal IDs, trigger IDs (e.g. "trg_001"), suppression keys, raw field names, snake_case labels, or dataset terminology like "MerchantContext" or "payload".

CATEGORY VOICE & TONE:
Category: {category.get("display_name", category.get("slug", ""))}
Tone: {voice.get("tone", "")}
Register: {voice.get("register", "")}
Allowed vocabulary: {voice.get("vocab_allowed", [])}
TABOOS (NEVER USE): {voice.get("vocab_taboo", [])}

MERCHANT CONTEXT:
Business Name: {identity.get("name", "")}
Owner: {identity.get("owner_first_name", "")}
Locality: {identity.get("locality", "")}, {identity.get("city", "")}
Languages: {identity.get("languages", ["en"])}
Performance: views={performance.get("views", "N/A")}, calls={performance.get("calls", "N/A")}, ctr={performance.get("ctr", "N/A")}, 7d_delta={performance.get("delta_7d", {})}
Active Offers (merchant-confirmed only): {active_offers}
Signals: {signals}
Customer Aggregate: {customer_agg if not (trigger_scope == "customer" or trigger.get("customer_id")) else "Not included for customer-scoped trigger"}
Review Themes: {review_themes}
Local Peer Benchmarks: {peer_stats}

TRIGGER CONTEXT:
Kind: {trigger_kind}
Scope: {trigger_scope}
Urgency: {trigger.get("urgency", 3)}/5
Payload: {json.dumps(trigger.get("payload", {}))}
{digest_block}{customer_block}
{f"RECENT CONVERSATION:{chr(10)}{json.dumps(conversation)}" if conversation else ""}

SPECIFIC MESSAGE STRATEGY:
{strategy}

OUTPUT FORMAT:
- Write ONE concise, polished WhatsApp message (2 to 4 sentences, 40-75 words).
- End with exactly ONE clear, singular, low-friction binary CTA at the very end (e.g. "Want me to send this recall invite? Reply YES to proceed." or "Want me to draft this post?").
- Do NOT repeat the question or CTA twice.
- Return ONLY the exact message text that Vera should send to {salutation}. Do not add labels like "Vera:" or quotes.
"""
    return prompt


# ============================================================
# GEMINI GENERATION
# ============================================================

def clean_model_text(value: str) -> str:
    text = (value or "").strip()
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        text = text[1:-1].strip()
    return text


def generate_message(
    merchant: Dict[str, Any],
    customer: Optional[Dict[str, Any]],
    trigger: Dict[str, Any],
    category: Dict[str, Any],
    conversation: Optional[list] = None,
) -> str:
    prompt = build_prompt(
        merchant=merchant,
        customer=customer,
        trigger=trigger,
        category=category,
        conversation=conversation,
    )

    cache_key = "single:" + _cache_key({
        "merchant": merchant,
        "customer": customer,
        "trigger": trigger,
        "category": category,
        "conversation": conversation or [],
    })
    cached = _cache_get(cache_key)
    if cached:
        return cached

    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
        )
        text = (response.text or "").strip()
    except Exception:
        text = ""

    if not text:
        salutation = resolve_salutation(merchant, category)
        return f"{salutation}, I have prepared a business update for your profile. Reply YES to review the next step."

    cleaned = clean_model_text(text)
    _cache_put(cache_key, cleaned)
    return cleaned


# ============================================================
# BATCH GEMINI GENERATION
# ============================================================

def resolve_category_evidence(category: Dict[str, Any], trigger: Dict[str, Any]) -> Dict[str, Any]:
    """Return only category evidence that can make a generated message more concrete."""
    evidence: Dict[str, Any] = {}
    digest_item = resolve_digest_item(trigger, category)
    if digest_item:
        evidence["resolved_digest_item"] = digest_item
    for key in ("trend_signals", "seasonal_beats", "peer_stats"):
        value = category.get(key)
        if value:
            evidence[key] = value
    return evidence


def generate_batch_messages(items: list[dict]) -> dict[int, str]:
    """Generate multiple merchant messages in one Gemini request."""

    if not items:
        return {}

    batch_key = "batch:" + _cache_key([
        {
            "merchant": item.get("merchant"),
            "customer": item.get("customer"),
            "trigger": item.get("trigger"),
            "category": item.get("category"),
        }
        for item in items
    ])

    cached_batch = _cache_get(batch_key)
    if cached_batch:
        try:
            parsed = json.loads(cached_batch)
            if isinstance(parsed, dict):
                return {int(k): str(v) for k, v in parsed.items()}
        except Exception:
            pass

    blocks = []

    for i, item in enumerate(items):
        merchant = item["merchant"]
        customer = item["customer"]
        trigger = item["trigger"]
        category = item["category"]

        identity = merchant.get("identity", {})
        performance = merchant.get("performance", {})
        offers = merchant.get("offers", [])
        active_offers = [o.get("title") for o in offers if o.get("status") == "active"]
        voice = category.get("voice", {})
        scope = trigger.get("scope", "merchant")
        kind = trigger.get("kind", "")
        payload = trigger.get("payload", {})
        category_evidence = resolve_category_evidence(category, trigger)

        customer_context = "No customer context."
        if scope == "customer" or trigger.get("customer_id"):
            if customer:
                customer_context = {
                    "name": customer.get("identity", {}).get("name"),
                    "state": customer.get("state"),
                    "relationship": customer.get("relationship", {}),
                }

        strategy_text = get_trigger_strategy(
            kind=kind,
            scope=scope,
            customer_name=resolve_customer_name(trigger, customer),
            salutation=resolve_salutation(merchant, category),
        )

        blocks.append(
            f"""
MESSAGE {i}

CATEGORY:
Name: {category.get("display_name", category.get("slug", ""))}
Tone: {voice.get("tone", "")}
Register: {voice.get("register", "")}
Allowed vocabulary: {voice.get("vocab_allowed", [])}
Taboos: {voice.get("vocab_taboo", [])}

MERCHANT:
Business: {identity.get("name", "")}
Owner: {identity.get("owner_first_name", "")}
Locality: {identity.get("locality", "")}, {identity.get("city", "")}
Languages: {identity.get("languages", [])}
Performance: {performance}
Active merchant offers ONLY: {active_offers}
Signals: {merchant.get("signals", [])}
Reviews: {merchant.get("review_themes", [])}

TRIGGER:
Kind: {kind}
Scope: {scope}
Urgency: {trigger.get("urgency", 3)}/5
Payload: {json.dumps(payload)}

CATEGORY EVIDENCE:
{json.dumps(category_evidence, ensure_ascii=False)}

CUSTOMER:
{json.dumps(customer_context)}

STRATEGY:
{strategy_text}
"""
        )

    prompt = f"""
You are Vera, magicpin's merchant growth assistant.

Generate ONE polished WhatsApp message for EACH MESSAGE block below.

STRICT RULES:
- Vera always communicates TO THE MERCHANT.
- Never speak as though you are messaging the customer.
- For customer-scoped triggers, translate the customer event into a relevant merchant business action.
- Use ONLY facts explicitly present in that message's context.
- Never invent metrics, counts, reviews, feedback, demand, outcomes, dates, prices, or customer intent.
- Never infer a numeric value from a qualitative signal.
- Never expose internal field names, IDs, trigger IDs, suppression keys, snake_case labels, or dataset terminology.
- Use merchant-confirmed active offers only.
- Match the supplied category voice, vocabulary, and taboos.
- Make the WHY-NOW clear.
- Prefer the most specific trigger facts over generic language.
- When the context contains two or more concrete evidence points, include at least two of them in the message.
- Give one concrete, useful next step tied to those facts.
- Use exactly ONE low-friction CTA.
- 2 to 4 sentences and roughly 40 to 75 words per message.
- Keep each message specific to its own context.
- Do not copy strategy labels into the merchant-facing message.
- NEVER use generic filler such as "relevant business opportunity", "re-engagement opportunity", "stay competitive", or "timely follow-up" unless the same sentence also names the exact trigger fact and the exact action Vera will prepare.
- For chronic_refill_due: merchant-facing wording must clearly name the customer and the actual refill medicine/item and due timing when supplied, then name one pharmacy action.
- For competitor_opened: include the exact competitor name + distance and at least one verified merchant strength or active offer when supplied, then propose one concrete response.
- For customer_lapsed_hard / dormant_with_vera / winback_eligible: include the exact inactivity/lapse signal and one concrete merchant fact, then propose a specific win-back/check-in asset Vera can draft.
- For wedding_package_followup / trial_followup: explicitly anchor the message to the salon/fitness trial or event context and the supplied service/package/timing; do not use generic "follow-up" wording alone.

Return ONLY valid JSON:
{{
  "messages": [
    {{"index": 0, "body": "..."}},
    {{"index": 1, "body": "..."}}
  ]
}}

{''.join(blocks)}
"""

    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
        )

        raw = (response.text or "").strip()
        match = re.search(r"\{[\s\S]*\}", raw)

        if not match:
            return {}

        data = json.loads(match.group())
        result: dict[int, str] = {}

        for item in data.get("messages", []):
            try:
                index = int(item.get("index"))
                body = clean_model_text(str(item.get("body", "")))
                if body:
                    result[index] = body
            except Exception:
                continue

        _cache_put(
            batch_key,
            json.dumps(result, ensure_ascii=False, sort_keys=True),
        )
        return result

    except Exception:
        return {}


# ============================================================
# HEALTH & METADATA ENDPOINTS
# ============================================================

@app.get("/v1/healthz")
@app.get("/")
def healthz():
    return {
        "status": "ok",
        "service": "vera-ai",
        "contexts_loaded": {
            "category": len(categories),
            "merchant": len(merchants),
            "customer": len(customers),
            "trigger": len(triggers),
        },
    }


@app.get("/v1/metadata")
def metadata():
    return {
        "team_name": "Vera AI",
        "model": MODEL,
    }


# ============================================================
# CONTEXT ENDPOINT
# ============================================================

@app.post("/v1/context")
def push_context(request: ContextRequest):

    accepted = store_context(
        scope=request.scope,
        context_id=request.context_id,
        version=request.version,
        payload=request.payload,
    )

    return {
        "accepted": accepted,
        "scope": request.scope,
        "context_id": request.context_id,
        "version": request.version,
    }


# ============================================================
# TICK ENDPOINT
# ============================================================

@app.post("/v1/tick")
def tick(request: TickRequest):

    actions = []
    pending = []

    for trigger_id in request.available_triggers:
        trigger = triggers.get(trigger_id)
        if not trigger:
            continue

        merchant_id = trigger.get("merchant_id")
        customer_id = trigger.get("customer_id")
        merchant = merchants.get(merchant_id)

        if not merchant:
            continue

        customer = get_customer(customer_id)
        category = get_category_for_merchant(merchant)
        payload = trigger.get("payload", {})
        merchant_last_message = payload.get("merchant_last_message", "")

        if merchant_last_message and is_opt_out(merchant_last_message):
            actions.append({
                "conversation_id": f"conv_{trigger_id}",
                "trigger_id": trigger_id,
                "merchant_id": merchant_id,
                "customer_id": customer_id,
                "action": "end",
                "body": "",
                "cta": "",
                "send_as": "vera",
            })
            continue

        if merchant_last_message and is_auto_reply(merchant_last_message):
            actions.append({
                "conversation_id": f"conv_{trigger_id}",
                "trigger_id": trigger_id,
                "merchant_id": merchant_id,
                "customer_id": customer_id,
                "action": "wait",
                "wait_seconds": 1800,
                "body": "",
                "cta": "",
                "send_as": "vera",
            })
            continue

        pending.append({
            "trigger_id": trigger_id,
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "merchant": merchant,
            "customer": customer,
            "category": category,
            "trigger": trigger,
        })

    generated = generate_batch_messages(pending)

    for index, item in enumerate(pending):
        message = generated.get(index)

        if not message:
            salutation = resolve_salutation(item["merchant"], item["category"])
            message = (
                f"{salutation}, I noticed a relevant business opportunity in your current profile. "
                f"Reply YES and I’ll prepare the next step."
            )

        actions.append({
            "conversation_id": f"conv_{item['trigger_id']}",
            "trigger_id": item["trigger_id"],
            "merchant_id": item["merchant_id"],
            "customer_id": item["customer_id"],
            "action": "send",
            "send_as": "vera",
            "body": message,
            "cta": "binary_yes_no",
        })

    return {"actions": actions}


# ============================================================
# LIVE REPLY ENDPOINT
# ============================================================

@app.post("/v1/reply")
def reply(request: ReplyRequest):

    message = request.message.strip()

    # Explicit opt-out
    if is_opt_out(message):
        return {
            "action": "end",
            "body": "",
        }

    # Automated acknowledgement
    if is_auto_reply(message):
        if request.turn_number >= 3:
            return {
                "action": "end",
                "body": "",
            }
        return {
            "action": "wait",
            "wait_seconds": 1800,
            "body": "",
        }

    # Direct action mode transition when merchant expresses commitment
    if is_action_ready(message):
        return {
            "action": "send",
            "body": "Done! Proceeding with this now. I have confirmed the details and am sending the draft over for your review next.",
        }

    merchant = merchants.get(request.merchant_id, {})
    if not merchant:
        merchant = {"identity": {"name": "Merchant", "owner_first_name": "Partner"}}

    category = get_category_for_merchant(merchant)

    conversation = conversations.setdefault(
        request.conversation_id,
        [],
    )

    conversation.append({
        "role": request.from_role or "merchant",
        "body": message,
        "turn": request.turn_number,
    })

    # Find a relevant trigger for this merchant if one exists.
    relevant_trigger = None

    for trigger in triggers.values():
        if trigger.get("merchant_id") == request.merchant_id:
            relevant_trigger = trigger
            break

    if relevant_trigger is None:
        relevant_trigger = {
            "scope": "merchant",
            "kind": "conversation_reply",
            "payload": {
                "merchant_last_message": message,
            },
        }

    customer = get_customer(
        relevant_trigger.get("customer_id") or request.customer_id
    )

    generated = generate_message(
        merchant=merchant,
        customer=customer,
        trigger=relevant_trigger,
        category=category,
        conversation=conversation,
    )

    return {
        "action": "send",
        "body": generated,
    }


# ============================================================
# LOCAL RUNNER
# ============================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "bot:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
    )