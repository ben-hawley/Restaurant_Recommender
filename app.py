import json
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import requests
import streamlit as st
from openai import OpenAI

# ---------------- Constants ----------------
GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"
PLACES_NEARBY_URL = "https://places.googleapis.com/v1/places:searchNearby"
PLACES_TEXT_SEARCH_URL = "https://places.googleapis.com/v1/places:searchText"

FIELD_MASK = (
    "places.id,places.displayName,places.formattedAddress,"
    "places.location,places.rating,places.userRatingCount,"
    "places.priceLevel,places.types,places.primaryType"
)

EXCLUDED_TYPES_NEARBY = ["bakery", "cafe", "meal_takeaway"]
RADIUS_FUDGE = 1.2  # allow 20% slack


# ---------------- Data models ----------------
@dataclass
class UserPrefs:
    where: str
    keywords: str
    radius_miles: float
    max_results: int
    excluded_terms: List[str]


# ---------------- Small utils ----------------
def tokenize(text: str) -> List[str]:
    text = (text or "").lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if t]


def miles_to_meters(miles: float) -> int:
    return int(miles * 1609.34)


def haversine_miles(lat1, lon1, lat2, lon2) -> float:
    R = 3958.7613
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def safe_get_name(p: Dict) -> str:
    return ((p.get("displayName") or {}) or {}).get("text", "Unknown")


def is_restaurant(p: Dict) -> bool:
    primary = (p.get("primaryType") or "").lower()
    types = [t.lower() for t in (p.get("types") or [])]
    return ("restaurant" in primary) or ("restaurant" in types)


def distance_from_user_miles(p: Dict, user_lat: float, user_lng: float) -> Optional[float]:
    plat = (p.get("location") or {}).get("latitude")
    plng = (p.get("location") or {}).get("longitude")
    if plat is None or plng is None:
        return None
    return haversine_miles(user_lat, user_lng, float(plat), float(plng))


def place_text_blob(p: Dict) -> str:
    name = safe_get_name(p)
    types = " ".join(p.get("types") or []) + " " + str(p.get("primaryType") or "")
    return (name + " " + types).lower()


def passes_exclusions(p: Dict, excluded_terms: List[str]) -> bool:
    if not excluded_terms:
        return True
    blob = place_text_blob(p)
    # simple substring gate, but terms are short tokens (e.g. "american", "bbq")
    return not any(t and (t in blob) for t in excluded_terms)


# ---- Refinement parsing (include + exclude tokens) ----
CUISINE_TERMS = [
    "mexican", "tacos", "italian", "pizza", "thai", "indian", "chinese", "japanese", "korean",
    "vietnamese", "pho", "ramen", "sushi", "mediterranean", "greek", "turkish", "lebanese",
    "middle eastern", "ethiopian", "moroccan", "french", "spanish", "peruvian", "brazilian",
    "cuban", "caribbean", "bbq", "barbecue", "seafood", "steakhouse", "vegetarian", "vegan",
    "gluten free", "halal", "kosher", "dimsum", "dim sum", "dumplings", "noodles", "burgers",
    "american", "southern"
]

VIBE_TERMS = [
    "upscale", "fine dining", "fancy", "casual", "relaxed", "family friendly", "date night",
    "romantic", "outdoor patio", "patio", "outdoor", "quiet", "lively"
]


def extract_refinement_tokens(answer: str) -> Tuple[List[str], List[str]]:
    """
    Convert free-text refinement into:
      - include_terms: short safe tokens to ADD to keywords
      - exclude_terms: short safe tokens to FILTER OUT in results

    Handles patterns like:
      - "not american cuisine"
      - "anything but bbq"
      - "no sushi"
      - "except italian"
    """
    t = (answer or "").lower().strip()
    if not t:
        return [], []

    # Normalize a few phrases
    t = t.replace("anything but", "not")
    t = t.replace("dont want", "not")
    t = t.replace("don't want", "not")
    t = t.replace("do not want", "not")

    include_terms: List[str] = []
    exclude_terms: List[str] = []

    # 1) Exclusions via "not/no/except" patterns
    #    Grab 1-3 tokens after not/no/except
    for m in re.finditer(r"\b(not|no|except)\b\s+([a-z0-9\s]{1,40})", t):
        phrase = m.group(2).strip()
        toks = tokenize(phrase)[:3]
        # if user says "except X", that implies include X (not exclude)
        if m.group(1) == "except":
            include_terms.extend(toks)
        else:
            exclude_terms.extend(toks)

    # 2) Add recognized cuisine/vibe terms if present
    #    (If user only says "not american", we still want exclude=['american'] to work)
    def add_if_present(terms: List[str], dest: List[str]):
        for s in terms:
            if s in t:
                dest.append(s)

    add_if_present(VIBE_TERMS, include_terms)
    add_if_present(CUISINE_TERMS, include_terms)

    # 3) If user typed a simple cuisine like "thai" (not in phrases above), include it
    #    but keep it very small to stay search-safe.
    raw_toks = tokenize(t)
    # keep meaningful tokens only
    raw_toks = [x for x in raw_toks if x not in {"breakfast", "lunch", "dinner", "food", "cuisine", "restaurant"}]
    include_terms.extend(raw_toks[:4])

    # Dedup, keep short
    def dedup_keep(xs: List[str], k: int) -> List[str]:
        out, seen = [], set()
        for x in xs:
            x = x.strip()
            if not x or x in seen:
                continue
            seen.add(x)
            out.append(x)
            if len(out) >= k:
                break
        return out

    include_terms = dedup_keep(include_terms, 8)
    exclude_terms = dedup_keep(exclude_terms, 8)

    # If "american" is excluded and also included (from raw tokens), remove from include
    exclude_set = set(exclude_terms)
    include_terms = [x for x in include_terms if x not in exclude_set]

    return include_terms, exclude_terms


# ---------------- Google API calls (cached) ----------------
@st.cache_data(show_spinner=False, ttl=60 * 60)
def geocode(api_key: str, where: str) -> Tuple[float, float]:
    params = {"address": where, "key": api_key}
    r = requests.get(GEOCODE_URL, params=params, timeout=20)
    data = r.json()
    if data.get("status") != "OK" or not data.get("results"):
        raise RuntimeError(f"Geocoding failed: {data.get('status')} {data.get('error_message','')}")
    loc = data["results"][0]["geometry"]["location"]
    return float(loc["lat"]), float(loc["lng"])


@st.cache_data(show_spinner=False, ttl=60 * 10)
def places_nearby(api_key: str, lat: float, lng: float, radius_m: int, max_results: int) -> List[Dict]:
    headers = {"Content-Type": "application/json", "X-Goog-Api-Key": api_key, "X-Goog-FieldMask": FIELD_MASK}
    body = {
        "includedTypes": ["restaurant"],
        "excludedTypes": EXCLUDED_TYPES_NEARBY,
        "maxResultCount": min(max_results, 20),
        "locationRestriction": {"circle": {"center": {"latitude": lat, "longitude": lng}, "radius": float(radius_m)}},
    }
    r = requests.post(PLACES_NEARBY_URL, headers=headers, json=body, timeout=20)
    if r.status_code != 200:
        raise RuntimeError(f"Places nearby error {r.status_code}: {r.text}")
    return r.json().get("places", [])


@st.cache_data(show_spinner=False, ttl=60 * 10)
def places_text_search(api_key: str, query: str, lat: float, lng: float, radius_m: int, max_results: int) -> List[Dict]:
    headers = {"Content-Type": "application/json", "X-Goog-Api-Key": api_key, "X-Goog-FieldMask": FIELD_MASK}
    body = {
        "textQuery": query,
        "maxResultCount": min(max_results, 20),
        "locationBias": {"circle": {"center": {"latitude": lat, "longitude": lng}, "radius": float(radius_m)}},
    }
    r = requests.post(PLACES_TEXT_SEARCH_URL, headers=headers, json=body, timeout=20)
    if r.status_code != 200:
        raise RuntimeError(f"Places text search error {r.status_code}: {r.text}")
    return r.json().get("places", [])


# ---------------- Scoring ----------------
def score_place(p: Dict, prefs: UserPrefs, user_lat: float, user_lng: float) -> Tuple[float, Dict[str, float]]:
    rating = float(p.get("rating") or 0.0)
    nratings = int(p.get("userRatingCount") or 0)

    dist = distance_from_user_miles(p, user_lat, user_lng)
    distance_miles = dist if dist is not None else 999.0

    name = safe_get_name(p)
    types = " ".join(p.get("types") or []) + " " + str(p.get("primaryType") or "")

    q_toks = set(tokenize(prefs.keywords))
    t_toks = set(tokenize(name + " " + types))
    match = (len(q_toks.intersection(t_toks)) / max(len(q_toks), 1)) if q_toks else 0.0

    s_rating = rating * 2.0
    s_pop = math.log1p(nratings) * 0.6
    s_dist = -distance_miles * 1.0
    s_match = match * 4.0

    total = s_rating + s_pop + s_dist + s_match
    return total, {"rating": s_rating, "popularity": s_pop, "distance": s_dist, "match": s_match}


def reason_line(p: Dict, parts: Dict[str, float], prefs: UserPrefs, user_lat: float, user_lng: float) -> str:
    bits = []
    if prefs.keywords.strip() and parts.get("match", 0) > 0.4:
        bits.append(f"matches “{prefs.keywords.strip()}”")
    if p.get("rating") is not None:
        bits.append(f"{p.get('rating')}★ ({p.get('userRatingCount', 0)} ratings)")
    dist = distance_from_user_miles(p, user_lat, user_lng)
    if dist is not None:
        bits.append(f"{dist:.1f} mi away")
    if prefs.excluded_terms:
        bits.append(f"filtered: not {', '.join(prefs.excluded_terms[:3])}{'…' if len(prefs.excluded_terms) > 3 else ''}")
    return " • ".join(bits)


# ---------------- OpenAI helpers ----------------
def get_openai_client() -> Optional[OpenAI]:
    try:
        key = st.secrets.get("OPENAI_API_KEY", None)
    except Exception:
        key = None
    return OpenAI(api_key=key) if key else None


def simplify_place_for_llm(p: Dict, user_lat: float, user_lng: float) -> Dict:
    dist = distance_from_user_miles(p, user_lat, user_lng)
    return {
        "name": safe_get_name(p),
        "address": p.get("formattedAddress", ""),
        "rating": p.get("rating", None),
        "userRatingCount": p.get("userRatingCount", None),
        "priceLevel": p.get("priceLevel", None),
        "primaryType": p.get("primaryType", None),
        "types": p.get("types", []),
        "distanceMiles": (None if dist is None else round(dist, 2)),
    }


def _extract_json_object(text: str) -> Optional[Dict]:
    text = (text or "").strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def openai_explain_recs(
    client: OpenAI,
    prefs: UserPrefs,
    candidates: List[Dict],
    model: str = "gpt-4.1",
) -> Optional[Dict]:
    payload = {
        "user_preferences": {
            "where": prefs.where,
            "keywords": prefs.keywords,
            "excluded_terms": prefs.excluded_terms,
            "radius_miles": prefs.radius_miles,
        },
        "shown_candidates": candidates,
        "output_schema": {
            "summary": "string, 2-4 sentences max",
            "follow_up_question": "string or null (ask exactly ONE question only if it would materially improve recommendations)",
            "why_by_name": "object mapping restaurant name -> 1 short sentence explanation",
        },
    }

    system = (
        "You are a helpful restaurant recommender. "
        "Return STRICT JSON only (no markdown, no extra text). "
        "Ask at most one short follow-up question if it would materially improve recommendations. "
        "Do NOT suggest non-restaurants."
    )

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload)},
        ],
        temperature=0.4,
    )

    return _extract_json_object(resp.output_text)


# ---------------- Compute + store results ----------------
def compute_and_store_results(
    google_key: str,
    openai_client: Optional[OpenAI],
    prefs: UserPrefs,
    use_ai: bool,
    model_choice: str,
):
    user_lat, user_lng = geocode(google_key, prefs.where)
    radius_m = miles_to_meters(prefs.radius_miles)

    # Query uses ONLY include keywords (exclusions are post-filters)
    if prefs.keywords.strip():
        query = f"{prefs.keywords.strip()} restaurant"
        places = places_text_search(google_key, query, user_lat, user_lng, radius_m, prefs.max_results)
    else:
        places = places_nearby(google_key, user_lat, user_lng, radius_m, prefs.max_results)

    # HARD FILTERS
    places = [p for p in places if is_restaurant(p)]

    # Distance sanity
    max_dist = prefs.radius_miles * RADIUS_FUDGE
    pruned = []
    for p in places:
        d = distance_from_user_miles(p, user_lat, user_lng)
        if d is not None and d <= max_dist:
            pruned.append(p)
    places = pruned

    # Exclusions (this is what makes "not american cuisine" actually change output)
    places = [p for p in places if passes_exclusions(p, prefs.excluded_terms)]

    if not places:
        st.session_state["results"] = {
            "prefs": prefs,
            "lat": user_lat,
            "lng": user_lng,
            "top5": [],
            "ai_data": None,
            "error": "No restaurants found after filtering. Try increasing radius, changing keywords, or relaxing exclusions.",
        }
        return

    scored = []
    for p in places:
        total, parts = score_place(p, prefs, user_lat, user_lng)
        scored.append((total, p, parts))
    scored.sort(key=lambda x: x[0], reverse=True)
    top5 = scored[:5]

    ai_data = None
    if use_ai and openai_client is not None:
        simplified = [simplify_place_for_llm(p, user_lat, user_lng) for (_, p, _) in top5]
        ai_data = openai_explain_recs(openai_client, prefs, simplified, model=model_choice)

    st.session_state["results"] = {
        "prefs": prefs,
        "lat": user_lat,
        "lng": user_lng,
        "top5": top5,
        "ai_data": ai_data,
        "error": None,
    }


# ---------------- App UI ----------------
st.set_page_config(page_title="Restaurant Recommender (Google Places)", page_icon="🍔")
st.title("🍔 Restaurant Recommender")

# Keys
try:
    google_key = st.secrets.get("GOOGLE_MAPS_API_KEY", None)
except Exception:
    google_key = None

if not google_key:
    st.warning("No GOOGLE_MAPS_API_KEY in secrets. Paste for local testing.")
    google_key = st.text_input("Google Maps API Key", type="password")

openai_client = get_openai_client()

# Session state init
if "results" not in st.session_state:
    st.session_state["results"] = None
if "should_run" not in st.session_state:
    st.session_state["should_run"] = False
if "active_keywords" not in st.session_state:
    st.session_state["active_keywords"] = ""
if "active_excluded" not in st.session_state:
    st.session_state["active_excluded"] = []
if "follow_answer" not in st.session_state:
    st.session_state["follow_answer"] = ""
if "last_follow_question" not in st.session_state:
    st.session_state["last_follow_question"] = ""


with st.form("prefs_form"):
    where = st.text_input("Where are you?", value="Los Angeles, CA")
    keywords_in = st.text_input("Cuisine or vibe keywords", value="tacos outdoor patio")
    radius_miles = st.slider("Radius (miles)", 0.5, 25.0, 2.5, 0.5)
    max_results = st.slider("Candidates to consider", 5, 20, 15, 1)

    use_ai = st.checkbox("Use AI to explain recommendations", value=True)
    model_choice = st.selectbox("AI model", ["gpt-4.1", "gpt-4.1-mini"], index=0)

    submit = st.form_submit_button("Recommend")

# Active filters display + clear (stable widgets, no duplication)
active_kw = (st.session_state["active_keywords"] or "").strip() or keywords_in.strip()
active_ex = st.session_state.get("active_excluded", []) or []

c1, c2 = st.columns([2, 1])
with c1:
    ex_txt = f" | Excluding: **{', '.join(active_ex)}**" if active_ex else ""
    st.caption(f"Active keywords: **{active_kw or '(none)'}**{ex_txt}")
with c2:
    if st.button("Clear refinement", key="clear_refinement_btn"):
        st.session_state["active_keywords"] = ""
        st.session_state["active_excluded"] = []
        st.session_state["follow_answer"] = ""
        st.session_state["last_follow_question"] = ""
        st.session_state["results"] = None
        st.session_state["should_run"] = False
        st.rerun()

if submit:
    st.session_state["active_keywords"] = keywords_in.strip()
    st.session_state["active_excluded"] = []  # new search resets exclusions
    st.session_state["should_run"] = True

# Compute step
if st.session_state["should_run"]:
    if not google_key:
        st.error("Missing Google Maps API key.")
        st.stop()

    prefs = UserPrefs(
        where=where,
        keywords=st.session_state["active_keywords"],
        radius_miles=radius_miles,
        max_results=max_results,
        excluded_terms=st.session_state.get("active_excluded", []) or [],
    )

    try:
        with st.spinner("Finding restaurants..."):
            compute_and_store_results(google_key, openai_client, prefs, use_ai, model_choice)
    except Exception as e:
        st.session_state["results"] = {
            "prefs": prefs,
            "lat": None,
            "lng": None,
            "top5": [],
            "ai_data": None,
            "error": str(e),
        }

    st.session_state["should_run"] = False

# Render
res = st.session_state.get("results")
if not res:
    st.caption("Enter your preferences and click **Recommend** to get started.")
    st.stop()

if res.get("error"):
    st.info(res["error"])
    st.stop()

prefs: UserPrefs = res["prefs"]
user_lat, user_lng = res["lat"], res["lng"]
top5 = res["top5"]
ai_data = res.get("ai_data")

why_by_name: Dict[str, str] = {}
follow_question: Optional[str] = None

if isinstance(ai_data, dict):
    summary = ai_data.get("summary")
    follow_question = ai_data.get("follow_up_question")
    why_by_name = ai_data.get("why_by_name") if isinstance(ai_data.get("why_by_name"), dict) else {}
    if isinstance(summary, str) and summary.strip():
        st.info(summary.strip())

# Follow-up refinement UI (FIX: use a FORM + stable keys so widgets don't duplicate)
if isinstance(follow_question, str) and follow_question.strip():
    fq = follow_question.strip()
    st.session_state["last_follow_question"] = fq

    st.write("**Quick question to refine this further:**")
    st.write(fq)

    with st.form("followup_form", clear_on_submit=False):
        ans = st.text_input("Your answer", value=st.session_state["follow_answer"], key="follow_answer_box")
        refine_submit = st.form_submit_button("Refine recommendations")

    if refine_submit:
        st.session_state["follow_answer"] = ans
        ans_clean = (ans or "").strip()
        if ans_clean:
            include_terms, exclude_terms = extract_refinement_tokens(ans_clean)

            # Update include keywords (safe, short tokens only)
            if include_terms:
                st.session_state["active_keywords"] = (prefs.keywords + " " + " ".join(include_terms)).strip()

            # Update exclusions (this is what makes "not american cuisine" work)
            if exclude_terms:
                existing = st.session_state.get("active_excluded", []) or []
                # merge + dedup
                merged = []
                seen = set()
                for x in (existing + exclude_terms):
                    if x and x not in seen:
                        seen.add(x)
                        merged.append(x)
                st.session_state["active_excluded"] = merged[:12]

            st.session_state["should_run"] = True

        st.rerun()

st.success(f"Top picks near {prefs.where}:")

for i, (total, p, parts) in enumerate(top5, start=1):
    name = safe_get_name(p)
    addr = p.get("formattedAddress", "")
    st.subheader(f"#{i} {name}")
    if addr:
        st.write(addr)

    ai_why = why_by_name.get(name) if isinstance(why_by_name, dict) else None
    if isinstance(ai_why, str) and ai_why.strip():
        st.caption(ai_why.strip())
    else:
        st.caption(reason_line(p, parts, prefs, user_lat, user_lng))

    st.divider()