"""Symbolic numeric canonicalization for financial facts.

Resolves scale multipliers (million/billion/crore/lakh), currency symbols,
accounting parenthetical negatives, and financial predicate standardization.

Design basis:
- SemEval-2024 NumEval Task 7: NLI models cannot do arithmetic on raw text
- TabFact symbolic reasoning channel: deterministic numeric comparison
- quantulum3/numerizer patterns adapted for Indian financial documents
"""

import re

# ---------------------------------------------------------------------------
# Scale multiplier map — covers International + Indian number systems
# ---------------------------------------------------------------------------

SCALE_MAP: dict[str, float] = {
    # International system
    "hundred": 1e2,
    "thousand": 1e3, "k": 1e3,
    "million": 1e6, "mn": 1e6, "mln": 1e6, "mm": 1e6,
    "billion": 1e9, "bn": 1e9, "bln": 1e9,
    "trillion": 1e12, "tn": 1e12, "trn": 1e12,
    # Indian system
    "lakh": 1e5, "lac": 1e5, "lakhs": 1e5, "lacs": 1e5,
    "crore": 1e7, "cr": 1e7, "crores": 1e7, "crs": 1e7,
}

# Single-letter abbreviations that need word-boundary context to avoid
# false positives (e.g. "M" in "Mumbai" should not match)
_AMBIGUOUS_SINGLE = {"m", "b", "t"}

# ---------------------------------------------------------------------------
# Currency symbol → ISO code
# ---------------------------------------------------------------------------

CURRENCY_MAP: dict[str, str] = {
    "₹": "INR", "rs": "INR", "rs.": "INR", "inr": "INR", "rupee": "INR", "rupees": "INR",
    "$": "USD", "usd": "USD", "us$": "USD",
    "€": "EUR", "eur": "EUR",
    "£": "GBP", "gbp": "GBP",
    "¥": "JPY", "jpy": "JPY",
    "cny": "CNY", "rmb": "CNY",
}

# Regex for currency symbols at the start of a value string
_CURRENCY_SYMBOL_RE = re.compile(r'^[₹$€£¥]')
_CURRENCY_WORD_RE = re.compile(
    r'\b(rs\.?|inr|usd|us\$|eur|gbp|jpy|cny|rmb|rupees?)\b',
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Core number parsing regex
# ---------------------------------------------------------------------------

# Matches patterns like: 1,500.30, (8,911.39), -1500, 1.5, .5
_NUMBER_RE = re.compile(
    r'(?P<paren_open>\()?'           # optional opening paren (accounting negative)
    r'(?P<sign>[-+])?'               # optional sign
    r'(?P<digits>[\d,]+\.?\d*|\.?\d+)'  # the number itself (with optional commas)
    r'(?P<paren_close>\))?'          # optional closing paren
)

# Scale suffix following a number
_SCALE_RE = re.compile(
    r'\b(' + '|'.join(re.escape(k) for k in sorted(SCALE_MAP.keys(), key=len, reverse=True)) + r')\b',
    re.IGNORECASE,
)


def _strip_commas(s: str) -> str:
    return s.replace(',', '')


def parse_numeric_value(text: str) -> tuple[float | None, str | None, float | None]:
    """Parse a raw value string into (base_number, scale_suffix, canonical_value).

    Returns (None, None, None) if text is not numeric.

    Examples:
        "$1,500 million"  → (1500.0, "million", 1_500_000_000.0)
        "₹2.5 crore"     → (2.5, "crore", 25_000_000.0)
        "(8,911.39)"      → (-8911.39, None, -8911.39)
        "14.5%"           → (14.5, None, 14.5)
        "13,087"          → (13087.0, None, 13087.0)
        "Not applicable"  → (None, None, None)
        "August 2021"     → (None, None, None)
    """
    if not text or not text.strip():
        return None, None, None

    clean = text.strip()

    # Guard: reject date/month strings that contain numbers but aren't numeric facts
    _month_names = (
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
        "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
    )
    clean_lower = clean.lower()
    if any(month in clean_lower for month in _month_names):
        return None, None, None
    # Also reject "FY2022", "Q3 2021", "CY2020" type temporal strings
    if re.match(r'^(fy|cy|q[1-4])\s*\d{2,4}', clean_lower):
        return None, None, None

    # Strip currency symbols for number extraction
    clean_for_number = _CURRENCY_SYMBOL_RE.sub('', clean).strip()

    # Find the number
    match = _NUMBER_RE.search(clean_for_number)
    if not match:
        return None, None, None

    digits_str = match.group('digits')
    if not digits_str:
        return None, None, None

    # Check this is actually a number (not just commas/dots)
    digits_clean = _strip_commas(digits_str)
    if not digits_clean or not any(c.isdigit() for c in digits_clean):
        return None, None, None

    try:
        base_number = float(digits_clean)
    except ValueError:
        return None, None, None

    # Handle accounting parenthetical negatives: (8,911.39) → -8911.39
    is_paren_negative = bool(match.group('paren_open') and match.group('paren_close'))
    is_sign_negative = match.group('sign') == '-'
    if is_paren_negative or is_sign_negative:
        base_number = -abs(base_number)

    # Find scale multiplier in the text after the number
    text_after_number = clean_for_number[match.end():].strip()
    scale_suffix = None
    multiplier = 1.0

    scale_match = _SCALE_RE.search(text_after_number)
    if scale_match:
        matched_word = scale_match.group(1).lower()
        # Guard against single-letter false positives
        if matched_word in _AMBIGUOUS_SINGLE:
            # Only accept if it's at a word boundary and looks intentional
            # (i.e., the text after number starts with this letter)
            if text_after_number.lower().startswith(matched_word):
                scale_suffix = matched_word
                multiplier = SCALE_MAP[matched_word]
        else:
            scale_suffix = matched_word
            multiplier = SCALE_MAP[matched_word]

    canonical_value = base_number * multiplier

    return base_number, scale_suffix, canonical_value


def extract_scale_from_unit(unit: str | None) -> float:
    """Extract the scale multiplier from a unit string.

    When the LLM puts the scale word in the unit field (e.g. "INR million")
    rather than in the value string, we need to pull it out separately.

    Returns 1.0 if no scale word found.

    Examples:
        "INR million"       → 1_000_000.0
        "million"           → 1_000_000.0
        "INR crore"         → 10_000_000.0
        "million square ft" → 1_000_000.0
        "%"                 → 1.0
        "locations"         → 1.0
        None                → 1.0
    """
    if not unit:
        return 1.0

    unit_lower = unit.strip().lower()
    # Search for any scale word in the unit string
    for scale_word, multiplier in sorted(SCALE_MAP.items(), key=lambda x: len(x[0]), reverse=True):
        if scale_word in _AMBIGUOUS_SINGLE:
            continue  # Skip ambiguous single-letter abbreviations in unit context
        pattern = re.compile(r'\b' + re.escape(scale_word) + r's?\b', re.IGNORECASE)
        if pattern.search(unit_lower):
            return multiplier

    return 1.0


def extract_currency(text: str) -> str | None:
    """Extract ISO currency code from a value string.

    Examples:
        "₹48,105.30"       → "INR"
        "$1,500 million"    → "USD"
        "14.5%"             → None
        "Rs. 500 crore"     → "INR"
    """
    if not text:
        return None

    # Check for currency symbols first
    sym_match = _CURRENCY_SYMBOL_RE.search(text)
    if sym_match:
        return CURRENCY_MAP.get(sym_match.group(), None)

    # Check for currency words
    word_match = _CURRENCY_WORD_RE.search(text)
    if word_match:
        return CURRENCY_MAP.get(word_match.group(1).lower().rstrip('.'), None)

    return None


def canonicalize_unit(raw_unit: str | None, currency: str | None) -> str | None:
    """Normalize a unit string into a canonical form.

    The scale multiplier is already folded into canonical_value,
    so we strip scale words from the unit, leaving only the base dimension.

    Examples:
        ("INR million", "INR")  → "INR"
        ("Rs. crore", "INR")    → "INR"
        ("%", None)             → "%"
        ("locations", None)     → "locations"
        ("shipments", None)     → "shipments"
        (None, "USD")           → "USD"
    """
    if not raw_unit and not currency:
        return None

    if not raw_unit:
        return currency

    # Strip scale words from unit
    clean_unit = raw_unit.strip()
    for scale_word in SCALE_MAP:
        pattern = re.compile(r'\b' + re.escape(scale_word) + r's?\b', re.IGNORECASE)
        clean_unit = pattern.sub('', clean_unit).strip()

    # Strip currency words/symbols from unit (they go into canonical_unit via currency)
    for curr_word in CURRENCY_MAP:
        if len(curr_word) <= 1:  # Skip single symbols, handle separately
            continue
        pattern = re.compile(r'\b' + re.escape(curr_word) + r'\b', re.IGNORECASE)
        clean_unit = pattern.sub('', clean_unit).strip()

    # Strip currency symbols
    clean_unit = _CURRENCY_SYMBOL_RE.sub('', clean_unit).strip()

    # Remove trailing/leading dots, spaces
    clean_unit = clean_unit.strip('. ')

    # If we cleaned everything away, return the currency or None
    if not clean_unit:
        return currency

    # If the remaining is a pure dimension (%, locations, etc.), return it
    # If currency was also detected, prefer currency for financial values
    if currency and clean_unit.lower() in ('', 'in'):
        return currency

    return clean_unit if clean_unit else currency


# ---------------------------------------------------------------------------
# Temporal normalization
# ---------------------------------------------------------------------------

def normalize_temporal(text: str | None) -> str | None:
    """Normalize fiscal years and temporal periods into canonical strings.

    Examples:
        '2024-25', '2024/25', 'FY25', 'FY2024/25' -> 'fy2024-25'
        'Q1:2024-25', 'Q1 FY25'                    -> 'fy2024-25 q1'
        'H1 FY25', 'H1:2024-25'                    -> 'fy2024-25 h1'
        '2025-26', 'FY26'                          -> 'fy2025-26'
    """
    if not text:
        return None
    s = text.strip().lower()

    # Extract quarter or half-year if present
    sub_period = None
    q_match = re.search(r'\b(q[1-4]|h[1-2])\b', s)
    if q_match:
        sub_period = q_match.group(1).lower()

    # Pattern: 4-digit start year + 2-to-4 digit end year (e.g., 2024-25, 2024/25, 2024-2025)
    m = re.search(r'\b(?:fy\s*)?(\d{4})[-/](\d{2,4})\b', s)
    if m:
        start_yr = int(m.group(1))
        end_str = m.group(2)
        end_yr = int(end_str) if len(end_str) == 4 else (start_yr // 100 * 100 + int(end_str))
        base = f"fy{start_yr}-{str(end_yr)[2:]}"
        return f"{base} {sub_period}" if sub_period else base

    # Pattern: FY followed by 2 digits (e.g., fy25 -> fy2024-25)
    m2 = re.search(r'\bfy\s*(\d{2})\b', s)
    if m2:
        end_yr = 2000 + int(m2.group(1))
        start_yr = end_yr - 1
        base = f"fy{start_yr}-{str(end_yr)[2:]}"
        return f"{base} {sub_period}" if sub_period else base

    # Pattern: FY followed by 4 digits (e.g., fy2025 -> fy2024-25)
    m3 = re.search(r'\bfy\s*(\d{4})\b', s)
    if m3:
        end_yr = int(m3.group(1))
        start_yr = end_yr - 1
        base = f"fy{start_yr}-{str(end_yr)[2:]}"
        return f"{base} {sub_period}" if sub_period else base

    return s


# ---------------------------------------------------------------------------
# Conditions that get extracted from predicate names
_CONDITION_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r'_?pre[_-]?tax', re.IGNORECASE), "pre-tax"),
    (re.compile(r'_?after[_-]?tax', re.IGNORECASE), "after-tax"),
    (re.compile(r'_?post[_-]?tax', re.IGNORECASE), "after-tax"),
    (re.compile(r'_?restated', re.IGNORECASE), "restated"),
    (re.compile(r'_?excluding[_-]?esop', re.IGNORECASE), "excluding ESOP"),
    (re.compile(r'_?proforma', re.IGNORECASE), "pro-forma"),
    (re.compile(r'_?pro[_-]?forma', re.IGNORECASE), "pro-forma"),
    (re.compile(r'_?adjusted', re.IGNORECASE), "adjusted"),
    (re.compile(r'_?unaudited', re.IGNORECASE), "unaudited"),
    (re.compile(r'_?audited', re.IGNORECASE), "audited"),
]


def standardize_predicate(raw_predicate: str) -> tuple[str, str | None]:
    """Standardize a financial predicate and extract qualifying conditions.

    Extracts conditions (e.g. pre-tax, adjusted, restated) and returns (clean_predicate, condition).
    Metric alignment across different naming variants is handled dynamically by Voyage embeddings.

    Examples:
        "operating_profit_pre_tax"  → ("operating_profit", "pre-tax")
        "net_profit_after_tax"      → ("net_profit", "after-tax")
        "revenue_from_contracts"    → ("revenue_from_contracts", None)
        "total_expenses"            → ("total_expenses", None)
        "pin_code_reach"            → ("pin_code_reach", None)
    """
    if not raw_predicate:
        return raw_predicate, None

    pred = raw_predicate.strip().lower()

    # Extract conditions from predicate name
    extracted_condition = None
    clean_pred = pred
    for pattern, condition in _CONDITION_PATTERNS:
        if pattern.search(clean_pred):
            extracted_condition = condition
            clean_pred = pattern.sub('', clean_pred).strip('_- ')
            break

    return clean_pred, extracted_condition

