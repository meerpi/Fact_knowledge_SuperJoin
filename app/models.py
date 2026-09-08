"""Data models for the fact knowledge layer."""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# iXBRL-pattern enums for structured unit/scale classification
# ---------------------------------------------------------------------------

class Dimension(str, Enum):
    """What kind of quantity a fact measures (iXBRL unit type category)."""
    MONETARY = "monetary"      # Currency amounts
    COUNT = "count"            # Discrete counts (locations, employees, customers)
    AREA = "area"              # Square feet, square meters
    PERCENTAGE = "percentage"  # Ratios expressed as %
    RATIO = "ratio"            # Dimensionless ratios (e.g. 1.5x)
    DURATION = "duration"      # Time periods (years, months, days)
    WEIGHT = "weight"          # Kilograms, tonnes
    VOLUME = "volume"          # Liters, cubic meters
    LENGTH = "length"          # Meters, km, miles
    OTHER = "other"            # Anything not covered above


KNOWN_DIMENSIONS: frozenset[str] = frozenset({d.value for d in Dimension})


class BoundingBox(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float


class TextBlock(BaseModel):
    """A chunk of text extracted from a PDF page."""
    content: str
    page: int
    block_type: str = "text"  # "text" or "table"
    bbox: BoundingBox | None = None


class TableBlock(BaseModel):
    """A table extracted from a PDF page, stored as list of rows."""
    headers: list[str]
    rows: list[list[str]]
    page: int
    bbox: BoundingBox | None = None


class PageData(BaseModel):
    """All extracted content from a single page."""
    page_number: int
    raw_text: str
    text_blocks: list[TextBlock] = Field(default_factory=list)
    tables: list[TableBlock] = Field(default_factory=list)


class DocumentData(BaseModel):
    """Complete parsed document."""
    doc_id: str
    filename: str
    page_count: int
    pages: list[PageData] = Field(default_factory=list)
    raw_text_index: dict[int, str] = Field(default_factory=dict)
    scanned_pages: list[int] = Field(default_factory=list, description="Pages identified as image-only scans (<20 characters text)")
    warnings: list[str] = Field(default_factory=list, description="Parser warnings (e.g. partially scanned pages)")


class Context(BaseModel):
    """Contextual qualifiers for a fact."""
    temporal: str | None = Field(default=None, description="Fiscal period or date (e.g. 'FY2021', 'Nine months ended Dec 31, 2021')")
    scope: str | None = Field(default=None, description="Scope or entity boundary (e.g. 'Consolidated', 'Standalone', 'Spoton subsidiary')")
    conditions: str | None = Field(default=None, description="Accounting or qualifying conditions (e.g. 'Restated', 'Excluding ESOP')")


class Provenance(BaseModel):
    """Traceable provenance and evidence grounding for a fact."""
    doc_id: str = Field(default="", description="ID of source document")
    page: int | None = Field(default=None, description="Page number where quote appears")
    evidence_quote: str = Field(description="Exact verbatim quote from the text")
    verified: bool = Field(default=False, description="Whether the quote was found in the PDF")
    match_type: str = Field(default="unverified", description="'exact', 'normalized', 'fuzzy', or 'unverified'")


class Fact(BaseModel):
    """Structured 6-tuple: (Subject, Predicate, Value, Context, Confidence, Provenance)."""
    subject: str = Field(description="Subject entity or topic (e.g. 'Delhivery', 'Spoton')")
    predicate: str = Field(description="Attribute or relation (e.g. 'revenue_from_contracts', 'pin_code_reach')")
    value: str = Field(description="Raw text representation as stated in document")
    numeric_value: float | None = Field(default=None, description="Normalized float value if numeric")
    unit: str | None = Field(default=None, description="Legacy combined unit string (e.g. 'INR million'). Prefer base_unit + scale.")
    base_unit: str | None = Field(default=None, description="Fundamental unit without scale (e.g. 'INR', 'square feet', '%')")
    scale: int = Field(default=0, description="Power of 10 multiplier (iXBRL convention). 6=millions, 7=crores, 9=billions")
    dimension: Dimension | str | None = Field(default=None, description="Category of quantity: monetary, count, area, percentage, etc. (open string)")
    canonical_value: float | None = Field(default=None, description="Scale-resolved canonical value: numeric_value × 10^scale")
    canonical_unit: str | None = Field(default=None, description="Canonical unit after scale folded in (e.g. 'INR', 'USD', '%')")
    context: Context = Field(default_factory=Context, description="Temporal, scope, and conditional qualifiers")
    confidence: float = Field(default=0.0, description="Verification confidence score between 0.0 and 1.0")
    provenance: Provenance = Field(description="Source document, page, quote, and verification metadata")


class RawFactItem(BaseModel):
    """Item emitted by the LLM before grounding. Uses iXBRL-pattern structured unit/scale."""
    subject: str = Field(description="Subject entity or topic")
    predicate: str = Field(description="Attribute or relation in snake_case")
    value: str = Field(description="Raw text representation as shown in document")
    numeric_value: float | None = Field(default=None, description="Normalized float if numeric, null otherwise")
    base_unit: str | None = Field(default=None, description="Fundamental unit WITHOUT scale prefix (e.g. 'INR' not 'INR million', 'square feet' not 'million square feet', '%', 'employees')")
    scale: int = Field(default=0, description="Power of 10 multiplier from table header/context. 0=ones, 3=thousands, 5=lakhs, 6=millions, 7=crores, 9=billions")
    dimension: Dimension | str | None = Field(default=None, description="Category: monetary, count, area, percentage, ratio, duration, weight, volume, length, custom string, or other")
    temporal: str | None = Field(default=None, description="Time period or date")
    scope: str | None = Field(default=None, description="Scope (e.g. 'Consolidated', 'Standalone')")
    conditions: str | None = Field(default=None, description="Qualifying conditions (e.g. 'Restated', 'Excluding ESOP')")
    evidence_quote: str = Field(description="Verbatim quote from text supporting this fact")
    page_number: int | None = Field(default=None, description="Source page number (0-indexed or displayed) where this fact appears")


class RawFactExtraction(BaseModel):
    """Schema expected from the Gemini structured output."""
    facts: list[RawFactItem] = Field(default_factory=list, description="List of extracted raw facts")


class ExtractedFacts(BaseModel):
    """Result container for facts extracted from a document or page."""
    doc_id: str
    page_number: int | None = None
    facts: list[Fact] = Field(default_factory=list)
    model_used: str = ""
    fallback_attempts: int = 0
    skipped_pages: list[int] = Field(default_factory=list, description="Pages skipped due to lack of text layer or being image-only")
    warnings: list[str] = Field(default_factory=list, description="Extraction warnings and partial-scan notices")


# ---------------------------------------------------------------------------
# Claim Graph models (ArbGraph / AttestDB pattern)
# ---------------------------------------------------------------------------

class EdgeType(str, Enum):
    """Typed relationship between two claims in the evidence graph."""
    CONTRADICTS = "contradicts"
    CORROBORATES = "corroborates"
    SUPERSEDES = "supersedes"  # temporal: FY22 data supersedes FY21


class ClaimEdge(BaseModel):
    """A directed edge between two facts in the claim graph."""
    source_fact_idx: int = Field(description="Index of the source fact")
    target_fact_idx: int = Field(description="Index of the target fact")
    edge_type: EdgeType
    detection_method: str = Field(description="'symbolic_numeric', 'symbolic_predicate', 'nli_deberta', 'temporal'")
    confidence: float = Field(description="Detection confidence (1.0 for symbolic, model score for NLI)")
    explanation: str = Field(description="Human-readable explanation of why this edge exists")


class ContradictionReport(BaseModel):
    """Result of contradiction detection across a set of facts."""
    doc_id: str
    cross_doc_id: str | None = None
    total_facts: int
    candidate_pairs_evaluated: int
    edges: list[ClaimEdge] = Field(default_factory=list)
    summary: dict = Field(default_factory=dict, description="Counts by edge_type")


# ---------------------------------------------------------------------------
# Claim Graph models — ArbGraph cluster-arbitration layer
# ---------------------------------------------------------------------------

class CaseType(str, Enum):
    """The four cases required by the Superjoin assignment."""
    CORROBORATED = "corroborated"                   # Case 1: same fact confirmed across docs
    CONTRADICTED = "contradicted"                    # Case 2: genuine conflict
    RECONCILED_TEMPORAL = "reconciled_temporal"       # Case 3a: explained by different time periods
    RECONCILED_SCOPE = "reconciled_scope"             # Case 3b: explained by Consolidated vs Standalone
    RECONCILED_UNIT = "reconciled_unit"               # Case 3c: explained by different units/scales
    RECONCILED_CONDITIONS = "reconciled_conditions"   # Case 3d: explained by different conditions
    EXTRACTION_FAILURE = "extraction_failure"         # Case 4: pipeline failure detected and handled


class DisputeCode(str, Enum):
    """Fine-grained dispute reason codes for contradictions and reconciliations.

    Provides machine-readable taxonomy tags that explain *why* two facts
    differ, enabling automated triage, UI color-coding, and spreadsheet
    export filtering.
    """
    # Genuine disputes (Case 2)
    DISPUTE_GENUINE_CONFLICT = "DISPUTE_GENUINE_CONFLICT"      # Same context, irreconcilable values
    DISPUTE_SIGN_MISMATCH = "DISPUTE_SIGN_MISMATCH"            # Positive vs negative (e.g. profit vs loss)
    DISPUTE_ORDER_OF_MAGNITUDE = "DISPUTE_ORDER_OF_MAGNITUDE"  # >10x difference (likely scale error)

    # Reconcilable disputes (Case 3)
    DISPUTE_TEMPORAL_DRIFT = "DISPUTE_TEMPORAL_DRIFT"          # Different fiscal periods (FY22 vs FY23)
    DISPUTE_UNIT_MISMATCH = "DISPUTE_UNIT_MISMATCH"            # Different units (USD vs EUR, millions vs crores)
    DISPUTE_SCOPE_DIFFERENCE = "DISPUTE_SCOPE_DIFFERENCE"      # Consolidated vs Standalone vs Subsidiary
    DISPUTE_ACCOUNTING_BASIS = "DISPUTE_ACCOUNTING_BASIS"      # GAAP vs Non-GAAP, pre-tax vs after-tax
    DISPUTE_ROUNDING = "DISPUTE_ROUNDING"                      # Values differ by <1% (rounding artifact)

    # Agreement
    AGREEMENT_EXACT = "AGREEMENT_EXACT"                        # Identical values
    AGREEMENT_APPROXIMATE = "AGREEMENT_APPROXIMATE"            # Within numeric tolerance

    # Unresolved
    UNRESOLVED = "UNRESOLVED"                                  # Could not determine reason

    # Extensible / Custom
    DISPUTE_CUSTOM = "DISPUTE_CUSTOM"                          # Custom or domain-specific dispute category


class EvidenceEntry(BaseModel):
    """Side-by-side evidence from a single document for a cluster."""
    doc_id: str
    doc_filename: str
    page: int | None = None
    value: str
    canonical_value: float | None = None
    canonical_unit: str | None = None
    temporal: str | None = None
    scope: str | None = None
    conditions: str | None = None
    evidence_quote: str
    confidence: float
    match_type: str


class FactCluster(BaseModel):
    """A group of facts from one or more documents that refer to the same assertion.

    This is the core ArbGraph 'claim subgraph': aligned facts + typed edges +
    a case classification + human-readable explanation.
    """
    cluster_id: str = Field(description="Unique cluster identifier")
    subject: str = Field(description="Canonical entity name for this cluster")
    predicate: str = Field(description="Canonical metric name for this cluster")
    fact_indices: list[int] = Field(description="Indices into the global fact pool")
    edges: list[ClaimEdge] = Field(default_factory=list, description="Edges within this cluster")
    case_type: CaseType = Field(description="Which of the 4 assignment cases this cluster represents")
    dispute_code: str = Field(default="UNRESOLVED", description="Fine-grained dispute reason code from DisputeCode taxonomy")
    dispute_detail: str | None = Field(default=None, description="Optional granular explanation or custom category details")
    evidence: list[EvidenceEntry] = Field(default_factory=list, description="Side-by-side evidence from each document")
    explanation: str = Field(default="", description="Human-readable reasoning for the case classification")
    credibility_scores: dict[int, float] = Field(default_factory=dict, description="ArbGraph credibility score per fact index")
    consensus_value: float | None = Field(default=None, description="Most credible canonical value (if applicable)")
    consensus_unit: str | None = Field(default=None, description="Unit of the consensus value")
    doc_count: int = Field(default=1, description="Number of distinct documents in this cluster")


class ExtractionFailure(BaseModel):
    """A detected extraction or reasoning failure (Case 4)."""
    failure_type: str = Field(description="Type: 'unverified_quote', 'scale_mismatch', 'empty_page', 'scale_in_unit', 'missing_dimension'")
    fact_index: int | None = Field(default=None, description="Index of the problematic fact, if applicable")
    doc_id: str = Field(default="", description="Document ID")
    page: int | None = Field(default=None, description="Page number")
    description: str = Field(description="What went wrong")
    mitigation: str = Field(description="How the system handled or would improve it")


class ClaimGraph(BaseModel):
    """The complete cross-document fact knowledge layer.

    Architecture: ArbGraph-inspired claim alignment → evidence graph → cluster arbitration.
    """
    documents: dict[str, str] = Field(description="doc_id → filename mapping")
    total_facts: int
    clusters: list[FactCluster] = Field(default_factory=list)
    extraction_failures: list[ExtractionFailure] = Field(default_factory=list, description="Case 4 examples")
    unmatched_facts: list[int] = Field(default_factory=list, description="Fact indices that did not align with any other fact")
    case_summary: dict[str, int] = Field(default_factory=dict, description="Counts by CaseType")
    pipeline_metadata: dict = Field(default_factory=dict, description="Timing, model info, etc.")
