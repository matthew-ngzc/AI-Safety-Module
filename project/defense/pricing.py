"""Lightweight token and cost estimation helpers for defense experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil


TOKEN_ESTIMATION_METHOD = "chars_div_4_ceiling_v1"
PRICING_SOURCE = "static_estimated_pricing_v1"
DEFAULT_MODEL_PRICING = {
    "input_per_1m_tokens": 0.50,
    "output_per_1m_tokens": 1.50,
}
MODEL_PRICING_USD_PER_1M_TOKENS = {
    "deepseek-chat": {
        "input_per_1m_tokens": 0.27,
        "output_per_1m_tokens": 1.10,
    },
    "gemini-2.5-flash": {
        "input_per_1m_tokens": 0.35,
        "output_per_1m_tokens": 1.05,
    },
}


@dataclass
class ModelCallRecord:
    call_type: str
    model_name: str
    status: str
    input_chars: int
    output_chars: int
    est_input_tokens: int
    est_output_tokens: int
    est_cost_usd: float
    estimation_method: str = TOKEN_ESTIMATION_METHOD
    pricing_source: str = PRICING_SOURCE
    input_price_per_1m_tokens: float = DEFAULT_MODEL_PRICING["input_per_1m_tokens"]
    output_price_per_1m_tokens: float = DEFAULT_MODEL_PRICING["output_per_1m_tokens"]
    tokens_estimated: bool = True
    cost_estimated: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


MODEL_CALL_RECORD_FIELDS = list(ModelCallRecord.__dataclass_fields__.keys())


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, ceil(len(text) / 4))


def get_model_pricing(model_name: str) -> dict:
    return MODEL_PRICING_USD_PER_1M_TOKENS.get(model_name, DEFAULT_MODEL_PRICING)


def build_call_record(
    call_type: str,
    model_name: str,
    input_text: str,
    output_text: str,
    status: str,
) -> ModelCallRecord:
    input_text = input_text or ""
    output_text = output_text or ""
    pricing = get_model_pricing(model_name)
    est_input_tokens = estimate_tokens(input_text)
    est_output_tokens = estimate_tokens(output_text)
    est_cost_usd = (
        est_input_tokens * pricing["input_per_1m_tokens"]
        + est_output_tokens * pricing["output_per_1m_tokens"]
    ) / 1_000_000
    return ModelCallRecord(
        call_type=call_type,
        model_name=model_name,
        status=status,
        input_chars=len(input_text),
        output_chars=len(output_text),
        est_input_tokens=est_input_tokens,
        est_output_tokens=est_output_tokens,
        est_cost_usd=est_cost_usd,
        input_price_per_1m_tokens=pricing["input_per_1m_tokens"],
        output_price_per_1m_tokens=pricing["output_per_1m_tokens"],
    )


def normalize_call_record(record: ModelCallRecord | dict | None) -> ModelCallRecord | None:
    if record is None:
        return None
    if isinstance(record, ModelCallRecord):
        return record
    if not isinstance(record, dict):
        raise TypeError(f"Unsupported call record type: {type(record)!r}")

    normalized = {}
    for field in MODEL_CALL_RECORD_FIELDS:
        normalized[field] = record.get(field)

    return ModelCallRecord(
        call_type=str(normalized["call_type"]),
        model_name=str(normalized["model_name"]),
        status=str(normalized["status"]),
        input_chars=int(normalized["input_chars"]),
        output_chars=int(normalized["output_chars"]),
        est_input_tokens=int(normalized["est_input_tokens"]),
        est_output_tokens=int(normalized["est_output_tokens"]),
        est_cost_usd=float(normalized["est_cost_usd"]),
        estimation_method=str(normalized.get("estimation_method") or TOKEN_ESTIMATION_METHOD),
        pricing_source=str(normalized.get("pricing_source") or PRICING_SOURCE),
        input_price_per_1m_tokens=float(
            normalized.get("input_price_per_1m_tokens")
            or DEFAULT_MODEL_PRICING["input_per_1m_tokens"]
        ),
        output_price_per_1m_tokens=float(
            normalized.get("output_price_per_1m_tokens")
            or DEFAULT_MODEL_PRICING["output_per_1m_tokens"]
        ),
        tokens_estimated=bool(
            normalized.get("tokens_estimated")
            if normalized.get("tokens_estimated") is not None
            else True
        ),
        cost_estimated=bool(
            normalized.get("cost_estimated")
            if normalized.get("cost_estimated") is not None
            else True
        ),
    )


def prefixed_call_record_dict(prefix: str, record: ModelCallRecord | dict | None) -> dict:
    normalized = normalize_call_record(record)
    if normalized is None:
        return {f"{prefix}_{field}": None for field in MODEL_CALL_RECORD_FIELDS}
    return {f"{prefix}_{field}": value for field, value in normalized.to_dict().items()}


def aggregate_call_records(records: list[ModelCallRecord | dict | None]) -> dict:
    valid_records = [normalize_call_record(record) for record in records]
    valid_records = [record for record in valid_records if record is not None]
    return {
        "total_estimated_cost_usd": sum(record.est_cost_usd for record in valid_records),
        "total_estimated_input_tokens": sum(record.est_input_tokens for record in valid_records),
        "total_estimated_output_tokens": sum(record.est_output_tokens for record in valid_records),
        "total_model_calls": len(valid_records),
    }
