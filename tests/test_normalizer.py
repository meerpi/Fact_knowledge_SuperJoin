"""Tests for the symbolic numeric canonicalization module."""

import pytest
from app.normalizer import (
    parse_numeric_value,
    extract_currency,
    canonicalize_unit,
    standardize_predicate,
)


# ---------------------------------------------------------------------------
# parse_numeric_value
# ---------------------------------------------------------------------------

class TestParseNumericValue:
    """Test scale resolution and canonical value computation."""

    def test_plain_integer(self):
        base, scale, canon = parse_numeric_value("13,087")
        assert base == 13087.0
        assert scale is None
        assert canon == 13087.0

    def test_plain_decimal(self):
        base, scale, canon = parse_numeric_value("48,105.30")
        assert base == 48105.30
        assert scale is None
        assert canon == 48105.30

    def test_million_suffix(self):
        base, scale, canon = parse_numeric_value("$1,500 million")
        assert base == 1500.0
        assert scale == "million"
        assert canon == pytest.approx(1_500_000_000.0)

    def test_billion_suffix(self):
        base, scale, canon = parse_numeric_value("$1.5 billion")
        assert base == 1.5
        assert scale == "billion"
        assert canon == pytest.approx(1_500_000_000.0)

    def test_million_vs_billion_equivalence(self):
        """Core test: $1,500 million == $1.5 billion in canonical form."""
        _, _, canon_m = parse_numeric_value("$1,500 million")
        _, _, canon_b = parse_numeric_value("$1.5 billion")
        assert canon_m == pytest.approx(canon_b)

    def test_crore_suffix(self):
        base, scale, canon = parse_numeric_value("₹2.5 crore")
        assert base == 2.5
        assert scale == "crore"
        assert canon == pytest.approx(25_000_000.0)

    def test_lakh_suffix(self):
        base, scale, canon = parse_numeric_value("₹50 lakh")
        assert base == 50.0
        assert scale == "lakh"
        assert canon == pytest.approx(5_000_000.0)

    def test_accounting_negative_parentheses(self):
        base, scale, canon = parse_numeric_value("(8,911.39)")
        assert base == -8911.39
        assert scale is None
        assert canon == pytest.approx(-8911.39)

    def test_negative_sign(self):
        base, scale, canon = parse_numeric_value("-500.5")
        assert base == -500.5
        assert canon == pytest.approx(-500.5)

    def test_percentage(self):
        base, scale, canon = parse_numeric_value("14.5%")
        assert base == 14.5
        assert canon == pytest.approx(14.5)

    def test_mn_abbreviation(self):
        base, scale, canon = parse_numeric_value("₹48,105.30 mn")
        assert base == 48105.30
        assert scale == "mn"
        assert canon == pytest.approx(48_105_300_000.0)

    def test_bn_abbreviation(self):
        _, scale, canon = parse_numeric_value("$2.1 bn")
        assert scale == "bn"
        assert canon == pytest.approx(2_100_000_000.0)

    def test_cr_abbreviation(self):
        _, scale, canon = parse_numeric_value("Rs. 100 cr")
        assert scale == "cr"
        assert canon == pytest.approx(1_000_000_000.0)

    def test_non_numeric_text(self):
        base, scale, canon = parse_numeric_value("Not applicable")
        assert base is None
        assert scale is None
        assert canon is None

    def test_empty_string(self):
        assert parse_numeric_value("") == (None, None, None)

    def test_none_input(self):
        assert parse_numeric_value(None) == (None, None, None)

    def test_rupee_symbol_stripped(self):
        """Currency symbol should not prevent number parsing."""
        base, scale, canon = parse_numeric_value("₹48,105.30")
        assert base == 48105.30
        assert canon == pytest.approx(48105.30)

    def test_dollar_symbol_stripped(self):
        base, scale, canon = parse_numeric_value("$1,500")
        assert base == 1500.0
        assert canon == pytest.approx(1500.0)


# ---------------------------------------------------------------------------
# extract_currency
# ---------------------------------------------------------------------------

class TestExtractCurrency:
    def test_rupee_symbol(self):
        assert extract_currency("₹48,105.30") == "INR"

    def test_dollar_symbol(self):
        assert extract_currency("$1,500 million") == "USD"

    def test_euro_symbol(self):
        assert extract_currency("€100") == "EUR"

    def test_pound_symbol(self):
        assert extract_currency("£50") == "GBP"

    def test_rs_word(self):
        assert extract_currency("Rs. 500 crore") == "INR"

    def test_inr_word(self):
        assert extract_currency("INR 1000") == "INR"

    def test_no_currency(self):
        assert extract_currency("14.5%") is None

    def test_none_input(self):
        assert extract_currency(None) is None

    def test_plain_number(self):
        assert extract_currency("13087") is None


# ---------------------------------------------------------------------------
# canonicalize_unit
# ---------------------------------------------------------------------------

class TestCanonicalizeUnit:
    def test_inr_million(self):
        assert canonicalize_unit("INR million", "INR") == "INR"

    def test_rs_crore(self):
        assert canonicalize_unit("Rs. crore", "INR") == "INR"

    def test_percentage(self):
        assert canonicalize_unit("%", None) == "%"

    def test_locations(self):
        assert canonicalize_unit("locations", None) == "locations"

    def test_shipments(self):
        assert canonicalize_unit("shipments", None) == "shipments"

    def test_none_unit_with_currency(self):
        assert canonicalize_unit(None, "USD") == "USD"

    def test_none_both(self):
        assert canonicalize_unit(None, None) is None

    def test_inr_mn(self):
        """'INR mn' should strip the scale and return just 'INR'."""
        result = canonicalize_unit("INR mn", "INR")
        assert result == "INR"


# ---------------------------------------------------------------------------
# standardize_predicate
# ---------------------------------------------------------------------------

class TestStandardizePredicate:
    def test_operating_profit_pre_tax(self):
        pred, cond = standardize_predicate("operating_profit_pre_tax")
        assert pred == "operating_profit"
        assert cond == "pre-tax"

    def test_net_profit_after_tax(self):
        pred, cond = standardize_predicate("net_profit_after_tax")
        assert pred == "net_profit"
        assert cond == "after-tax"

    def test_revenue_from_contracts(self):
        pred, cond = standardize_predicate("revenue_from_contracts")
        assert pred == "revenue_from_contracts"
        assert cond is None

    def test_total_expenses(self):
        pred, cond = standardize_predicate("total_expenses")
        assert pred == "total_expenses"
        assert cond is None

    def test_non_financial_predicate_unchanged(self):
        pred, cond = standardize_predicate("pin_code_reach")
        assert pred == "pin_code_reach"
        assert cond is None

    def test_ebitda(self):
        pred, cond = standardize_predicate("ebitda")
        assert pred == "ebitda"
        assert cond is None

    def test_profit_before_tax(self):
        pred, cond = standardize_predicate("profit_before_tax")
        assert pred == "profit_before_tax"
        assert cond is None

    def test_loss_for_the_period(self):
        pred, cond = standardize_predicate("loss_for_the_period")
        assert pred == "loss_for_the_period"
        assert cond is None


    def test_restated_condition(self):
        pred, cond = standardize_predicate("revenue_restated")
        # "restated" should be extracted as condition
        assert cond == "restated"

    def test_empty_predicate(self):
        pred, cond = standardize_predicate("")
        assert pred == ""
        assert cond is None
