"""Unit tests for PII Masking and Data Security Guardrails."""

import pytest
from src.core.security import PIISecurityGuard

def test_pii_masking_and_unmasking() -> None:
    """Verify that PIISecurityGuard correctly masks and unmasks various PII types."""
    guard = PIISecurityGuard()
    
    raw_text = (
        "Hello, my email is alice.johnson@example.com and my phone is +1-555-019-2834. "
        "Please check my card: 4111-2222-3333-4444 and SSN 123-45-6789. "
        "Connecting from IP 192.168.1.50."
    )
    
    masked, mapping = guard.mask(raw_text)
    
    # Assertions on masked content
    assert "alice.johnson@example.com" not in masked
    assert "+1-555-019-2834" not in masked
    assert "4111-2222-3333-4444" not in masked
    assert "123-45-6789" not in masked
    assert "192.168.1.50" not in masked
    
    assert "__[MASKED_EMAIL_0]__" in masked
    assert "__[MASKED_PHONE_0]__" in masked
    assert "__[MASKED_CREDIT_CARD_0]__" in masked
    assert "__[MASKED_SSN_0]__" in masked
    assert "__[MASKED_IP_ADDRESS_0]__" in masked
    
    # Assertions on mapping dictionary
    assert mapping["__[MASKED_EMAIL_0]__"] == "alice.johnson@example.com"
    assert mapping["__[MASKED_PHONE_0]__"] == "+1-555-019-2834"
    assert mapping["__[MASKED_CREDIT_CARD_0]__"] == "4111-2222-3333-4444"
    assert mapping["__[MASKED_SSN_0]__"] == "123-45-6789"
    assert mapping["__[MASKED_IP_ADDRESS_0]__"] == "192.168.1.50"
    
    # Assertions on unmasking
    unmasked = guard.unmask(masked, mapping)
    assert unmasked == raw_text

def test_placeholder_reuse_and_turn_consistency() -> None:
    """Verify that existing placeholders are reused across turns to maintain consistency."""
    guard = PIISecurityGuard()
    
    # Turn 1
    text_1 = "My email is bob@example.com."
    masked_1, mapping_1 = guard.mask(text_1)
    assert "__[MASKED_EMAIL_0]__" in masked_1
    assert mapping_1["__[MASKED_EMAIL_0]__"] == "bob@example.com"
    
    # Turn 2: same email, new phone
    text_2 = "Send updates to bob@example.com or call 555-1234."
    masked_2, mapping_2 = guard.mask(text_2, mapping_1)
    
    # bob@example.com should keep index 0
    assert "__[MASKED_EMAIL_0]__" in masked_2
    # 555-1234 should get index 0 for phone
    assert "__[MASKED_PHONE_0]__" in masked_2
    assert mapping_2["__[MASKED_EMAIL_0]__"] == "bob@example.com"
    assert mapping_2["__[MASKED_PHONE_0]__"] == "555-1234"
    
    # Turn 3: new email
    text_3 = "Actually, use charlie@example.com."
    masked_3, mapping_3 = guard.mask(text_3, mapping_2)
    # charlie@example.com should get index 1
    assert "__[MASKED_EMAIL_1]__" in masked_3
    assert mapping_3["__[MASKED_EMAIL_1]__"] == "charlie@example.com"
    assert mapping_3["__[MASKED_EMAIL_0]__"] == "bob@example.com" # maintained

def test_streaming_buffer_splitting() -> None:
    """Verify the logic used in SSE streaming endpoint for buffering partial tokens."""
    # This matches the splitting logic in main.py event_generator
    
    # Case 1: normal text, odd splits count
    buf = "Hello, this is a response."
    parts = buf.split("__")
    assert len(parts) % 2 != 0
    
    # Case 2: partial placeholder forming, even splits count
    buf = "Check the email: __[MASKED"
    parts = buf.split("__")
    assert len(parts) % 2 == 0
    split_idx = buf.rfind("__")
    assert buf[:split_idx] == "Check the email: "
    assert buf[split_idx:] == "__[MASKED"
    
    # Case 3: complete placeholder, odd splits count
    buf = "The email is __[MASKED_EMAIL_0]__ here."
    # Simulated mapping replacement
    buf = buf.replace("__[MASKED_EMAIL_0]__", "alice@example.com")
    parts = buf.split("__")
    assert len(parts) % 2 != 0
    assert buf == "The email is alice@example.com here."
