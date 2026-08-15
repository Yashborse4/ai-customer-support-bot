"""Security utilities for PII masking and unmasking in conversational data."""

import re
import logging
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)

class PIISecurityGuard:
    """Utility class to scan, mask, and unmask PII (Personally Identifiable Information).

    Supports masking of emails, phone numbers, credit card numbers, SSNs, and IP addresses.
    """
    # Regex patterns for common PII categories, ordered from most specific to least specific
    PATTERNS = {
        "EMAIL": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]*[a-zA-Z0-9]",
        "CREDIT_CARD": r"\b(?:\d[ -]*?){13,16}\b",
        "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
        "IP_ADDRESS": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
        # Phone matching is placed last and supports international, 10-digit, and 7-digit formats
        "PHONE": r"\+?\b(?:\d{1,4}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b|\b\d{3}[-.\s]?\d{4}\b"
    }

    def __init__(self) -> None:
        """Initializes PIISecurityGuard and compiles regex patterns."""
        self.compiled_patterns = {k: re.compile(v) for k, v in self.PATTERNS.items()}

    def mask(self, text: str, existing_map: Optional[Dict[str, str]] = None) -> Tuple[str, Dict[str, str]]:
        """Scans input text, replaces PII with placeholders, and returns masked text and mapping.

        Reuses existing placeholders for values already present in the existing_map.

        Args:
            text: The raw input string containing potential PII.
            existing_map: Optional pre-existing mapping of placeholders to raw values.

        Returns:
            A tuple of (masked_text, updated_mapping).
        """
        if not text:
            return text, existing_map or {}

        mapping = dict(existing_map) if existing_map else {}
        # Reverse mapping to easily look up if a raw value already has a placeholder
        reverse_mapping = {v: k for k, v in mapping.items()}
        
        masked_text = text

        for key, pattern in self.compiled_patterns.items():
            matches = list(set(pattern.findall(masked_text)))
            # Sort matches by length descending to prevent sub-string replacement collisions
            matches.sort(key=len, reverse=True)

            for match in matches:
                # Skip if matched value is empty
                if not match.strip():
                    continue

                if match in reverse_mapping:
                    placeholder = reverse_mapping[match]
                else:
                    # Determine the next index for this category
                    prefix = f"__[MASKED_{key}_"
                    category_indices = [
                        int(k.replace(prefix, "").replace("]__", ""))
                        for k in mapping.keys()
                        if k.startswith(prefix)
                    ]
                    next_idx = max(category_indices) + 1 if category_indices else 0
                    placeholder = f"{prefix}{next_idx}]__"
                    
                    mapping[placeholder] = match
                    reverse_mapping[match] = placeholder

                masked_text = masked_text.replace(match, placeholder)

        return masked_text, mapping

    def unmask(self, text: str, masking_map: Dict[str, str]) -> str:
        """Restores original PII values into masked text using the masking map.

        Args:
            text: The masked text.
            masking_map: Mapping of placeholders to original values.

        Returns:
            The unmasked text.
        """
        if not text or not masking_map:
            return text

        unmasked_text = text
        for placeholder, original in masking_map.items():
            unmasked_text = unmasked_text.replace(placeholder, original)
        return unmasked_text
