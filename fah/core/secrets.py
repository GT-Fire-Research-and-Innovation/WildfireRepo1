"""Keys the grab reads, and redaction of every text that gets published (manifests, logs)."""

from __future__ import annotations

import os
import re
from pathlib import Path
from urllib.parse import quote, quote_plus


class Secrets:
    """Secrets from the environment or an env file, and text made safe to publish."""

    PLACEHOLDERS = {
        "FIRMS_MAP_KEY": "{MAP_KEY}",
        "LANDFIRE_EMAIL": "{EMAIL}",
        "HF_TOKEN": "{HF_TOKEN}",
        "EARTHDATA_TOKEN": "{EARTHDATA_TOKEN}",
        "GEE_SERVICE_ACCOUNT_KEY": "{GEE_KEY}",
    }
    ENV_FILES = (Path(".env"), Path.home() / ".config" / "wildfire" / "env")

    @classmethod
    def lookup(cls, name: str) -> str | None:
        """The value of `name` from the environment, else from the first env file that sets it (last line wins
        within a file; a trailing '# comment' is dropped)."""
        value = os.environ.get(name)
        for path in cls.ENV_FILES:
            if value:
                break
            try:
                for line in path.read_text().splitlines():
                    if line.startswith(name + "="):
                        value = line.split("=", 1)[1].split("#")[0].strip()
            except OSError:
                continue
        return value or None

    @classmethod
    def _known_values(cls) -> dict[str, str]:
        """{value: placeholder} for every secret this process can see, as typed and as it appears inside URLs."""
        out = {}
        for name, placeholder in cls.PLACEHOLDERS.items():
            value = cls.lookup(name)
            if value and len(value) >= 4:
                for form in (value, quote(value, safe=""), quote_plus(value)):
                    out[form] = placeholder
        return out

    @classmethod
    def redact(cls, text: object) -> str:
        """Text safe to publish: no FIRMS map key, LANDFIRE email, HF or Earthdata token, Earth Engine key or
        signed-URL parameters, whether they appear as URL parts (also inside exception messages) or literally."""
        text = str(text or "")
        for value, placeholder in cls._known_values().items():
            text = text.replace(value, placeholder)
        text = re.sub(r"(/api/area/csv/)[^/\s]+/", r"\1{MAP_KEY}/", text)
        text = re.sub(r"([?&])Email=[^&\s]*&?", r"\1", text)
        text = re.sub(r"(Bearer\s+)[A-Za-z0-9._\-]+", r"\1{EARTHDATA_TOKEN}", text)
        text = re.sub(
            r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", "{GEE_KEY}", text, flags=re.S
        )
        text = re.sub(r'("private_key(?:_id)?"\s*:\s*")[^"]*"', r'\1{GEE_KEY}"', text)
        text = re.sub(r"(?<=[?&])(st|se|sp|sv|sr|skoid|sktid|skt|ske|sks|skv|sig|X-Amz-[A-Za-z]+)=[^&\s]*&?", "", text)
        return re.sub(r"[?&]+(?=\s|$)", "", text)
