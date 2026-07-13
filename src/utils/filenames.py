"""Filename charset-repair helpers.

Uploaded filenames sometimes arrive mis-decoded as cp437/cp1252 bytes
reinterpreted as UTF-8 (a multipart/form-data client-encoding quirk),
producing mojibake like "bela_m[mojibake]_ller_cv.pdf" instead of
"bela_muller_cv.pdf" (originally containing u-umlaut). This repairs that in one place instead of the
same try/except chain being copy-pasted at every call site.
"""

import logging

logger = logging.getLogger(__name__)

_REPAIR_ENCODINGS = ("cp437", "cp1252")


def repair_mojibake_filename(name: str) -> str:
    """Best-effort repair of a mis-decoded filename.

    Tries re-encoding the (wrongly-decoded) string back to bytes under each
    candidate legacy codepage, then decoding those bytes as UTF-8. Returns
    the original name unchanged if no repair round-trip succeeds.
    """
    for encoding in _REPAIR_ENCODINGS:
        try:
            return name.encode(encoding).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
    logger.debug(f"Could not repair filename encoding for {name!r}; using as-is")
    return name
