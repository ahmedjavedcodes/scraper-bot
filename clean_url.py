import os
import re
from urllib.parse import parse_qsl, urlparse


def sanitize_and_generate_dynamic_folder(raw_url: str) -> tuple:
    """Cleans trailing whitespaces, fixes broken URL schemes, and generates a dynamic folder path."""
    if not raw_url or not isinstance(raw_url, str):
        return None, None

    # Clean whitespace and linebreaks
    sanitized_url = raw_url.strip().replace("\n", "").replace("\r", "")

    # Fix malformed protocol slashes (e.g., "https:/www..." -> "https://www...")
    sanitized_url = re.sub(r"^(https?):/+(?=[^/])", r"\1://", sanitized_url)

    parsed = urlparse(sanitized_url)
    netloc = parsed.netloc.lower()

    # Generate a clean folder name
    query_dict = dict(parse_qsl(parsed.query, keep_blank_values=False))
    domain_clean = netloc.replace("www.", "")
    id_marker = (
        query_dict.get("id") or query_dict.get("oga_service") or "index"
    )

    # Safe folder slug conversion
    id_marker_clean = re.sub(r"[^a-zA-Z0-9_\-]", "_", str(id_marker))[
        :30
    ].strip("_")
    final_folder_name = f"{domain_clean}_{id_marker_clean}"

    BASE_DOWNLOAD_DIR = "downloads"  # Replace with your BASE_DOWNLOAD_DIR path
    unique_target_path = os.path.normpath(
        os.path.join(BASE_DOWNLOAD_DIR, final_folder_name)
    )

    return sanitized_url, unique_target_path