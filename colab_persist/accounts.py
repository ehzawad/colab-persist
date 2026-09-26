"""Google identity and private ADC credentials; never changes global gcloud login."""
import logging
import os
from pathlib import Path
import warnings

import google.auth
from google.auth.transport.requests import AuthorizedSession
from colab_cli.auth import _get_adc_credentials

SCOPES = (
    "openid", "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/colaboratory",
)


def normalize_email(email):
    if not isinstance(email, str) or email.count("@") != 1 or any(c.isspace() for c in email):
        raise ValueError("Provide the Google email address you use for Colab and Drive.")
    local, domain = email.split("@")
    if not local or not domain or email.startswith("-"):
        raise ValueError("Provide the Google email address you use for Colab and Drive.")
    return email.lower()


def credentials_path(cfg):
    value = cfg.get("credentials_file")
    return Path(value).expanduser().absolute() if value else None


def environment(cfg):
    """Explicit private ADC wins over any unrelated shell credential override."""
    result = os.environ.copy()
    path = credentials_path(cfg)
    if path:
        if not path.is_file():
            raise RuntimeError("Saved account credentials are missing. Run `colab-persist login --email "
                               + normalize_email(cfg["expected_email"]) + " --reauth`.")
        result["GOOGLE_APPLICATION_CREDENTIALS"] = str(path)
    return result


def verified_credentials(cfg):
    expected = normalize_email(cfg["expected_email"])
    logging.getLogger("colab_cli.auth").setLevel(logging.ERROR)
    logging.getLogger("google.auth._default").setLevel(logging.ERROR)
    path = credentials_path(cfg)
    try:
        if path:
            # Match the official CLI's narrow suppression for user ADC without
            # a quota project. Colab does not require one for this login.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"Your application has authenticated using end user credentials.*",
                    category=UserWarning,
                )
                credentials, _ = google.auth.load_credentials_from_file(str(path), scopes=list(SCOPES))
        else:
            credentials = _get_adc_credentials()
        with AuthorizedSession(credentials) as session:
            response = session.get("https://openidconnect.googleapis.com/v1/userinfo", timeout=20)
            response.raise_for_status()
            email = response.json().get("email", "").lower()
    except (Exception, SystemExit) as error:
        raise RuntimeError("Could not verify Google credentials. Run `colab-persist login --email "
                           + expected + " --reauth` to sign in again.") from error
    if email != expected:
        raise RuntimeError(f"Google account mismatch: expected {expected}, got {email or 'no email'}. "
                           "No runtime was accessed. Use `colab-persist login --email "
                           + expected + " --reauth`.")
    return credentials
