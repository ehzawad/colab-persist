"""Google identity, private per-account credentials and quota projects.

Never changes the global gcloud login. Credentials stay on this computer; the VM
only receives short-lived access tokens narrowed to Drive files this tool created.
"""
import calendar
import hashlib
import json
import logging
import os
from pathlib import Path
import time
import warnings

import google.auth
from google.auth.transport.requests import AuthorizedSession, Request
from google.oauth2.credentials import Credentials

COLAB_SCOPE = "https://www.googleapis.com/auth/colaboratory"
DRIVE_FILE_SCOPE = "https://www.googleapis.com/auth/drive.file"
DRIVE_FULL_SCOPE = "https://www.googleapis.com/auth/drive"
BASE_SCOPES = ("openid", "https://www.googleapis.com/auth/cloud-platform",
               "https://www.googleapis.com/auth/userinfo.email", COLAB_SCOPE)
SCOPES = BASE_SCOPES + (DRIVE_FILE_SCOPE,)
USERINFO = "https://openidconnect.googleapis.com/v1/userinfo"


def normalize_email(email):
    if not isinstance(email, str) or email.count("@") != 1 or any(c.isspace() for c in email):
        raise ValueError("Provide the Google email address you use for Colab and Drive.")
    local, domain = email.split("@")
    if not local or not domain or email.startswith("-"):
        raise ValueError("Provide the Google email address you use for Colab and Drive.")
    return email.lower()


def login_scopes(full_drive=False):
    """drive.file works with gcloud's own client; full Drive needs the user's OAuth client."""
    return BASE_SCOPES + ((DRIVE_FULL_SCOPE,) if full_drive else (DRIVE_FILE_SCOPE,))


def account_dir(config_dir: Path, email: str) -> Path:
    """An account's private directory, named without the email address."""
    normalized = normalize_email(email)
    account_id = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    root = Path(config_dir) / "accounts"
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    root.chmod(0o700)
    directory = root / account_id
    directory.mkdir(exist_ok=True, mode=0o700)
    directory.chmod(0o700)
    return directory


def _private_write(path: Path, value: dict):
    temporary = path.with_name(path.name + ".partial")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w") as target:
        json.dump(value, target, indent=2, sort_keys=True)
        target.write("\n")
        target.flush()
        os.fsync(target.fileno())
    os.replace(temporary, path)


def meta(config_dir, email) -> dict:
    path = account_dir(config_dir, email) / "meta.json"
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return {}


def save_meta(config_dir, email, **values) -> dict:
    path = account_dir(config_dir, email) / "meta.json"
    merged = {**meta(config_dir, email), **values}
    _private_write(path, merged)
    return merged


def credentials_path(cfg):
    value = cfg.get("credentials_file")
    return Path(value).expanduser().absolute() if value else None


def _load(path, scopes):
    logging.getLogger("google.auth._default").setLevel(logging.ERROR)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"Your application has authenticated using end user credentials.*",
                                category=UserWarning)
        credentials, _ = google.auth.load_credentials_from_file(str(path), scopes=list(scopes))
    return credentials


def verified_credentials(cfg):
    """Load the selected account's private credentials and confirm their Google identity."""
    expected = normalize_email(cfg["expected_email"])
    path = credentials_path(cfg)
    if path is None or not path.is_file():
        raise RuntimeError("This account has no private colab-persist login yet. Run `colab-persist login --email "
                           + expected + " --reauth` once; no runtime was accessed.")
    try:
        credentials = _load(path, cfg.get("scopes") or SCOPES)
        with AuthorizedSession(credentials) as session:
            response = session.get(USERINFO, timeout=20)
            response.raise_for_status()
            email = response.json().get("email", "").lower()
    except Exception as error:
        raise RuntimeError("Could not verify Google credentials. Run `colab-persist login --email "
                           + expected + " --reauth` to sign in again.") from error
    if email != expected:
        raise RuntimeError(f"Google account mismatch: expected {expected}, got {email or 'no email'}. "
                           "No runtime was accessed. Use `colab-persist login --email "
                           + expected + " --reauth`.")
    return credentials


def drive_token(cfg, quota_project, folder, *, full_drive=False):
    """Mint a short-lived access token narrowed to Drive for the VM. No refresh token leaves here."""
    path = credentials_path(cfg)
    if path is None or not path.is_file():
        raise RuntimeError("Saved account credentials are missing. Run `colab-persist login --reauth`.")
    info = json.loads(path.read_text())
    scope = DRIVE_FULL_SCOPE if full_drive else DRIVE_FILE_SCOPE
    try:
        narrowed = Credentials.from_authorized_user_info(info, scopes=[scope])
        narrowed.refresh(Request())
    except Exception as error:
        raise RuntimeError("Google did not issue a Drive token for this login. It probably predates "
                           "colab-persist 0.4. Run `colab-persist login --email "
                           + normalize_email(cfg["expected_email"]) + " --reauth` once.") from error
    # google-auth reports expiry as a naive UTC datetime.
    expiry = calendar.timegm(narrowed.expiry.timetuple()) if narrowed.expiry else time.time() + 3000
    return {"token": narrowed.token, "quota_project": quota_project, "expires_at": int(expiry),
            "folder": folder}
