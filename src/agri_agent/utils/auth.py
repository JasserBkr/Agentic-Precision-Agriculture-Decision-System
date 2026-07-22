"""
Authentication helpers for the two EO data sources this project uses.

Every other module should import from here rather than re-implementing
auth. Credentials are read from environment variables (see .env.example),
never hardcoded.
"""

import os

from dotenv import load_dotenv
from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session

load_dotenv()

CDSE_TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE"
    "/protocol/openid-connect/token"
)


def get_cdse_session() -> OAuth2Session:
    """
    Return an authenticated OAuth2 session for the Copernicus Data Space
    Ecosystem (CDSE). Use this session for any request that needs your
    identity — e.g. downloading a Sentinel-2 asset. Note that STAC catalog
    *search* itself is public and doesn't require this token; you mainly
    need it for asset download.
    """
    client_id = os.environ.get("CDSE_CLIENT_ID")
    client_secret = os.environ.get("CDSE_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise RuntimeError(
            "CDSE_CLIENT_ID / CDSE_CLIENT_SECRET not set. "
            "Copy .env.example to .env and fill them in."
        )

    client = BackendApplicationClient(client_id=client_id)
    oauth = OAuth2Session(client=client)
    oauth.fetch_token(
        token_url=CDSE_TOKEN_URL,
        client_secret=client_secret,
        include_client_id=True,
    )
    return oauth


def init_earth_engine() -> None:
    """
    Initialize the Earth Engine Python API against the registered Cloud
    project. Requires having already run `ee.Authenticate()` once locally
    (opens a browser OAuth flow, stores a token on disk).
    """
    import ee

    project_id = os.environ.get("EE_PROJECT_ID")
    if not project_id:
        raise RuntimeError(
            "EE_PROJECT_ID not set. Copy .env.example to .env and fill it "
            "in with your registered Earth Engine Cloud project ID."
        )
    ee.Initialize(project=project_id)
