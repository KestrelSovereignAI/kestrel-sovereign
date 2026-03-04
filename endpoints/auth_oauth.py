"""
Google OAuth2 authentication endpoints.

Provides browser-based Google sign-in with email allowlist.
Works alongside existing API key auth — OAuth for browsers, API keys for programmatic access.
"""

import os
import logging
from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse, HTMLResponse, JSONResponse
from authlib.integrations.starlette_client import OAuth

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])

# --- OAuth setup ---

oauth = OAuth()


def _get_allowed_emails() -> set[str]:
    """Get the set of allowed email addresses from env."""
    raw = os.environ.get("KESTREL_ALLOWED_EMAILS", "")
    return {e.strip().lower() for e in raw.split(",") if e.strip()}


def register_oauth(app):
    """Register OAuth with the app (call after app creation).

    Must be called at startup so authlib can read GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET
    from environment.  If the env vars are missing, OAuth endpoints will return 503.
    """
    client_id = os.environ.get("GOOGLE_CLIENT_ID")
    client_secret = os.environ.get("GOOGLE_CLIENT_SECRET")

    if client_id and client_secret:
        oauth.register(
            name="google",
            client_id=client_id,
            client_secret=client_secret,
            server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
            client_kwargs={"scope": "openid email profile"},
        )
        logger.info("Google OAuth configured")
    else:
        logger.warning("GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET not set — OAuth disabled")


# --- Endpoints ---


@router.get("/login")
async def login(request: Request):
    """Redirect to Google OAuth consent screen."""
    if "google" not in oauth._clients:
        return HTMLResponse(
            "<h2>OAuth not configured</h2><p>Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET.</p>",
            status_code=503,
        )

    # Build the callback URL from the current request
    redirect_uri = os.environ.get("KESTREL_OAUTH_REDIRECT_URI")
    if not redirect_uri:
        callback_url = request.url_for("callback")
        # Cloud Run terminates TLS, so the app sees HTTP internally.
        # Use X-Forwarded-Proto to build the correct HTTPS URL.
        proto = request.headers.get("x-forwarded-proto", callback_url.scheme)
        redirect_uri = str(callback_url.replace(scheme=proto))

    return await oauth.google.authorize_redirect(request, redirect_uri)


@router.get("/callback")
async def callback(request: Request):
    """Handle Google OAuth callback."""
    if "google" not in oauth._clients:
        return HTMLResponse("<h2>OAuth not configured</h2>", status_code=503)

    try:
        token = await oauth.google.authorize_access_token(request)
    except Exception as exc:
        logger.error(f"OAuth token exchange failed: {exc}")
        return HTMLResponse(
            "<h2>Authentication failed</h2><p>Could not complete Google sign-in.</p>",
            status_code=400,
        )

    userinfo = token.get("userinfo")
    if not userinfo:
        return HTMLResponse("<h2>Authentication failed</h2><p>No user info returned.</p>", status_code=400)

    email = userinfo.get("email", "").lower()
    allowed = _get_allowed_emails()

    if allowed and email not in allowed:
        logger.warning(f"OAuth login denied for {email} (not in allowlist)")
        return HTMLResponse(
            f"<h2>Access Denied</h2>"
            f"<p>{email} is not authorized to access this application.</p>"
            f'<p><a href="/auth/logout">Try a different account</a></p>',
            status_code=403,
        )

    # Set session
    request.session["user_email"] = email
    request.session["user_name"] = userinfo.get("name", "")
    request.session["user_picture"] = userinfo.get("picture", "")

    logger.info(f"OAuth login: {email}")
    return RedirectResponse(url="/", status_code=302)


@router.get("/logout")
async def logout(request: Request):
    """Clear session and redirect to login."""
    email = request.session.get("user_email", "unknown")
    request.session.clear()
    logger.info(f"OAuth logout: {email}")
    return RedirectResponse(url="/auth/login", status_code=302)


@router.get("/me")
async def me(request: Request):
    """Return current authenticated user info (from session)."""
    email = request.session.get("user_email")
    if not email:
        return JSONResponse({"detail": "Not authenticated"}, status_code=401)

    return {
        "email": email,
        "name": request.session.get("user_name", ""),
        "picture": request.session.get("user_picture", ""),
    }
