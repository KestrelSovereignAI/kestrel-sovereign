"""
Authentication endpoints: Google OAuth2, JWT email/password, API key info.

Provides:
- Browser-based Google sign-in with email allowlist (OAuth)
- Email/password login returning JWT tokens for API/dashboard access
- Works alongside existing API key auth

Auth priority: OAuth session → JWT Bearer → API key header
"""

import os
import re
import logging
import secrets
from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse, HTMLResponse, JSONResponse
from authlib.integrations.starlette_client import OAuth
from pydantic import BaseModel, EmailStr

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])

# --- OAuth setup ---

oauth = OAuth()


def _get_allowed_emails() -> set[str]:
    """Get the set of allowed email addresses from env.

    Accepts ``,`` or ``;`` as the separator. Cloud Run deploys via
    ``--update-env-vars ^,^...`` use ``,`` as the env-var separator,
    so commas inside any single value get misparsed as additional env
    vars (with ``@`` in the name → invalid POSIX → revision fails to
    start). Setting the secret with ``;`` avoids the collision; ``,``
    still works for local/dev configs.
    """
    raw = os.environ.get("KESTREL_ALLOWED_EMAILS", "")
    return {e.strip().lower() for e in re.split(r"[,;]", raw) if e.strip()}


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
        if proto not in ("http", "https"):
            proto = callback_url.scheme
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
    """Return current authenticated user info (from session or JWT)."""
    # Check session first
    email = request.session.get("user_email")
    if email:
        return {
            "email": email,
            "name": request.session.get("user_name", ""),
            "picture": request.session.get("user_picture", ""),
            "auth_method": "oauth",
        }

    # Check JWT Bearer token
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
        payload = _verify_jwt(token)
        if payload:
            return {
                "email": payload["sub"],
                "name": payload.get("name", ""),
                "picture": "",
                "auth_method": "jwt",
            }

    return JSONResponse({"detail": "Not authenticated"}, status_code=401)


# --- JWT Email/Password Auth ---

def _get_jwt_secret() -> str:
    """Get the JWT signing secret from env."""
    secret = os.environ.get("JWT_SECRET_KEY") or os.environ.get("KESTREL_SESSION_SECRET") or os.environ.get("KESTREL_API_KEY")
    if not secret:
        raise RuntimeError("No JWT_SECRET_KEY, KESTREL_SESSION_SECRET, or KESTREL_API_KEY set")
    return secret


def _create_jwt(email: str, name: str = "", expires_hours: int = 24) -> str:
    """Create a signed JWT token."""
    import jwt
    secret = _get_jwt_secret()
    payload = {
        "sub": email,
        "name": name,
        "iat": datetime.now(timezone.utc),
        "exp": datetime.now(timezone.utc) + timedelta(hours=expires_hours),
    }
    return jwt.encode(payload, secret, algorithm="HS256")


def _verify_jwt(token: str) -> dict | None:
    """Verify and decode a JWT token. Returns payload or None."""
    import jwt
    try:
        secret = _get_jwt_secret()
        return jwt.decode(token, secret, algorithms=["HS256"])
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
        return None


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


@router.post("/token")
async def login_token(body: LoginRequest):
    """Authenticate via email/password and return a JWT token.

    Password is verified against KESTREL_USER_PASSWORDS env var,
    which is a comma-separated list of email:password pairs.
    Falls back to checking the KESTREL_API_KEY as a universal password.
    """
    allowed_emails = _get_allowed_emails()
    email = body.email.lower()

    if allowed_emails and email not in allowed_emails:
        return JSONResponse({"detail": "Email not authorized"}, status_code=403)

    # Check user-specific passwords: "email1:pass1,email2:pass2"
    user_passwords = os.environ.get("KESTREL_USER_PASSWORDS", "")
    authenticated = False

    if user_passwords:
        for pair in user_passwords.split(","):
            pair = pair.strip()
            if ":" in pair:
                stored_email, stored_pass = pair.split(":", 1)
                if stored_email.strip().lower() == email and secrets.compare_digest(stored_pass.strip(), body.password):
                    authenticated = True
                    break

    # Fallback: KESTREL_API_KEY as universal password (for dev/staging)
    if not authenticated:
        api_key = os.environ.get("KESTREL_API_KEY", "")
        if api_key and secrets.compare_digest(api_key, body.password):
            authenticated = True

    if not authenticated:
        return JSONResponse({"detail": "Invalid credentials"}, status_code=401)

    token = _create_jwt(email)
    logger.info(f"JWT login: {email}")
    return {"access_token": token, "token_type": "bearer", "email": email}


@router.get("/verify")
async def verify_token(request: Request):
    """Verify a JWT token from the Authorization header."""
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return JSONResponse({"detail": "No bearer token"}, status_code=401)

    payload = _verify_jwt(auth_header[7:])
    if not payload:
        return JSONResponse({"detail": "Invalid or expired token"}, status_code=401)

    return {"valid": True, "email": payload["sub"], "name": payload.get("name", "")}
