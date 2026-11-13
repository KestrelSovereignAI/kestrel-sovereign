"""Headers for first-party OAuth shells with external, same-origin assets."""


OAUTH_BROWSER_HEADERS = {
    "Cache-Control": "no-store",
    "Referrer-Policy": "no-referrer",
    "X-Content-Type-Options": "nosniff",
    "Content-Security-Policy": "default-src 'none'; script-src 'self'; connect-src 'self'; "
        "style-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'",
}
