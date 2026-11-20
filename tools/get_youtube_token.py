"""One-time: get a long-lived YouTube refresh token for your channel.

Run on your own computer (not in GitHub):
    pip install google-auth-oauthlib
    python tools/get_youtube_token.py path/to/client_secret.json

A browser opens -> pick the Google account that owns the clips channel -> pick the CHANNEL
(brand account) -> allow. Copy the three printed values into GitHub -> Settings -> Secrets
and variables -> Actions: YT_CLIENT_ID, YT_CLIENT_SECRET, YT_REFRESH_TOKEN.

Important: the OAuth consent screen must be "In production" (not "Testing"), otherwise
Google expires the refresh token after 7 days and uploads stop.
"""
import json
import sys

from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ["https://www.googleapis.com/auth/youtube.upload", "https://www.googleapis.com/auth/youtube.readonly"]

if len(sys.argv) != 2:
    sys.exit(__doc__)
flow = InstalledAppFlow.from_client_secrets_file(sys.argv[1], SCOPES)
creds = flow.run_local_server(port=0, prompt="consent", access_type="offline")
cfg = json.load(open(sys.argv[1]))
inst = cfg.get("installed") or cfg.get("web")
print("\nYT_CLIENT_ID     =", inst["client_id"])
print("YT_CLIENT_SECRET =", inst["client_secret"])
print("YT_REFRESH_TOKEN =", creds.refresh_token)
