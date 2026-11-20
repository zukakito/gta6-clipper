# GTA 6 Clipping Pipeline

This pipeline runs every 30 minutes on free GitHub Actions. Each run does the following:

1. **Scout.** Reads public campaign boards (clipcpm.com, gtaclipper.com, Content Rewards discover) for new GTA 6 campaigns and records them in `state/discovered.json`.
2. **Watch.** Pulls recent VODs from every enabled campaign's authorised source channels (Twitch, Kick or YouTube).
3. **Find moments.** Scores chat-replay spikes and audio loudness to find the hype moments. It doesn't need to download the full video for this.
4. **Cut and caption.** Downloads only those moments and transcribes them with Whisper. Gemini picks the exact cut and writes the hook, title and description from the campaign brief. A clip that breaks the rules is rejected.
5. **Render.** Makes a 1080×1920 Short: the full frame on a blurred background (stream overlays stay visible), word-by-word captions, the streamer's name, and a "Paid partnership" disclosure for the first 3.5 s. Campaigns marked `format: long` get 16:9 compilations instead.
6. **Upload** to YouTube, respecting daily caps and spacing between uploads.
7. **Submit** the link to the campaign through the Whop API, or queue it in `state/pending_submissions.json`.
8. **Commit state** back to the repo so the next run picks up where this one stopped. Rendered clips are also saved as workflow artifacts for 7 days.

## Setup (one time)

| # | What | Where it goes |
|---|---|---|
| 1 | Create a **public** repo from this folder. Actions minutes are unlimited on public repos, and secrets stay private. | GitHub |
| 2 | Get a Gemini API key from aistudio.google.com (free) | Secret `GEMINI_API_KEY` |
| 3 | In Google Cloud: enable YouTube Data API v3, set the OAuth consent screen to External and **In production**, and create a **Desktop** OAuth client. Then run `python tools/get_youtube_token.py client_secret.json` and choose the clips channel. | Secrets `YT_CLIENT_ID`, `YT_CLIENT_SECRET`, `YT_REFRESH_TOKEN` |
| 4 | Submit the API audit (draft answers in `docs/youtube_api_audit.md`). Until it passes, API uploads are **private**. | Google form |
| 5 | Optional stop-gap: an Upload-Post free account (10 public uploads a month) with your channel connected. | Secrets `UPLOAD_POST_API_KEY`, `UPLOAD_POST_USER`; set `publisher: upload_post` |
| 6 | Whop: join each campaign. Connect your YouTube channel under Settings → Connected Accounts, and set up payouts. | whop.com |
| 7 | Optional auto-submit: a Whop OAuth user token with the `bounty:submission:create` scope, plus `whop_bounty_id` on each campaign. | Secret `WHOP_USER_TOKEN` |
| 8 | Optional: cookies.txt from a spare YouTube account if YouTube blocks VOD downloads ("confirm you're not a bot"). Use a spare account, not your channel's. | Secret `YTDLP_COOKIES` |

Start with `publisher: dry_run` in `config/settings.yaml`. Run the workflow by hand (Actions → gta6-clipper → Run workflow), download the clips artifact and check them. Then switch to `youtube_api` or `upload_post`.

## Daily operation

- **Add or enable campaigns** in `config/campaigns.yaml`. Each needs `sources`, `must_include` and `rules` copied from the brief.
- **Check `state/status.json`** for the last run, uploads in the last 24 h, queued moments and pending submissions.
- **Submit anything in `state/pending_submissions.json`** before its `deadline`. Many campaigns require submission within 30 minutes of posting.

## Rules this pipeline enforces (campaigns ban accounts for less)

- Only footage from each campaign's listed sources is clipped. No other creators' content is used.
- No logos, no view botting, no reposting the same moment, and no AI-generated video. Clips are real stream footage; only the captions and edit are automated.
- A paid-partnership disclosure appears on screen, in the description, and in YouTube's paid-promotion flag.
- Uploads are capped per day and spaced apart, so a new channel doesn't look like spam.
