# YouTube API audit: draft answers

Until this audit passes, YouTube makes every video uploaded through your Google Cloud project **private**.
Submit the form at https://support.google.com/youtube/contact/yt_api_form ("YouTube API Services - Audit and Quota Extension Form").
Be truthful. Edit anything that doesn't match your situation.

**API client name:** Six Star Clips Uploader (internal tool)

**Who uses the client:** Only the channel owner. It's a private, single-user tool and is not offered to other users.

**What the client does:**
It publishes short video clips to my own YouTube channel. The clips come from livestreams whose creators run
public clipping campaigns that authorise clippers to repost and edit their streams. The tool:
1. renders a clip and adds captions,
2. uploads it to my channel with `videos.insert` (OAuth, `youtube.upload` scope),
3. reads my own uploads' public view counts with `videos.list` so I can report views to the campaign.

It doesn't download YouTube content through the API, collect data about other users, change other channels or post comments.

**API methods used:** videos.insert, videos.list (own videos only), channels.list (mine=true).

**Expected volume:** About 5 to 15 uploads a day, well under the default quota. No quota increase is requested; this audit only asks to lift the private-only restriction.

**Data storage:** No YouTube API data is stored except my own video IDs and view counts, kept in a private state file and refreshed within 30 days.

**Compliance:** Uploads are original edits (cuts, captions, layout) of footage the creators authorise for clipping, with the paid-promotion flag set where a campaign is paid.

**Screenshots/screencast:** Record a 1-minute screen capture of a run log showing a render and upload, and a screenshot of the
OAuth consent screen. Google often asks for these.
