# x-news-bot

Telegram bot that turns your X/Twitter feed into a clustered topic digest. Pulls your actual For You and Following timelines, groups posts by semantic similarity, and lets Claude name each cluster and write a short summary — so you read ten topics instead of scrolling 400 posts.

Built around one principle: **respect the user's real X feed**. The bot doesn't flood your timeline with topic-search results or re-rank by keyword heuristics. What's in the cluster list is what's in your actual feed, grouped and summarised.

---

## what it does

- **📊 что обсуждают** — snapshot of your For You timeline over the last 20h, clustered into topics. Top 7 clusters are pre-summarised (Grok-Stories style) in parallel before the digest is rendered.
- **📰 моя лента** — same, but from your Following (chronological) timeline over 24h. Per-author cap of 8 keeps Reuters/Economist from eating 75% of the digest without silencing them entirely.
- **💬 хочу больше X** — free-form text → Claude picks real X handles for the topic, validates them via `get_author_info` (exists + ≥500 followers or verified), adds them to your tracked list, and immediately fetches their recent posts with a cosine-similarity gate (≥ 0.25 to the query anchor) so junk from topic-matching-but-unrelated accounts is dropped.
- **👍 / 👎** — updates a persistent preference vector and dampens disliked topics for the next 6 hours (≥ 5 dislikes excludes the named cluster).
- **dual-media** — when a post quotes another post that carries media, the bot sends two Telegram messages (author + quote) and tracks both message IDs so `← Назад` cleans up both.
- **/reset**, **⏸ pause**, custom delivery interval.

The overview shows a **flat list of real cluster names**, not synthetic super-categories. Super-topic grouping is off by default — abstract buckets like "IT/Технологии · 7" that mixed Bitcoin + XRP + Jensen Huang obscured what was actually in the feed.

---

## pipeline

```
telegram (aiogram)
    │
    ▼
handlers (report / feed / preferences / discussion / onboarding)
    │
    ▼
core/report.build_report
    ├── x_parser.get_home_timeline / get_for_you_timeline  (paginated, up to 400 posts)
    ├── bot-tracked author injection  (top 5 × 2 posts, most-recently-added)
    ├── per-author cap=8 (Following)  (keeps prolific sources from dominating)
    ├── filters.is_low_signal         (ads, hype, info-density, list-dumps,
    │                                  cross-platform promo, nsfw)
    ├── pending_boost_ids merge       (posts pre-fetched by «хочу больше X»,
    │                                  capped at 20 per report)
    ├── embeddings.embed_batch        (openai text-embedding-3-small)
    ├── cosine union-find clustering  (0.42 threshold → name-merge at 0.62)
    ├── ai_client.name_topic          (Claude Haiku, emoji + short name)
    └── ai_client.summarize_discussion (upfront for top 7, lazy for the rest)
    │
    ▼
delivery.send_one_tweet  (photo / video / animation / text + dual-media)
    │
    ▼
sqlite (tweets, users, sent_news, feedback, followed_authors)
```

---

## how ranking works

Clusters are ranked **by size, not by keyword match**. Formula inside `_cluster_and_name`:

```python
weighted = (min(cluster_size, 12) + 5) * (1.0 + boost * boost_multiplier)
```

With `boost_multiplier=0` (current default), this collapses to pure size ranking — whatever's actually dominant in your X feed wins. The Claude-based interest scorer still exists in `ai_client.score_clusters_against_interests`, but it's a no-op unless you raise the multiplier in code.

Why: a 12-post AI cluster from your real subscriptions should beat a 3-post fashion cluster just because you once typed "хочу больше моды". The user's X algorithm already knows what they want — we respect it.

Dislike dampening is the main learning signal: 5+ dislikes of a named cluster within 6 hours excludes it from the next report.

---

## «хочу больше X» without flooding

Typing `хочу больше про AI agents` does **not** trigger a 400-post X-search flood. Instead:

1. Claude parses intent into 2-3 short English queries (`"ai agents"`, `"agent orchestration"`, `"claude tools"`) and proposes 4-8 real X handles.
2. Handles are validated via `get_author_info` (must exist, ≥500 followers or verified) and added to `FollowedAuthor` with weight 2.0.
3. `_immediate_topic_fetch` runs: search + recent posts from the handles, all gated by cosine similarity ≥ 0.25 to the query anchor. Typical yield: 20-50 relevant posts.
4. Their IDs are stashed into `user.onboarding_payload["pending_boost_ids"]` — capped at 200.
5. The next report pulls the top 20 of those pending IDs into the clustering pool. The rest of the pool is still the user's real X timeline.

Result: the topic actually shows up in the next digest, but it doesn't bury AI/IT subscriptions that the user cares about.

---

## filters

Universal, author-agnostic:

- `too_short` / `hashtag_spam` / `all_caps` / `url_only` / `pure_retweet` / `too_old`
- `ad_marker_en` / `ad_marker_ru` — sponsored / promo / airdrop / DYOR
- `cross_platform_promo` / `cross_platform_promo_quote` — only universal markers: `links in bio`, `join our telegram/discord/whatsapp`, `t.me/...`, `follow me for more`
- `nsfw_body` / `nsfw_quote` — `onlyfans`, `nsfw`, `porn`, `xxx`, `18+` markers (body + quoted text)
- `list_dump` — Reuters-style multi-line bulleted headline dumps
- `hype:N.NN` — hype_score threshold (CAPS spam, emoji storms, "to the moon")
- `low_density:N.NN` — info-density floor (no numbers, no URL, no names, short)

No hardcoded author blocklists, no topic-specific regexes. Engagement floors (e.g. "0 likes after 2h = trash") were explicitly removed — they silenced legitimate posts from small AI/IT accounts that get 0 likes in the first few hours but carry real signal.

If you want to block a specific handle, the bot adds it to `user.blocked_authors` when you type "исключи @handle".

---

## rate-limit protection

X hammers 429 on read endpoints under active scraping. `core/x_parser.py` has:

- **`_X_API_SEM = asyncio.Semaphore(2)`** — one per process, caps concurrent GraphQL calls.
- **`_USER_LOOKUP_CACHE`** — 1h TTL on OK, 10min on 404/429. Same handle isn't re-queried in one report.
- **`_SEARCH_COOLDOWN`** — 10min cooldown on a query after 404/rate-limit.
- **`_TWEETS_COOLDOWN`** — handles that 429 on `get_user_tweets` get skipped.
- **`_shorten_query`** — Claude queries truncated to 3 words (longer phrases reliably 404 on SearchTimeline).
- Pagination on `get_home_timeline` with early-abort on failure — up to 5 pages (~200-400 posts) instead of a single 100-post snapshot.

---

## stack

- `python 3.12`
- `aiogram 3.x` — telegram bot
- `twikit` — X/Twitter GraphQL with cookie auth
- `anthropic` SDK — Claude Haiku for naming / summarisation, Sonnet for onboarding
- `openai` SDK — `text-embedding-3-small` (via ProxyAPI or direct)
- `sqlalchemy 2.x + aiosqlite`
- `apscheduler` — delivery-job every 20 min, implicit-decay every 6h
- `chromadb` — optional vector store

---

## setup

```bash
git clone https://github.com/sakenuGOD/x-news-bot.git
cd x-news-bot
python -m venv venv
venv/Scripts/activate          # windows
# source venv/bin/activate     # linux/mac
pip install -r requirements.txt
cp .env.example .env           # fill in tokens
python -m bot.main
```

### `.env`

```
TELEGRAM_BOT_TOKEN=<from @BotFather>
PROXYAPI_KEY=<proxyapi.ru or direct anthropic/openai>
ANTHROPIC_BASE_URL=https://api.proxyapi.ru/anthropic
OPENAI_BASE_URL=https://api.proxyapi.ru/openai/v1
MODEL_HAIKU=claude-haiku-4-5
MODEL_SONNET=claude-sonnet-4-5

X_AUTH_TOKEN=<cookie auth_token from devtools>
X_CT0=<optional — cookie ct0>

DATABASE_URL=sqlite+aiosqlite:///./bot.db
DEFAULT_DELIVERY_INTERVAL_HOURS=3
```

`auth_token` comes from your browser X session: DevTools → Application → Cookies → x.com → `auth_token`. Use a burner account — the bot hits internal GraphQL endpoints.

---

## structure

```
x-news-bot/
├── bot/
│   ├── main.py              # entrypoint — polling + scheduler
│   ├── delivery.py          # send_one_tweet, dual-media, cleanup
│   ├── keyboards.py         # inline keyboards
│   └── handlers/
│       ├── report.py        # «что обсуждают» / «моя лента»
│       ├── feed.py          # 👍 / 👎 / 🇷🇺 / 💬 under a post
│       ├── preferences.py   # «хочу больше X» — Claude + immediate fetch + similarity gate
│       ├── discussion.py    # reply-to-post via Claude
│       └── onboarding.py    # /start
├── core/
│   ├── x_parser.py          # twikit + TTL cache + semaphore + cooldowns + pagination
│   ├── ai_client.py         # all Claude calls (pydantic schemas)
│   ├── embeddings.py        # openai embeddings + nearest_cluster
│   ├── filters.py           # is_low_signal (trash / hype / density / promo / nsfw)
│   ├── report.py            # build_report: fetch → filter → embed → cluster → name → summary
│   ├── recommender.py       # score_tweet, pick_top_for_user
│   └── demo_data.py         # demo pool fallback
├── db/
│   ├── database.py          # async engine + in-place migrations
│   ├── models.py            # user, tweet, feedback, sent_news, followed_author
│   └── vector_store.py      # chroma
├── scheduler.py             # delivery-job + implicit-decay-job
├── config.py                # settings (env)
└── requirements.txt
```

---

## what's cached

- `tweet.embedding` — forever (1536-float JSON)
- `tweet.summary_ru` — lazy ru translation, cached in DB
- `cluster.summary` — upfront (top 7) + lazy (on topic open)
- `Report` — in-memory per-user
- `x_parser._USER_LOOKUP_CACHE` — 1h TTL
- `x_parser._SEARCH_COOLDOWN` / `_TWEETS_COOLDOWN` — 10min on failed endpoints

---

## troubleshooting

- **429 flooding the logs** — X cookies expired, refresh `X_AUTH_TOKEN`. Cooldowns prevent a death-loop but won't recover without live auth.
- **«⏳ смотрю что обсуждают» hangs** — rate-limited on For You. Wait 15min or use «моя лента» (Following is less restricted).
- **"разное · 3 поста"** — timeline is quiet right now, or filters are too tight for this hour. Try again in an hour, or widen `window_hours` in `bot/handlers/report.py`.
- **SQLite migration errors** — columns are added via `ALTER TABLE` in `db/database.py:init_db`. "column already exists" is idempotent-ignored.

---

## license

mit
