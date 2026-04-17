# x-news-bot

Телеграм-бот, который превращает твой X/Twitter feed в дайджест тем с постами. Тянет ленту (For You и Following), фильтрует мусор, кластеризует по смыслу через embeddings, даёт Claude назвать тему и написать выжимку.

## что умеет

- **📊 что обсуждают** — снапшот For You за 12ч, сгруппирован в супер-категории → под-темы → посты. Листаешь стрелочками, лайкаешь, переводишь, читаешь топ-комменты.
- **📰 моя лента** — то же самое, но источник Following за 24ч (вся лента подписок за вечер/ночь/утро).
- **💬 хочу больше X** — свободный текст → Claude определяет кластеры, подбирает реальные X-аккаунты по теме, валидирует через X API и сразу тянет свежие посты по теме (с similarity-гейтом от мусора).
- **👍 / 👎** — учат persistent preference-vector + понижают дизлайкнутые темы.
- **dual-media** — если автор цитирует пост с медиа, бот шлёт 2 сообщения (автор + квота) и корректно удаляет оба при листании.
- **/reset**, **⏸ пауза**, кастомный интервал доставки.

## архитектура

```
telegram (aiogram) ──► handlers (report / feed / preferences / discussion / onboarding)
                               │
                               ▼
                       core/report.build-report
                          ├── x-parser.get-home-timeline / get-for-you-timeline
                          ├── filters.is-low-signal (regex + hype + density)
                          ├── embeddings.embed-batch (openai text-embedding-3-small)
                          ├── cosine-union-find кластеризация (threshold 0.42)
                          ├── ai-client.name-topic (claude haiku)
                          ├── ai-client.group-super-topics
                          └── ai-client.summarize-discussion (upfront, топ-7 тем)
                               │
                               ▼
                       delivery.send-one-tweet (photo/video/animation/text + dual-media)
                               │
                               ▼
                       sqlite (tweets, users, sent-news, feedback, followed-authors)
```

## ключевые защиты от rate-limit

X очень быстро отдаёт 429 на read-endpoints при активном скрапинге. Внутри `core/x_parser.py`:

- **`_X_API_SEM = asyncio.Semaphore(2)`** — один на процесс, ограничивает параллельные GraphQL запросы.
- **`_USER_LOOKUP_CACHE`** — TTL 1ч для ok, 10м для 404/429. Один handle не бьётся повторно в том же build-report.
- **`_SEARCH_COOLDOWN`** — 10м cooldown для query после 404/rate-limit.
- **`_TWEETS_COOLDOWN`** — handle, у которого `get_user_tweets` упал на 429, пропускается.
- **`_shorten_query`** в `ai-client` — обрезает Claude-queries до 3 слов (длинные запросы стабильно дают 404 от SearchTimeline).

## стек

- `python 3.12`
- `aiogram 3.x` — telegram bot
- `twikit` — X/Twitter GraphQL cookie-auth
- `anthropic` SDK — Claude Haiku для naming/summarization, Sonnet для onboarding
- `openai` SDK — embeddings (через ProxyAPI или напрямую)
- `sqlalchemy 2.x + aiosqlite`
- `apscheduler` — delivery-job каждые 20 мин, implicit-decay каждые 6ч
- `chromadb` — вспомогательный vector store

## установка

```bash
git clone <repo-url> x-news-bot
cd x-news-bot
python -m venv venv
venv/Scripts/activate          # windows
# source venv/bin/activate     # linux/mac
pip install -r requirements.txt
cp .env.example .env           # заполни токены
python -m bot.main
```

## конфигурация (.env)

```
TELEGRAM_BOT_TOKEN=<from @BotFather>
PROXYAPI_KEY=<proxyapi.ru or direct anthropic/openai>
ANTHROPIC_BASE_URL=https://api.proxyapi.ru/anthropic
OPENAI_BASE_URL=https://api.proxyapi.ru/openai/v1
MODEL_HAIKU=claude-haiku-4-5
MODEL_SONNET=claude-sonnet-4-5

X_AUTH_TOKEN=<cookie auth_token из devtools>
X_CT0=<опционально — cookie ct0>

DATABASE_URL=sqlite+aiosqlite:///./bot.db
DEFAULT_DELIVERY_INTERVAL_HOURS=3
```

auth_token берётся из браузерного сеанса X: DevTools → Application → Cookies → x.com → auth_token. Рекомендуется отдельный burner-аккаунт.

## структура

```
x-news-bot/
├── bot/
│   ├── main.py              # entrypoint — polling + scheduler
│   ├── delivery.py          # send-one-tweet, dual-media, cleanup-post-by-message
│   ├── keyboards.py         # inline клавиатуры
│   └── handlers/
│       ├── report.py        # «что обсуждают» / «моя лента»
│       ├── feed.py          # 👍/👎/🇷🇺/💬 под постом
│       ├── preferences.py   # «хочу больше X» — claude + immediate-fetch + similarity-гейт
│       ├── discussion.py    # reply-ответы на пост через claude
│       └── onboarding.py    # /start — первичная настройка
├── core/
│   ├── x_parser.py          # twikit + ttl-кэш + семафор + cooldown
│   ├── ai_client.py         # все вызовы claude (pydantic-схемы)
│   ├── embeddings.py        # openai embeddings + nearest-cluster
│   ├── filters.py           # is-low-signal (trash/hype/density/list-dump)
│   ├── report.py            # build-report: fetch → filter → embed → cluster → name → summary
│   ├── recommender.py       # score-tweet, pick-top-for-user
│   └── demo_data.py         # демо-пул для fallback
├── db/
│   ├── database.py          # async engine + in-place миграции
│   ├── models.py            # user, tweet, feedback, sent-news, followed-author
│   └── vector_store.py      # chroma
├── scheduler.py             # delivery-job + implicit-decay-job
├── config.py                # settings (env)
└── requirements.txt
```

## как работает ранжирование

```
score(tweet, user) =
    0.40 * relevance       # cosine(tweet.embedding, user.preference-vector)
  + 0.15 * cluster-boost   # взвешенная сумма cosine к якорям кластеров
  + 0.25 * trust           # source-trust-score (followers + age + verified)
  + 0.20 * freshness       # exp(-hours-old / 12)
  - 0.15 * diversity-pen   # похожесть на уже выбранные в батче
  + author-bonus           # (followed-author.weight - 1) * 0.25
```

- **дизлайк** — главный сигнал обучения: `update-on-dislike` двигает preference-vector ОТ твита.
- **implicit-skip** — если пост показан 12ч назад и нет клика → слабый негативный сигнал.
- **exploration-ratio 0.20** — 1 из 5 слотов под разведку менее очевидных кандидатов.
- **«хочу больше X»** — Claude boost/suppress кластеров + immediate-fetch по query + автоматическое подключение релевантных X-аккаунтов с weight=2.0.

## что кэшируется

- `tweet.embedding` — навсегда (1536-float JSON), пересчитывать дорого
- `tweet.summary-ru` — ленивый перевод на русский, кэшируется в БД
- `cluster.summary` — upfront (топ-7) + lazy (при открытии темы)
- `Report` — в памяти процесса, per-user
- `x-parser._USER_LOOKUP_CACHE` — TTL 1ч
- `x-parser._SEARCH_COOLDOWN` / `_TWEETS_COOLDOWN` — 10м на failed endpoints

## troubleshooting

- **429 в логах снова и снова** — cookies X устарели, обнови `X_AUTH_TOKEN`. Cooldown в коде предотвращает death-loop, но не вернёт данные без живой auth.
- **бот висит на «⏳ смотрю что обсуждают»** — X rate-limit на For You. Подожди 15 мин или используй «моя лента» (Following менее лимитирован).
- **пустые темы «разное · 3 поста»** — лента спокойная в это время суток. Попробуй через час или расширь окно в `report.py`.
- **миграция SQLite** — колонки добавляются через `ALTER TABLE` в `db/database.py:init-db`. Ошибки «column already exists» игнорируются (idempotent).

## лицензия

mit
