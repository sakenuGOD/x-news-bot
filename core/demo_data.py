"""Демо-твиты для режима /demo — чтобы можно было тестить бота без X-креденшлов.

Набор реалистичных коротких постов по разным темам с фейковыми author_username
и свежими created_at. Embedding и topic-классификация считаются реально
(через ProxyAPI), так что рекомендер работает на настоящих векторах.

Картинки — стабильные URL от picsum.photos (seed-based), чтобы Telegram их
нормально подхватывал при send_photo.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from core import embeddings as emb
from db.models import Tweet

log = logging.getLogger(__name__)


def _img(seed: str) -> str:
    # Картинки к демо-твитам не делаем — пользователь справедливо заметил что
    # картинка не по теме поста это хуже чем её отсутствие. Реальные твиты с X
    # приходят со своими image_url — им картинки ставим.
    return None  # type: ignore[return-value]


DEMO_TWEETS: list[dict] = [
    {
        "author": "sama",
        "display": "Sam Altman",
        "text": "We just crossed 200M weekly active users on ChatGPT. The pace of AI adoption is unlike anything I've seen. Memory rollout to all Plus users next week.",
        "hours_ago": 2,
        "likes": 48000,
        "image": _img("chatgpt"),
    },
    {
        "author": "karpathy",
        "display": "Andrej Karpathy",
        "text": "Finished a great weekend hacking on nanochat — a tiny LLM you can train on a single 8xH100 box in a day. Code up on GitHub. Surprising how much capability fits in 124M params when data is clean.",
        "hours_ago": 4,
        "likes": 12000,
        "image": _img("nanochat"),
    },
    {
        "author": "elonmusk",
        "display": "Elon Musk",
        "text": "Starship Flight 5 caught the booster with chopsticks. Fully reusable orbital rocket is no longer science fiction. Mars is next.",
        "hours_ago": 6,
        "likes": 890000,
        "image": _img("starship"),
    },
    {
        "author": "dhh",
        "display": "DHH",
        "text": "Left AWS entirely last year. Savings so far: $2M/year. Our own hardware pays for itself in under 6 months. The cloud markup is no longer worth it for steady workloads.",
        "hours_ago": 8,
        "likes": 15400,
        "image": _img("servers"),
    },
    {
        "author": "paulg",
        "display": "Paul Graham",
        "text": "The best founders I've met share one trait: they can't not build the thing. Not motivation, not discipline — compulsion. If you have to force yourself to work on it, it's probably the wrong idea.",
        "hours_ago": 10,
        "likes": 28000,
        "image": _img("founders"),
    },
    {
        "author": "AnthropicAI",
        "display": "Anthropic",
        "text": "Claude Opus 4.7 is out today. Better at long-horizon coding (SWE-bench +6pp), improved agentic reasoning, and 1M context for Enterprise. Available in API and Claude.ai.",
        "hours_ago": 1,
        "likes": 9800,
        "image": _img("claude-opus"),
    },
    {
        "author": "VitalikButerin",
        "display": "Vitalik Buterin",
        "text": "Thinking about social recovery wallets again. The bottleneck for crypto adoption isn't UX anymore — it's trust in self-custody. We need guardian systems that feel as safe as banks without being banks.",
        "hours_ago": 14,
        "likes": 6700,
        "image": _img("ethereum"),
    },
    {
        "author": "naval",
        "display": "Naval",
        "text": "Read what you love until you love what you read. Don't grind through books you hate — that's just school with extra steps.",
        "hours_ago": 18,
        "likes": 42000,
        "image": _img("books"),
    },
    {
        "author": "levelsio",
        "display": "@levelsio",
        "text": "Shipped flight simulator in browser. $0 to $1M ARR in 3 weeks, solo dev, no VC. Three.js + a weekend of pain. Indie hacking is back.",
        "hours_ago": 20,
        "likes": 35000,
        "image": _img("flightsim"),
    },
    {
        "author": "nasa",
        "display": "NASA",
        "text": "JWST captured direct images of water ice clouds on a gas giant 40 light years away. First time we've resolved weather patterns on a non-solar-system planet.",
        "hours_ago": 12,
        "likes": 120000,
        "image": _img("jwst"),
    },
    {
        "author": "patio11",
        "display": "Patrick McKenzie",
        "text": "Reminder: enterprise SaaS pricing should not start with 'what will the customer pay'. It should start with 'how much pain am I removing and how do we split that value'. The former is a race to the bottom.",
        "hours_ago": 16,
        "likes": 4200,
        "image": _img("saas"),
    },
    {
        "author": "ylecun",
        "display": "Yann LeCun",
        "text": "LLMs will not reach human-level intelligence by scaling alone. We need world models, planning, and hierarchical reasoning. Autoregressive text prediction is a local maximum.",
        "hours_ago": 22,
        "likes": 18000,
        "image": _img("neuralnet"),
    },
    {
        "author": "OpenAI",
        "display": "OpenAI",
        "text": "Introducing Sora 2: text-to-video with 60 second clips, persistent characters across scenes, and physics-accurate simulation. Available to Pro users starting today.",
        "hours_ago": 3,
        "likes": 64000,
        "image": _img("sora"),
    },
    {
        "author": "garrytan",
        "display": "Garry Tan",
        "text": "YC W26 batch is the most AI-native we've ever seen. 74% of founders shipped their MVP entirely with AI coding agents. Average team size: 1.8 people. The leverage curve is bending.",
        "hours_ago": 5,
        "likes": 8900,
        "image": _img("ycombinator"),
    },
    {
        "author": "SpaceX",
        "display": "SpaceX",
        "text": "Starlink V3 satellites deployed. Each one delivers 1 Tbps of capacity — 10x V2. Direct-to-cell service expanding to 15 new countries this quarter.",
        "hours_ago": 7,
        "likes": 52000,
        "image": _img("starlink"),
    },
    {
        "author": "balajis",
        "display": "Balaji",
        "text": "The 2020s lesson so far: information moves at the speed of network, capital at the speed of crypto, and talent at the speed of remote work. Geography is no longer destiny.",
        "hours_ago": 11,
        "likes": 22000,
        "image": _img("network"),
    },
    {
        "author": "DeepMind",
        "display": "Google DeepMind",
        "text": "AlphaFold 4 predicts protein-ligand binding affinity with 0.93 correlation to wet-lab data. Drug discovery workflows that took 18 months now compress to weeks. Paper in Nature today.",
        "hours_ago": 9,
        "likes": 14500,
        "image": _img("alphafold"),
    },
    {
        "author": "tim_cook",
        "display": "Tim Cook",
        "text": "Apple Vision Pro 2 ships next month. Half the weight, twice the resolution, 8-hour battery. Starting at $2499. For developers: 400M+ spatial apps ready at launch.",
        "hours_ago": 13,
        "likes": 38000,
        "image": _img("visionpro"),
    },
    {
        "author": "GergelyOrosz",
        "display": "Gergely Orosz",
        "text": "Senior engineer survey 2026: 68% use an AI coding assistant daily, 41% have shipped features primarily through an agent. But 72% say review time has doubled — humans are becoming QA for machines.",
        "hours_ago": 15,
        "likes": 6100,
        "image": _img("engineers"),
    },
    {
        "author": "pmarca",
        "display": "Marc Andreessen",
        "text": "Every industry will have an AI-native incumbent killer within 18 months. Software is eating software. The question isn't whether to build with AI — it's whether you'll be the disruptor or the disrupted.",
        "hours_ago": 19,
        "likes": 25000,
        "image": _img("disruption"),
    },
    {
        "author": "CoinDesk",
        "display": "CoinDesk",
        "text": "Bitcoin breaks $150k for the first time. Spot ETF inflows hit $4.2B this week — institutional money is no longer testing waters, it's swimming.",
        "hours_ago": 17,
        "likes": 31000,
        "image": _img("bitcoin"),
    },
    {
        "author": "fchollet",
        "display": "François Chollet",
        "text": "ARC-AGI-2 benchmark results are in. Best model: 44%. Humans: 95%. Scaling is hitting diminishing returns on tasks requiring genuine abstraction. The frontier moved.",
        "hours_ago": 21,
        "likes": 11200,
        "image": _img("arc-agi"),
    },
]


async def seed_demo_tweets(session) -> int:
    """Вставляет демо-твиты в БД с реальными embedding и topic-классификацией.

    Идемпотентно: проверяет существование перед вставкой, НО на существующих
    dem-твитах докидывает image_url / display_name если их раньше не было
    (апгрейд старого seed после расширения арсенала).
    Возвращает число новых записей.
    """
    now = datetime.now(timezone.utc)
    existing_rows = (
        await session.execute(select(Tweet).where(Tweet.tweet_id.like("demo_%")))
    ).all()
    existing_by_id: dict[str, Tweet] = {row[0].tweet_id: row[0] for row in existing_rows}

    to_insert: list[tuple[dict, str]] = []
    for item in DEMO_TWEETS:
        # Стабильный ID — чтобы повторные /demo не плодили дубли.
        stable_id = f"demo_{item['author']}_{abs(hash(item['text'])) % 10_000_000}"
        existing = existing_by_id.get(stable_id)
        if existing is not None:
            # Если картинки в новом конфиге нет (мы отказались от picsum) —
            # зачищаем возможный старый picsum URL на существующих записях.
            if not item.get("image") and existing.image_url and "picsum" in (existing.image_url or ""):
                existing.image_url = None
            elif not existing.image_url and item.get("image"):
                existing.image_url = item["image"]
            if (not existing.author_display_name or existing.author_display_name == existing.author_username) \
                    and item.get("display"):
                existing.author_display_name = item["display"]
            continue
        to_insert.append((item, stable_id))

    if not to_insert:
        log.info("demo tweets already seeded")
        return 0

    texts = [it[0]["text"] for it in to_insert]
    embs = await emb.embed_batch(texts)

    inserted = 0
    for (item, stable_id), e in zip(to_insert, embs):
        if not e:
            continue
        topic, _sim = await emb.nearest_cluster(e)
        created = (now - timedelta(hours=item["hours_ago"])).replace(tzinfo=None)
        session.add(
            Tweet(
                tweet_id=stable_id,
                author_username=item["author"],
                author_display_name=item.get("display") or item["author"],
                text=item["text"],
                url=f"https://x.com/{item['author']}/status/{stable_id}",
                image_url=item.get("image"),
                embedding=e,
                summary_ru=None,  # посчитаем при нажатии "Перевод"
                source_trust_score=0.82,  # демо-источники условно доверенные
                misleading_score=None,
                topic=topic,
                likes_count=item.get("likes", 0),
                retweets_count=0,
                replies_count=0,
                created_at=created,
            )
        )
        inserted += 1

    log.info("seeded %d demo tweets", inserted)
    return inserted
