"""X (Twitter) парсинг через twikit — cookie-auth к внутреннему GraphQL.

Почему twikit, а не tweety-ns / official API:
  - Official free tier X API — write-only, читать нельзя.
  - Platform Basic начинается от ~$200/мес.
  - tweety-ns в 2026 ломается на anti-bot transactions без логина.
  - twikit использует cookie (auth_token) от обычного браузерного сеанса и
    ходит к тем же GraphQL-эндпоинтам, что x.com в браузере. Работает стабильно,
    пока X не меняет doc_id (раз в 2-3 месяца — обновить twikit).

Приоритет источников auth:
  1. Файл x_cookies.json (json от client.save_cookies() или ручной) — X_COOKIES_PATH.
  2. Env X_AUTH_TOKEN (+ X_CT0 опционально) — если нет файла.
  3. Логин по паролю через client.login() — если нет ни файла, ни токенов.
     После первого успешного логина cookies сохраняются в файл.

Рекомендация: используй отдельный burner X-аккаунт. X может забанить за активный
скрапинг. Не хостить на той же IP что продакшн. Задержки между запросами —
твоя ответственность (мы ставим Semaphore(2) для параллельных fetch).
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import math
import random
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from config import settings

log = logging.getLogger(__name__)


# ----------------------- X API concurrency + cache -----------------------
#
# Без этого было так: Claude генерит 5 interest-queries, каждая зовёт
# topic_authors_fallback → 12 authors → get_user_by_screen_name. Итого
# 60 concurrent hit'ов в X, мгновенный 429 на 15+ минут. А т.к. негативный
# результат нигде не запоминается, при следующем build_report бот снова
# бьёт те же 12 handles (GQMagazine, ELLEmagazine, …) которые в cooldown'е
# у X. Логи 400→700 строк показывают именно эту картину.
#
# Два уровня защиты:
#   1) Global Semaphore(2) — одновременно не более 2 X-запросов из процесса.
#   2) TTL-кэш для screen_name → User. Положительный ответ живёт 1ч
#      (followers/verified меняются медленно), отрицательный (429/404) — 10 мин
#      (даём X остыть, но не гоняем handles в цикле).

_X_API_SEM = asyncio.Semaphore(2)

# Screen_name → (expiry_monotonic, twikit.User | None).
# None — negative result: 404 / 429 / любой фейл. Не повторяем пока не истечёт TTL.
_USER_LOOKUP_CACHE: dict[str, tuple[float, Optional[object]]] = {}
_USER_LOOKUP_TTL_OK = 3600.0
_USER_LOOKUP_TTL_BAD = 600.0

# Handle → expiry: временный cooldown если get_user_tweets упал на 429.
# Отдельно от user_lookup: resolve может пройти, а tweets endpoint 429-ит.
_TWEETS_COOLDOWN: dict[str, float] = {}
_TWEETS_COOLDOWN_SEC = 600.0

# Query → expiry: не повторяем провальный search более одного раза за 10 мин.
_SEARCH_COOLDOWN: dict[str, float] = {}
_SEARCH_COOLDOWN_SEC = 600.0


def _cache_get(cache: dict, key: str):
    entry = cache.get(key)
    if not entry:
        return False, None
    exp = entry[0] if isinstance(entry, tuple) else entry
    if time.monotonic() >= exp:
        cache.pop(key, None)
        return False, None
    return True, (entry[1] if isinstance(entry, tuple) else None)


def _cooldown_active(cache: dict, key: str) -> bool:
    if not key:
        return False
    exp = cache.get(key)
    if not exp:
        return False
    if time.monotonic() >= exp:
        cache.pop(key, None)
        return False
    return True


def _cooldown_set(cache: dict, key: str, seconds: float) -> None:
    if key:
        cache[key] = time.monotonic() + seconds

# ----------------------- twikit runtime patches -----------------------
#
# X в апреле-2026 выпилил `ondemand.s.<hash>a.js` и переделал JS-бандлы в
# `vendor.<hash>.js` / `main.<hash>.js`. Регулярка twikit для добычи
# KEY_BYTE индексов больше ничего не находит, ClientTransaction.init падает
# с "Couldn't get KEY_BYTE indices", и все последующие запросы ломаются на
# `'ClientTransaction' object has no attribute 'key'`.
#
# Выяснено эмпирически: X сейчас не валидирует X-Client-Transaction-Id
# жёстко на read-операции (GraphQL GET-запросы). Подсовываем случайный base64
# той же длины — read-путь проходит.
#
# Побочно: в той же волне X убрал из `legacy.entities.description.urls` это
# поле у части аккаунтов, twikit.User.__init__ падает KeyError('urls').
# Переопределяем User.__init__ на defensive версию с .get() везде.
#
# Эти патчи ставятся один раз при импорте модуля.

_patches_applied = False

# Реальный X Client Transaction ID генератор. Инициализируется лениво при
# первом запросе (нужен httpx-клиент для homepage + ondemand.s.js fetch).
# Пакет `x-client-transaction-id` корректно реверсит алгоритм X — важно,
# потому что без него X возвращает 404 на SearchTimeline (selective enforcement:
# HomeTimeline/UserTweets пропускают dummy-TID, Search — нет).
_REAL_TID_CT = None  # type: ignore[var-annotated]
_REAL_TID_INIT_LOCK: Optional[asyncio.Lock] = None


async def _ensure_real_tid_generator(_unused_client=None):
    """Создаёт ClientTransaction из x_client_transaction (single-shot).

    Делаем отдельный httpx-клиент (не тот что в twikit), чтобы `set-cookie`
    от x.com не дублировал `twid`-куку в twikit-сессии (иначе httpx падает
    с `Multiple cookies exist with name=twid` на последующих запросах).
    """
    global _REAL_TID_CT, _REAL_TID_INIT_LOCK
    if _REAL_TID_CT is not None:
        return _REAL_TID_CT
    if _REAL_TID_INIT_LOCK is None:
        _REAL_TID_INIT_LOCK = asyncio.Lock()
    async with _REAL_TID_INIT_LOCK:
        if _REAL_TID_CT is not None:
            return _REAL_TID_CT
        try:
            import bs4
            import httpx
            from x_client_transaction import ClientTransaction as _RealCT
            from x_client_transaction.utils import (
                handle_x_migration_async, get_ondemand_file_url, generate_headers,
            )
            # Отдельный клиент, свои cookies — не мутируем основную twikit-сессию.
            async with httpx.AsyncClient(
                follow_redirects=True, timeout=30,
            ) as tmp:
                home = await handle_x_migration_async(tmp)
                ondemand_url = get_ondemand_file_url(home)
                r = await tmp.get(ondemand_url, headers=generate_headers())
                if r.status_code != 200:
                    log.warning("ondemand.s.js fetch %d — real TID disabled", r.status_code)
                    return None
                ondemand = bs4.BeautifulSoup(r.text, "html.parser")
                _REAL_TID_CT = _RealCT(home, ondemand)
                log.info("real X client-transaction-id generator initialized")
                return _REAL_TID_CT
        except Exception as e:
            log.warning("real TID generator init failed: %s", e)
            return None


def _apply_twikit_patches() -> None:
    global _patches_applied
    if _patches_applied:
        return
    try:
        from twikit.x_client_transaction import ClientTransaction
        from twikit import user as _twikit_user_module
    except ImportError:
        return

    _orig_init = ClientTransaction.init
    _orig_gen = ClientTransaction.generate_transaction_id

    async def patched_init(self, session, headers):
        try:
            await _orig_init(self, session, headers)
            self._init_ok = True
        except Exception as e:
            log.debug("twikit ClientTransaction.init soft-fail (patched): %s", e)
            self._init_ok = False
            # home_page_response должен быть truthy — иначе request() будет бесконечно re-init
            from bs4 import BeautifulSoup  # twikit уже её тянет
            self.home_page_response = BeautifulSoup("<html></html>", "html.parser")

    def patched_gen(self, method=None, path=None, response=None, key=None, animation_key=None, time_now=None):
        # Приоритет 1 — настоящий X-алгоритм через x_client_transaction (pypi).
        # Он реверс-инженерит тот же путь что использует x.com в браузере.
        # Инициализация ленивая и глобальная (_REAL_TID_CT), делается при старте
        # клиента. Если есть — X принимает TID везде, включая SearchTimeline.
        if _REAL_TID_CT is not None:
            try:
                return _REAL_TID_CT.generate_transaction_id(method=method or "GET", path=path or "/")
            except Exception as e:
                log.debug("real TID generate failed, falling back to dummy: %s", e)

        # Приоритет 2 — оригинальный twikit-путь (работает если init прошёл).
        if getattr(self, "_init_ok", True):
            try:
                return _orig_gen(self, method, path, response, key, animation_key, time_now)
            except Exception as e:
                log.debug("twikit generate_transaction_id fallthrough: %s", e)

        # Приоритет 3 — рандом. Работает только для «мягких» endpoint'ов
        # (HomeTimeline/UserTweets/TweetDetail). SearchTimeline с рандомом — 404.
        rnd = bytes(random.randint(0, 255) for _ in range(70))
        return base64.b64encode(rnd).decode().rstrip("=")

    ClientTransaction.init = patched_init
    ClientTransaction.generate_transaction_id = patched_gen

    # --- User.__init__ defensive ---
    _orig_user_init = _twikit_user_module.User.__init__

    def _coalesce_screen_name(data: dict, legacy: dict) -> str:
        """В X-2026 для SearchTimeline screen_name живёт в `core.screen_name`,
        а в `legacy.screen_name` пусто (до 2025 было наоборот). UserTweets/
        HomeTimeline до сих пор кладут в legacy. Берём первое непустое.
        Без этого 100% search-results дропаются в _convert_tweet с
        `ValueError: no author for tweet ...` — каждый «больше моды/футбола»
        отдаёт 0 постов и падает в curated-fallback (там 429 loop)."""
        core = data.get("core") or {}
        return (core.get("screen_name") or legacy.get("screen_name") or "").strip()

    def _coalesce_name(data: dict, legacy: dict) -> str:
        core = data.get("core") or {}
        return core.get("name") or legacy.get("name", "")

    def patched_user_init(self, client, data):
        try:
            _orig_user_init(self, client, data)
            # Постфикс для SEARCH-результатов: оригинальный init читает
            # legacy.screen_name, у search-веток этого поля нет → screen_name=''
            # → весь tweet дропается. Если пусто — добираем из core.
            if not getattr(self, "screen_name", ""):
                legacy_p = data.get("legacy") or {}
                sn = _coalesce_screen_name(data, legacy_p)
                if sn:
                    self.screen_name = sn
            if not getattr(self, "name", ""):
                legacy_p = data.get("legacy") or {}
                nm = _coalesce_name(data, legacy_p)
                if nm:
                    self.name = nm
            return
        except (KeyError, TypeError) as e:
            log.debug("User.__init__ schema drift, falling back to defensive: %s", e)

        self._client = client
        legacy = data.get("legacy") or {}
        ent = legacy.get("entities") or {}
        self.id = data.get("rest_id") or legacy.get("id_str", "")
        self.created_at = legacy.get("created_at", "")
        self.name = _coalesce_name(data, legacy)
        self.screen_name = _coalesce_screen_name(data, legacy)
        self.profile_image_url = legacy.get("profile_image_url_https", "")
        self.profile_banner_url = legacy.get("profile_banner_url")
        self.url = legacy.get("url")
        self.location = legacy.get("location", "")
        self.description = legacy.get("description", "")
        self.description_urls = (ent.get("description") or {}).get("urls") or []
        self.urls = (ent.get("url") or {}).get("urls") or []
        self.pinned_tweet_ids = legacy.get("pinned_tweet_ids_str", [])
        self.is_blue_verified = data.get("is_blue_verified", False)
        self.verified = legacy.get("verified", False)
        self.possibly_sensitive = legacy.get("possibly_sensitive", False)
        self.can_dm = legacy.get("can_dm", False)
        self.can_media_tag = legacy.get("can_media_tag", False)
        self.want_retweets = legacy.get("want_retweets", False)
        self.default_profile = legacy.get("default_profile", False)
        self.default_profile_image = legacy.get("default_profile_image", False)
        self.has_custom_timelines = legacy.get("has_custom_timelines", False)
        self.followers_count = legacy.get("followers_count", 0)
        self.fast_followers_count = legacy.get("fast_followers_count", 0)
        self.normal_followers_count = legacy.get("normal_followers_count", 0)
        self.following_count = legacy.get("friends_count", 0)
        self.favourites_count = legacy.get("favourites_count", 0)
        self.listed_count = legacy.get("listed_count", 0)
        self.media_count = legacy.get("media_count", 0)
        self.statuses_count = legacy.get("statuses_count", 0)
        self.is_translator = legacy.get("is_translator", False)
        self.translator_type = legacy.get("translator_type", "none")
        self.profile_interstitial_type = legacy.get("profile_interstitial_type", "")
        self.withheld_in_countries = legacy.get("withheld_in_countries", [])
        self.protected = legacy.get("protected", False)

    _twikit_user_module.User.__init__ = patched_user_init

    # --- патч 3: _get_user_state на 429 → не рекурсировать ---
    # Баг в twikit: при 429 request() зовёт _get_user_state для проверки «suspended»,
    # тот делает GET, получает 429 снова → бесконечная рекурсия → RecursionError.
    # Возвращаем 'active' без похода в сеть. Мы всё равно ловим rate-limit выше по коду.
    try:
        from twikit.client.client import Client as _TwikitClient

        async def patched_get_user_state(self):
            return "active"

        _TwikitClient._get_user_state = patched_get_user_state
    except Exception as e:
        log.warning("failed to patch _get_user_state: %s", e)

    # --- патч 5: свежий SearchTimeline (queryId + features + variables) ---
    # X ротирует GraphQL-эндпоинты: меняется и queryId, и набор features, и
    # иногда variables. twikit 2.3.3 застрял на весенней схеме (21 feature),
    # а X в апреле-2026 требует 37 features и дополнительное поле
    # `withGrokTranslatedBio` в variables. Старый endpoint отдаёт 404
    # целиком (не selectively по features) — так X отсекает устаревших
    # клиентов, вместо «частичного» ответа без новых полей.
    #
    # Снятые значения — из curl-дампа реального запроса из браузера юзера
    # (DevTools → Network → SearchTimeline → Copy as cURL). Актуальность
    # сохраняется 1-3 месяца, потом X сдвинет очередной doc_id — тогда
    # тот же путь: открыть Network, пересмотреть, обновить константы ниже.
    try:
        from twikit.client import gql as _twikit_gql
        _NEW_SEARCH_DOC_ID = "R0u1RWRf748KzyGBXvOYRA"
        _twikit_gql.Endpoint.SEARCH_TIMELINE = (
            f"https://x.com/i/api/graphql/{_NEW_SEARCH_DOC_ID}/SearchTimeline"
        )

        # Полный набор features, которые X ожидает для SearchTimeline весной-2026.
        _SEARCH_FEATURES = {
            "rweb_video_screen_enabled": False,
            "rweb_cashtags_enabled": True,
            "profile_label_improvements_pcf_label_in_post_enabled": True,
            "responsive_web_profile_redirect_enabled": False,
            "rweb_tipjar_consumption_enabled": False,
            "verified_phone_label_enabled": False,
            "creator_subscriptions_tweet_preview_api_enabled": True,
            "responsive_web_graphql_timeline_navigation_enabled": True,
            "responsive_web_graphql_skip_user_profile_image_extensions_enabled": False,
            "premium_content_api_read_enabled": False,
            "communities_web_enable_tweet_community_results_fetch": True,
            "c9s_tweet_anatomy_moderator_badge_enabled": True,
            "responsive_web_grok_analyze_button_fetch_trends_enabled": False,
            "responsive_web_grok_analyze_post_followups_enabled": True,
            "responsive_web_jetfuel_frame": True,
            "responsive_web_grok_share_attachment_enabled": True,
            "responsive_web_grok_annotations_enabled": True,
            "articles_preview_enabled": True,
            "responsive_web_edit_tweet_api_enabled": True,
            "graphql_is_translatable_rweb_tweet_is_translatable_enabled": True,
            "view_counts_everywhere_api_enabled": True,
            "longform_notetweets_consumption_enabled": True,
            "responsive_web_twitter_article_tweet_consumption_enabled": True,
            "content_disclosure_indicator_enabled": True,
            "content_disclosure_ai_generated_indicator_enabled": True,
            "responsive_web_grok_show_grok_translated_post": True,
            "responsive_web_grok_analysis_button_from_backend": True,
            "post_ctas_fetch_enabled": True,
            "freedom_of_speech_not_reach_fetch_enabled": True,
            "standardized_nudges_misinfo": True,
            "tweet_with_visibility_results_prefer_gql_limited_actions_policy_enabled": True,
            "longform_notetweets_rich_text_read_enabled": True,
            "longform_notetweets_inline_media_enabled": False,
            "responsive_web_grok_image_annotation_enabled": True,
            "responsive_web_grok_imagine_annotation_enabled": True,
            "responsive_web_grok_community_note_auto_translation_is_enabled": True,
            "responsive_web_enhance_cards_enabled": False,
        }

        # Заменяем метод search_timeline целиком — он строит variables+features
        # и зовёт gql_get. Мы передаём свои свежие features.
        async def patched_search_timeline(self, query, product, count, cursor):
            variables = {
                "rawQuery": query,
                "count": count,
                "querySource": "typed_query",
                "product": product,
                "withGrokTranslatedBio": True,  # новое поле для свежего endpoint
            }
            if cursor is not None:
                variables["cursor"] = cursor
            return await self.gql_get(
                _twikit_gql.Endpoint.SEARCH_TIMELINE,
                variables,
                _SEARCH_FEATURES,
            )

        _twikit_gql.GQLClient.search_timeline = patched_search_timeline
        log.debug("twikit patch 5 applied: fresh SearchTimeline (queryId + features + variables)")
    except Exception as e:
        log.warning("failed to patch SearchTimeline: %s", e)

    # --- патч 4: get_tweet_by_id — cursor schema drift ---
    # twikit 2.3.3 в client.py:1635 ожидает `entries[-1].content.itemContent.value`
    # для cursor'а, но X весной-2026 сменил формат на `entries[-1].content.value`
    # (TimelineTimelineCursor напрямую, без itemContent-обёртки). Падает KeyError,
    # весь get_tweet_by_id ломается → get_top_replies возвращает пусто.
    # Патчим: копируем тело оригинала, cursor достаём через defensive helper.
    try:
        from twikit.client.client import Client as _TwikitClient
        from twikit.utils import find_dict, Result as _TwikitResult
        from twikit.tweet import tweet_from_data
        from functools import partial as _partial
        from twikit.errors import TweetNotAvailable

        def _extract_cursor_value(entry):
            """`content.itemContent.value` ИЛИ `content.value` — обе схемы видел."""
            if not isinstance(entry, dict):
                return None
            content = entry.get("content") or {}
            item = content.get("itemContent")
            if isinstance(item, dict) and "value" in item:
                return item["value"]
            if "value" in content:
                return content["value"]
            return None

        async def patched_get_tweet_by_id(self, tweet_id, cursor=None):
            response, _ = await self.gql.tweet_detail(tweet_id, cursor)
            if "errors" in response:
                raise TweetNotAvailable(response["errors"][0]["message"])

            entries_found = find_dict(response, "entries", find_one=True)
            if not entries_found:
                raise TweetNotAvailable(f"no entries in TweetDetail for {tweet_id}")
            entries = entries_found[0]
            reply_to, replies_list, related_tweets = [], [], []
            tweet = None

            for entry in entries:
                eid = entry.get("entryId", "")
                if eid.startswith("cursor"):
                    continue
                try:
                    tweet_object = tweet_from_data(self, entry)
                except Exception as e:
                    log.debug("tweet_from_data failed for entry %s: %s", eid, e)
                    continue
                if tweet_object is None:
                    continue

                if eid.startswith("tweetdetailrelatedtweets"):
                    related_tweets.append(tweet_object)
                    continue

                if eid == f"tweet-{tweet_id}":
                    tweet = tweet_object
                    continue

                if tweet is None:
                    reply_to.append(tweet_object)
                    continue

                # Reply conversation — thread с главным твитом наверху и реплаями ниже.
                items = ((entry.get("content") or {}).get("items") or [])[1:]
                sub_replies = []
                sr_cursor = None
                show_replies = None
                for reply in items:
                    r_eid = reply.get("entryId", "")
                    if "tweetcomposer" in r_eid:
                        continue
                    if "tweet" in r_eid:
                        try:
                            rpl = tweet_from_data(self, reply)
                        except Exception:
                            rpl = None
                        if rpl is not None:
                            sub_replies.append(rpl)
                    if "cursor" in r_eid:
                        item = reply.get("item") or {}
                        ic = item.get("itemContent") or {}
                        sr_cursor = ic.get("value") or item.get("value")
                        if sr_cursor:
                            show_replies = _partial(
                                self._show_more_replies, tweet_id, sr_cursor,
                            )
                try:
                    tweet_object.replies = _TwikitResult(sub_replies, show_replies, sr_cursor)
                except Exception:
                    tweet_object.replies = sub_replies
                replies_list.append(tweet_object)

            # Defensive cursor extraction — было KeyError('itemContent') на этом месте.
            reply_next_cursor = None
            _fetch_more_replies = None
            if entries and entries[-1].get("entryId", "").startswith("cursor"):
                reply_next_cursor = _extract_cursor_value(entries[-1])
                if reply_next_cursor:
                    _fetch_more_replies = _partial(
                        self._get_more_replies, tweet_id, reply_next_cursor,
                    )

            if tweet is None:
                raise TweetNotAvailable(f"main tweet {tweet_id} not in entries")

            try:
                tweet.replies = _TwikitResult(
                    replies_list, _fetch_more_replies, reply_next_cursor,
                )
            except Exception:
                tweet.replies = replies_list
            tweet.reply_to = reply_to
            tweet.related_tweets = related_tweets
            return tweet

        _TwikitClient.get_tweet_by_id = patched_get_tweet_by_id
        log.debug("twikit patch 4 applied: defensive get_tweet_by_id cursor")
    except Exception as e:
        log.warning("failed to patch get_tweet_by_id: %s", e)

    _patches_applied = True
    log.info("twikit runtime patches applied (anti-bot bypass + defensive User parsing + no-recurse on 429)")


@dataclass
class RawTweet:
    tweet_id: str
    author_username: str
    author_display_name: str
    text: str
    url: str
    image_url: Optional[str]
    created_at: datetime
    likes_count: int = 0
    retweets_count: int = 0
    replies_count: int = 0
    is_retweet_no_comment: bool = False
    hashtags: list[str] = field(default_factory=list)
    # "photo" / "video" / "animation" / None
    media_type: Optional[str] = None
    # Вторая дорожка медиа — если автор пост-кво с своим видео, а квота тоже
    # с видео, telegram-сообщение одно может нести только ОДНО медиа. Мы
    # сохраняем обе: image_url = автор-медиа, quote_image_url = медиа квоты.
    # Доставщик шлёт 2 сообщения последовательно: пост автора + пост квоты.
    quote_image_url: Optional[str] = None
    quote_media_type: Optional[str] = None
    # Кто квотируется — чтобы caption второго сообщения был «↪ @<handle>: ...».
    quote_author: Optional[str] = None
    quote_text: Optional[str] = None
    # Если в тексте есть t.co ссылка на ДРУГОЙ статус-твит (RT/embed без
    # native quote) — сохраняем target tweet_id. Delivery позже использует
    # его чтобы подтянуть медиа (типичный кейс «by @kianbazza t.co/X» где
    # видео физически у @kianbazza, а текущий пост без медиа).
    linked_tweet_id: Optional[str] = None

    @property
    def age_hours(self) -> float:
        now = datetime.now(timezone.utc)
        created = self.created_at if self.created_at.tzinfo else self.created_at.replace(tzinfo=timezone.utc)
        return (now - created).total_seconds() / 3600


@dataclass
class AuthorInfo:
    username: str
    display_name: str
    followers_count: int
    following_count: int
    verified: bool
    account_age_days: int
    recent_engagement_rate: float = 0.0


class XAuthError(RuntimeError):
    """Нет рабочего способа авторизоваться в X."""


class XParser:
    """Один клиент twikit на процесс. Авторизуется лениво при первом вызове."""

    def __init__(self) -> None:
        self._client = None  # type: ignore[var-annotated]
        self._auth_lock = asyncio.Lock()
        self._auth_ok: bool = False

    @property
    def is_authenticated(self) -> bool:
        return self._auth_ok

    async def _ensure_client(self):
        if self._client is not None and self._auth_ok:
            return self._client

        async with self._auth_lock:
            if self._client is not None and self._auth_ok:
                return self._client

            try:
                from twikit import Client  # type: ignore
            except ImportError as e:
                raise XAuthError("twikit не установлен. pip install twikit") from e

            _apply_twikit_patches()
            client = Client("en-US")
            cookies_path = Path(settings.x_cookies_path)

            # 1) Файл cookies — самый живучий путь.
            if cookies_path.exists() and cookies_path.stat().st_size > 0:
                try:
                    client.load_cookies(str(cookies_path))
                    self._client = client
                    self._auth_ok = True
                    log.info("X auth: cookies loaded from %s", cookies_path)
                    # Инициализируем настоящий X-client-transaction-id генератор.
                    # Без него SearchTimeline возвращает 404 (selective validation).
                    try:
                        await _ensure_real_tid_generator(client.http)
                    except Exception as e:
                        log.warning("real TID init failed (search may 404): %s", e)
                    return client
                except Exception as e:
                    log.warning("X cookies file malformed (%s): %s", cookies_path, e)

            # 2) Токены напрямую из env — создаём cookies dict.
            if settings.x_auth_token:
                cookies = {"auth_token": settings.x_auth_token}
                if settings.x_ct0:
                    cookies["ct0"] = settings.x_ct0
                try:
                    client.set_cookies(cookies)
                    self._client = client
                    self._auth_ok = True
                    # Сохраняем для будущих запусков — дальше полный формат cookies.json.
                    try:
                        client.save_cookies(str(cookies_path))
                    except Exception as e:
                        log.debug("save_cookies after env set: %s", e)
                    log.info("X auth: cookies set from env X_AUTH_TOKEN")
                    return client
                except Exception as e:
                    log.warning("X set_cookies from env failed: %s", e)

            # 3) Логин по паролю — самый хрупкий путь, но он даёт нам свежие cookies.
            if settings.x_username and settings.x_password:
                try:
                    login_kwargs = {
                        "auth_info_1": settings.x_username,
                        "password": settings.x_password,
                    }
                    if settings.x_email:
                        login_kwargs["auth_info_2"] = settings.x_email
                    if settings.x_totp_secret:
                        login_kwargs["totp_secret"] = settings.x_totp_secret
                    await client.login(**login_kwargs)
                    try:
                        client.save_cookies(str(cookies_path))
                        log.info("X auth: logged in, cookies saved to %s", cookies_path)
                    except Exception as e:
                        log.warning("save_cookies after login: %s", e)
                    self._client = client
                    self._auth_ok = True
                    return client
                except Exception as e:
                    log.error("X login failed: %s", e)

            raise XAuthError(
                "X auth недоступен. Заполни X_AUTH_TOKEN в .env "
                "(из DevTools → Application → Cookies → x.com → auth_token) "
                "или положи рабочий cookies.json по пути X_COOKIES_PATH."
            )

    # ----------------------- public API -----------------------

    async def _resolve_user(self, username: str):
        """Кэшированный lookup screen_name → twikit.User.

        Возвращает User или None (404/429/timeout). None тоже кэшируется на 10 мин —
        это и есть ключ борьбы с 429-циклом: пока X не остынет, мы не бомбим тот же
        handle десятки раз в рамках одного build_report.
        """
        key = (username or "").lower().strip()
        if not key:
            return None
        hit, cached = _cache_get(_USER_LOOKUP_CACHE, key)
        if hit:
            return cached

        client = await self._ensure_client()
        user = None
        ttl = _USER_LOOKUP_TTL_BAD
        try:
            async with _X_API_SEM:
                user = await asyncio.wait_for(
                    client.get_user_by_screen_name(username),
                    timeout=12,
                )
            ttl = _USER_LOOKUP_TTL_OK
        except RecursionError:
            log.debug("resolve_user(%s): 429 recursion — cooldown 10min", username)
        except asyncio.TimeoutError:
            log.debug("resolve_user(%s): timeout — cooldown 10min", username)
        except Exception as e:
            log.debug("resolve_user(%s): %s — cooldown 10min", username, str(e)[:140])

        _USER_LOOKUP_CACHE[key] = (time.monotonic() + ttl, user)
        return user

    async def get_following(self, username: str, limit: int = 200) -> list[str]:
        """usernames тех на кого подписан `username`. Постранично до limit."""
        user = await self._resolve_user(username)
        if user is None:
            return []
        client = await self._ensure_client()

        collected: list[str] = []
        try:
            async with _X_API_SEM:
                page = await client.get_user_following(user.id, count=min(40, limit))
            while page:
                for u in page:
                    uname = getattr(u, "screen_name", None) or getattr(u, "username", None)
                    if uname:
                        collected.append(uname)
                    if len(collected) >= limit:
                        break
                if len(collected) >= limit:
                    break
                # Пагинация: у Result есть .next() — если cursor есть.
                if hasattr(page, "next"):
                    try:
                        async with _X_API_SEM:
                            page = await page.next()
                    except Exception as e:
                        log.debug("get_following pagination end: %s", e)
                        break
                else:
                    break
                await asyncio.sleep(0.8)  # щадим rate-limit
        except Exception as e:
            log.error("get_user_following(%s): %s", username, e)

        return collected[:limit]

    async def get_author_info(self, username: str) -> Optional[AuthorInfo]:
        u = await self._resolve_user(username)
        if u is None:
            return None

        followers = int(getattr(u, "followers_count", 0) or 0)
        following = int(getattr(u, "following_count", 0) or getattr(u, "friends_count", 0) or 0)
        verified = bool(getattr(u, "verified", False) or getattr(u, "is_blue_verified", False))
        created = getattr(u, "created_at_datetime", None) or getattr(u, "created_at", None)
        age_days = 365
        if isinstance(created, datetime):
            created = created if created.tzinfo else created.replace(tzinfo=timezone.utc)
            age_days = max(1, (datetime.now(timezone.utc) - created).days)
        return AuthorInfo(
            username=username,
            display_name=getattr(u, "name", username) or username,
            followers_count=followers,
            following_count=following,
            verified=verified,
            account_age_days=age_days,
        )

    async def get_recent_tweets(self, username: str, limit: int = 15) -> list[RawTweet]:
        key = (username or "").lower().strip()
        if _cooldown_active(_TWEETS_COOLDOWN, key):
            log.debug("get_recent_tweets(%s): skipped — handle in tweets-cooldown", username)
            return []
        user = await self._resolve_user(username)
        if user is None:
            return []
        client = await self._ensure_client()
        try:
            async with _X_API_SEM:
                tweets_result = await asyncio.wait_for(
                    client.get_user_tweets(user.id, "Tweets", count=min(40, limit)),
                    timeout=15,
                )
        except RecursionError:
            log.warning("get_user_tweets(%s): rate-limited — cooldown 10min", username)
            _cooldown_set(_TWEETS_COOLDOWN, key, _TWEETS_COOLDOWN_SEC)
            return []
        except asyncio.TimeoutError:
            log.warning("get_user_tweets(%s): timeout — cooldown 10min", username)
            _cooldown_set(_TWEETS_COOLDOWN, key, _TWEETS_COOLDOWN_SEC)
            return []
        except Exception as e:
            msg = str(e)[:200]
            log.warning("get_user_tweets(%s): %s", username, msg)
            if "429" in msg or "rate limit" in msg.lower():
                _cooldown_set(_TWEETS_COOLDOWN, key, _TWEETS_COOLDOWN_SEC)
            return []

        out: list[RawTweet] = []
        for t in tweets_result:
            try:
                out.append(_convert_tweet(t, username))
            except Exception as e:
                log.debug("convert tweet failed: %s", e)
            if len(out) >= limit:
                break
        return out

    async def get_home_timeline(self, limit: int = 100) -> list[RawTweet]:
        """Лента «Following/Latest» — хронологический feed подписок.

        То что пользователь видит у себя в X на вкладке «Following» —
        без алгоритма, просто новое сверху.

        Пагинация: X за один запрос отдаёт ~40-80 постов. Чтобы добрать
        до `limit`, идём через .next(). Юзер жаловался «он не собирает мою
        фоловинг ленту, как было» — раньше один запрос давал топ 100, сейчас
        с большим числом подписок видно тонкий срез. 2-3 страницы = 200-250
        постов = настоящая 24-часовая лента подписок.
        """
        client = await self._ensure_client()
        try:
            async with _X_API_SEM:
                page = await asyncio.wait_for(
                    client.get_latest_timeline(count=40),
                    timeout=25,
                )
        except RecursionError:
            log.warning("get_latest_timeline: rate-limited by X (429 recursion)")
            return []
        except asyncio.TimeoutError:
            log.warning("get_latest_timeline: timeout")
            return []
        except Exception as e:
            log.warning("get_latest_timeline: %s", str(e)[:200])
            return []

        all_tweets = list(page)
        # Пагинация — набираем до limit. 3 шага обычно хватает: 40+40+40=120.
        # При limit=400 — до 5 страниц. 429 на странице → прерываемся, то что
        # уже набрали отдадим.
        for _ in range(5):
            if len(all_tweets) >= limit:
                break
            try:
                if not hasattr(page, "next"):
                    break
                async with _X_API_SEM:
                    page = await asyncio.wait_for(page.next(), timeout=15)
                new = list(page)
                if not new:
                    break
                all_tweets.extend(new)
                await asyncio.sleep(0.5)
            except Exception as e:
                log.debug("get_home_timeline pagination stopped: %s", e)
                break

        return self._convert_timeline(all_tweets, limit)

    async def get_for_you_timeline(self, limit: int = 100) -> list[RawTweet]:
        """Лента «For You» — алгоритмическая.

        X за один вызов отдаёт ~34 поста (независимо от count). Чтобы набрать
        ~100 — пагинируем через .next() несколько раз.
        """
        client = await self._ensure_client()
        try:
            async with _X_API_SEM:
                page = await asyncio.wait_for(client.get_timeline(count=40), timeout=25)
        except RecursionError:
            log.warning("get_timeline(for_you): rate-limited")
            return []
        except asyncio.TimeoutError:
            log.warning("get_timeline(for_you): timeout")
            return []
        except Exception as e:
            log.warning("get_timeline(for_you): %s", str(e)[:200])
            return []

        all_tweets = list(page)
        # Пагинация — 2-3 раза хватит чтобы добрать до 100.
        for _ in range(3):
            if len(all_tweets) >= limit:
                break
            try:
                if not hasattr(page, "next"):
                    break
                async with _X_API_SEM:
                    page = await asyncio.wait_for(page.next(), timeout=15)
                new = list(page)
                if not new:
                    break
                all_tweets.extend(new)
                await asyncio.sleep(0.5)  # щадящий тайминг
            except Exception as e:
                log.debug("get_for_you pagination stopped: %s", e)
                break

        return self._convert_timeline(all_tweets, limit)

    def _convert_timeline(self, tl, limit: int) -> list[RawTweet]:
        out: list[RawTweet] = []
        for t in tl:
            try:
                uname_fb = getattr(getattr(t, "user", None), "screen_name", None) or "unknown"
                out.append(_convert_tweet(t, uname_fb))
            except Exception as e:
                log.debug("convert timeline tweet failed: %s", e)
            if len(out) >= limit:
                break
        return out

    async def get_recent_tweets_for_authors(
        self,
        usernames: Iterable[str],
        limit_per_author: int = 15,
    ) -> list[RawTweet]:
        """Параллельный сбор с умеренной конкурентностью и случайной задержкой —
        чтобы X не пометил нас как bot-wave и не заморозил аккаунт."""
        sem = asyncio.Semaphore(2)

        async def _one(u: str) -> list[RawTweet]:
            async with sem:
                # Небольшая джиттер-задержка между запросами — меньше паттерна.
                await asyncio.sleep(0.3)
                return await self.get_recent_tweets(u, limit=limit_per_author)

        results = await asyncio.gather(
            *[_one(u) for u in usernames], return_exceptions=True
        )
        flat: list[RawTweet] = []
        for r in results:
            if isinstance(r, list):
                flat.extend(r)
            elif isinstance(r, Exception):
                log.debug("author fetch exception: %s", r)
        return flat

    async def search_tweets(
        self,
        query: str,
        product: str = "Top",
        count: int = 30,
    ) -> list[RawTweet]:
        """Поиск постов по ключевой фразе. product: 'Top' / 'Latest' / 'Media'.

        'Top' ранжируется X-алгоритмом — это то что сейчас активно обсуждают
        по теме, не просто последнее. Нужно для interest-driven topic discovery.

        Если endpoint возвращает 404 (X часто ротирует query_id для SearchTimeline),
        пробуем альтернативные product'ы. При полном фейле — возвращаем пусто,
        а вызывающий код должен падать на curated-fallback (`topic_authors_fallback`).
        """
        q_norm = (query or "").lower().strip()
        if not q_norm:
            return []
        # Cooldown: если эта же фраза недавно дала 404/429, не долбим X.
        if _cooldown_active(_SEARCH_COOLDOWN, q_norm):
            log.debug("search_tweets(%s): skipped — query in cooldown", query)
            return []
        client = await self._ensure_client()

        products_to_try = [product]
        # Ротация: если Top не сработал, пробуем Latest (разные endpoint-пути бывают OK).
        if product == "Top":
            products_to_try.append("Latest")
        elif product == "Latest":
            products_to_try.append("Top")

        res = None
        last_err = None
        for prod in products_to_try:
            try:
                async with _X_API_SEM:
                    res = await asyncio.wait_for(
                        client.search_tweet(query, prod, count=min(40, count)),
                        timeout=18,
                    )
                if res:
                    break
            except RecursionError:
                log.warning("search_tweets(%s/%s): rate-limited — cooldown 10min", query, prod)
                _cooldown_set(_SEARCH_COOLDOWN, q_norm, _SEARCH_COOLDOWN_SEC)
                return []
            except asyncio.TimeoutError:
                log.warning("search_tweets(%s/%s): timeout", query, prod)
                continue
            except Exception as e:
                last_err = e
                log.warning("search_tweets(%s/%s): %s", query, prod, str(e)[:200])
                continue

        if res is None:
            if last_err:
                log.info("search_tweets(%s): all products failed, caller should fallback", query)
                # 404/rate-limit — кулдаун чтобы не повторять провальный query в том же build_report.
                _cooldown_set(_SEARCH_COOLDOWN, q_norm, _SEARCH_COOLDOWN_SEC)
            return []

        out: list[RawTweet] = []
        for t in res:
            try:
                uname_fb = getattr(getattr(t, "user", None), "screen_name", None) or "unknown"
                out.append(_convert_tweet(t, uname_fb))
            except Exception as e:
                log.debug("search convert failed: %s", e)
            if len(out) >= count:
                break
        return out

    async def topic_authors_fallback(
        self,
        topic_keyword: str,
        per_author: int = 5,
        max_authors: int = 4,
        extra_authors: Optional[list[str]] = None,
    ) -> list[RawTweet]:
        """Фолбэк когда SearchTimeline вернул 404 / пусто.

        Приоритет:
          1. `extra_authors` — handle'ы от вызывающего (обычно Claude уже
             подобрал их под точный запрос юзера в `process_preference_request`).
          2. Claude `suggest_authors_for_query(topic_keyword)` — для ЛЮБОЙ
             темы модель сама даёт список реальных X-аккаунтов. Кэш на 1ч.
             Это заменяет хардкод словарей вида `{"football": [...], ...}` —
             те не покрывали «лоу-фай джаз», «реставрация мебели», «античная
             философия» и тысячи других ниш.
          3. Минимальный safety-net — если Claude/proxy лежит, отдаём
             generic-новостные хендлы, чтобы не вернуть 0.
        """
        from core import ai_client as _ai

        key = (topic_keyword or "").lower().strip()
        picked: list[str] = []

        # 1) Caller-supplied (Claude уже подобрал в preferences-флоу).
        for a in (extra_authors or []):
            if a and a.lower() not in {p.lower() for p in picked}:
                picked.append(a)

        # 2) Claude per-query suggestion (cached). Только если caller не
        # дал достаточно — иначе Claude-suggested от preferences важнее.
        if len(picked) < max_authors and key:
            try:
                claude_handles = await _ai.suggest_authors_for_query(
                    topic_keyword, n=max_authors,
                )
                for a in claude_handles:
                    if a.lower() not in {p.lower() for p in picked}:
                        picked.append(a)
                    if len(picked) >= max_authors:
                        break
            except Exception as e:
                log.debug("suggest_authors_for_query(%r) failed: %s", topic_keyword, e)

        # 3) Last-resort safety-net — generic новостные источники, только
        # если Claude вообще ничего не дал (offline/proxy down). Не пытаемся
        # подобрать «правильный» tag — просто хоть что-то вместо пустоты.
        if not picked:
            picked = ["Reuters", "AP", "FinancialTimes"]

        picked = picked[:max_authors]

        out: list[RawTweet] = []
        for uname in picked:
            try:
                got = await self.get_recent_tweets(uname, limit=per_author)
                out.extend(got)
            except Exception as e:
                log.debug("topic_authors_fallback(%s→@%s): %s", topic_keyword, uname, e)
            await asyncio.sleep(0.4)
        return out

    async def search_trending_hot(
        self,
        queries: list[str],
        count_each: int = 15,
    ) -> list[RawTweet]:
        """Набираем «hot but older» посты — то что юзер бы увидел в X по Top-search.

        Используется чтобы feed не состоял только из 19-минутных постов.
        Top-search возвращает посты отранжированные по engagement за последние сутки.
        """
        all_out: list[RawTweet] = []
        seen_ids: set[str] = set()
        for q in queries:
            if not q:
                continue
            got = await self.search_tweets(q, product="Top", count=count_each)
            if not got:
                # Fallback если search сломан — curated авторы по теме.
                got = await self.topic_authors_fallback(q, per_author=4, max_authors=3)
            for t in got:
                if t.tweet_id in seen_ids:
                    continue
                seen_ids.add(t.tweet_id)
                all_out.append(t)
            await asyncio.sleep(0.3)
        return all_out

    async def get_top_replies(
        self,
        tweet_id: str,
        limit: int = 5,
        author_username: Optional[str] = None,
    ) -> list[RawTweet]:
        """Топ залайканных реплаев под твитом. Два пути:

        1. `get_tweet_by_id` → `.replies`. Иногда twikit-парсер возвращает
           пустой список (schema drift на tweetdetail).
        2. Fallback: `search_tweet("conversation_id:<tweet_id>", Latest)` —
           X GraphQL поиск по conversation_id отдаёт все реплаи треда.
        """
        client = await self._ensure_client()

        # 1) Путь через get_tweet_by_id.
        flat: list = []
        try:
            async with _X_API_SEM:
                tw = await asyncio.wait_for(client.get_tweet_by_id(tweet_id), timeout=18)
            try:
                replies_list = list(getattr(tw, "replies", None) or [])
            except Exception:
                replies_list = []
            for r in replies_list:
                flat.append(r)
                sub = getattr(r, "replies", None)
                if sub:
                    try:
                        for sr in list(sub)[:3]:
                            flat.append(sr)
                    except Exception:
                        pass
            log.info("get_top_replies(%s): tweetdetail gave %d reply-objects",
                     tweet_id, len(flat))
        except Exception as e:
            log.warning("get_top_replies(%s) tweetdetail: %s", tweet_id, str(e)[:160])

        # 2) Fallback — поиск по conversation_id. Даёт плоский список реплаев
        # и работает даже когда tweetdetail кривой (schema drift).
        if not flat:
            # Два захода: сначала с `to:<author>` (точнее), потом без (шире).
            queries_to_try: list[str] = []
            if author_username:
                queries_to_try.append(f"conversation_id:{tweet_id} to:{author_username}")
            queries_to_try.append(f"conversation_id:{tweet_id}")

            combined: list = []
            seen_q_ids: set[str] = set()
            for q in queries_to_try:
                try:
                    res = await self.search_tweets(q, product="Latest", count=40)
                    res = [r for r in res if r.tweet_id != tweet_id]
                    for r in res:
                        if r.tweet_id not in seen_q_ids:
                            combined.append(r)
                            seen_q_ids.add(r.tweet_id)
                    log.info("get_top_replies(%s) q=%r gave %d (cum %d)",
                             tweet_id, q, len(res), len(combined))
                    if len(combined) >= limit * 2:
                        break  # хватит для ранжирования
                except Exception as e:
                    log.warning("get_top_replies(%s) q=%r: %s",
                                tweet_id, q, str(e)[:160])
            combined.sort(key=lambda r: r.likes_count + 2 * r.retweets_count, reverse=True)
            return combined[:limit]

        # Ранжируем объекты из tweetdetail-пути.
        def _score(x) -> int:
            return int(getattr(x, "favorite_count", 0) or 0) + 2 * int(getattr(x, "retweet_count", 0) or 0)
        flat.sort(key=_score, reverse=True)
        out: list[RawTweet] = []
        for t in flat[:limit]:
            try:
                uname_fb = getattr(getattr(t, "user", None), "screen_name", None) or "unknown"
                out.append(_convert_tweet(t, uname_fb))
            except Exception as e:
                log.debug("reply convert failed: %s", e)
        return out

    async def get_tweet_with_media(self, tweet_id: str) -> Optional[RawTweet]:
        """Легковесный fetch одного твита по id — для t.co expansion в delivery.

        Когда юзеру отправляется пост вида «implementation by @kianbazza t.co/X»
        без своего медиа, а у kianbazza видео — мы достаём kianbazza-твит
        и берём его media. Один X-запрос, кэшируем не надо (delivery редкий).
        """
        if not tweet_id:
            return None
        client = await self._ensure_client()
        try:
            async with _X_API_SEM:
                tw = await asyncio.wait_for(
                    client.get_tweet_by_id(tweet_id), timeout=12,
                )
        except Exception as e:
            log.debug("get_tweet_with_media(%s): %s", tweet_id, str(e)[:120])
            return None
        try:
            uname = getattr(getattr(tw, "user", None), "screen_name", None) or "unknown"
            return _convert_tweet(tw, uname)
        except Exception as e:
            log.debug("get_tweet_with_media(%s) convert: %s", tweet_id, e)
            return None

    async def search_users(self, query: str, count: int = 10) -> list[AuthorInfo]:
        """Поиск аккаунтов по ключевой фразе — для auto-add в «больше/меньше».

        Если X-поиск отвалился (404/rate-limit), падаем на curated-список
        по теме (те же авторы что в topic_authors_fallback) и подтягиваем
        AuthorInfo по ним через get_author_info — так хотя бы
        «добавить @voguemagazine при просьбе про моду» сработает.
        """
        if not query or not query.strip():
            return []
        client = await self._ensure_client()

        out: list[AuthorInfo] = []
        res = None
        try:
            async with _X_API_SEM:
                res = await asyncio.wait_for(
                    client.search_user(query, count=min(20, count)),
                    timeout=18,
                )
        except Exception as e:
            log.warning("search_users(%s): %s", query, str(e)[:200])

        if res:
            for u in res:
                try:
                    followers = int(getattr(u, "followers_count", 0) or 0)
                    verified = bool(
                        getattr(u, "verified", False) or getattr(u, "is_blue_verified", False)
                    )
                    out.append(AuthorInfo(
                        username=getattr(u, "screen_name", "") or "",
                        display_name=getattr(u, "name", "") or "",
                        followers_count=followers,
                        following_count=int(getattr(u, "following_count", 0) or 0),
                        verified=verified,
                        account_age_days=365,
                    ))
                except Exception as e:
                    log.debug("search_users convert failed: %s", e)
                if len(out) >= count:
                    break

        # CURATED FALLBACK — если X ничего не вернул (или мало), добиваем ручным списком.
        if len(out) < 2:
            key = query.lower()
            curated_by_topic: dict[str, list[str]] = {
                "fashion": ["voguemagazine", "voguerunway", "highsnobiety",
                            "hypebeast", "businessoffashion", "ssense"],
                "luxury":  ["voguerunway", "businessoffashion", "netaporter", "mrporterlive"],
                "streetwear": ["highsnobiety", "hypebeast", "complex", "stockx"],
                "ai":       ["OpenAI", "AnthropicAI", "GoogleDeepMind", "xai",
                             "sama", "karpathy", "ylecun"],
                "crypto":   ["cz_binance", "VitalikButerin", "coinbase", "binance"],
                "tech":     ["techcrunch", "verge", "paulg", "pmarca", "naval"],
                "science":  ["nature", "ScienceMagazine", "NewScientist", "NASA"],
                "gaming":   ["IGN", "polygon", "Kotaku", "Xbox", "PlayStation"],
                "sports":   ["espn", "SportsCenter", "FIFAcom"],
                "culture":  ["voguemagazine", "gq", "newyorker", "Pitchfork"],
                "politics": ["Reuters", "FinancialTimes", "TheEconomist", "BBCBreaking"],
                "business": ["FinancialTimes", "WSJ", "TheEconomist", "Reuters"],
                "news":     ["Reuters", "AP", "BBCBreaking", "FinancialTimes"],
            }
            picked: list[str] = []
            for marker, authors in curated_by_topic.items():
                if marker in key or any(w in key for w in marker.split()):
                    for a in authors:
                        if a not in picked and not any(
                            o.username.lower() == a.lower() for o in out
                        ):
                            picked.append(a)
            for uname in picked[:count]:
                info = await self.get_author_info(uname)
                if info:
                    out.append(info)
                if len(out) >= count:
                    break

        # Ранжируем: проверенные и с большой аудиторией — выше.
        out.sort(key=lambda a: (a.verified, a.followers_count), reverse=True)
        return out[:count]

    async def save_cookies_from_dict(self, cookies: dict) -> None:
        """Утилита: пользователь прислал {'auth_token': '...', 'ct0': '...'} — сохраняем в файл."""
        path = Path(settings.x_cookies_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cookies), encoding="utf-8")
        # Сбрасываем клиента, чтобы при следующем вызове подхватились новые.
        self._client = None
        self._auth_ok = False


# ----------------------- helpers -----------------------


_HASHTAG_RE = re.compile(r"#(\w+)", re.UNICODE)


def _convert_tweet(t, fallback_author: str) -> RawTweet:
    """Нормализует twikit.Tweet в наш RawTweet."""
    tweet_id = str(getattr(t, "id", "") or "")
    if not tweet_id:
        raise ValueError("empty tweet id")

    author_obj = getattr(t, "user", None)
    author_username = (
        getattr(author_obj, "screen_name", None)
        or getattr(author_obj, "username", None)
        or fallback_author
    )
    # Защита от «нейрослопа»: SearchTimeline иногда отдаёт твиты где user
    # не получилось распарсить — в этом случае `fallback_author` остаётся
    # равным 'unknown' (дефолт вызывающих). Такие посты обычно AI-сгенерированный
    # мусор низкого качества (fashion-слоп, crypto-spam, generic templates),
    # и пользователю они только портят ленту. Откидываем.
    if not author_username or author_username.lower() == "unknown":
        raise ValueError(f"no author for tweet {tweet_id}")
    display = getattr(author_obj, "name", None) or author_username

    text = getattr(t, "full_text", None) or getattr(t, "text", "") or ""

    # Убираем висячие t.co ссылки везде (в хвосте и дубликаты).
    # Telegram в caption оставляет их как «https://t.co/...», что выглядит мусорно.
    text = re.sub(r"\s*https?://t\.co/\S+", "", text).strip()

    # Квоты: если пост — реакция на чужой твит (Peter Yang: «Anthropic has no chill» +
    # quote @claudeai с видео про Claude Design), без контекста это выглядит
    # как голый однострочник. Подклеим короткий payload квоты в текст, чтобы
    # юзер понимал на что реагируют.
    quoted_obj = getattr(t, "quote", None)
    if quoted_obj is not None:
        q_author = getattr(getattr(quoted_obj, "user", None), "screen_name", None) or "—"
        q_text = (getattr(quoted_obj, "full_text", None) or getattr(quoted_obj, "text", "") or "")
        q_text = q_text.strip()
        # Убираем t.co из quote text — обычно это ссылка обратно на сам квот-твит
        # (при самоцитировании) или битая ссылка, в caption только шум.
        q_text = re.sub(r"\s*https?://t\.co/\S+", "", q_text).strip()
        if q_text:
            if len(q_text) > 260:
                q_text = q_text[:259].rstrip() + "…"
            # Формат «↪ цитирует @user: ...» — явный и читается сразу, без
            # пунктирных разделителей или чёрточки. Так видно что это вставка
            # чужого текста, а не продолжение автора.
            prefix = "\n\n↪ цитирует " if text.strip() else "↪ цитирует "
            text = f"{text}{prefix}@{q_author}: {q_text}"

    # Медиа: фото, видео, gif. Приоритет photo > video > gif.
    # Для video/gif берём mp4 с максимальным битрейтом из variants.
    media_url: Optional[str] = None
    media_type: Optional[str] = None
    media_list = getattr(t, "media", None) or []

    def _best_mp4(media_obj) -> Optional[str]:
        """Из media с видео достаём mp4 с наибольшим bitrate.

        В twikit 2.x Stream.content_type читает `_data['content-type']` (с дефисом),
        но Twitter в JSON-данных variants использует ключ `content_type` (с подчёркиванием)
        — property возвращает None. Поэтому читаем сырой variants напрямую из video_info.
        """
        try:
            vi = getattr(media_obj, "video_info", None) or {}
            variants = vi.get("variants") if isinstance(vi, dict) else None
            best_url, best_br = None, -1
            if variants:
                for v in variants:
                    ct = str(v.get("content_type") or v.get("content-type") or "").lower()
                    url_v = v.get("url")
                    try:
                        br = int(v.get("bitrate") or 0)
                    except (TypeError, ValueError):
                        br = 0
                    # Принимаем любые mp4 (в URL или в content_type). HLS (application/x-mpegURL)
                    # явно пропускаем — Telegram не умеет играть m3u8.
                    if not url_v:
                        continue
                    if "mpegurl" in ct or url_v.endswith(".m3u8"):
                        continue
                    is_mp4 = ("mp4" in ct) or (".mp4" in url_v.lower())
                    if is_mp4 and br > best_br:
                        best_br = br
                        best_url = url_v
            # Фолбэк: обычный .streams (для старых версий twikit и photo-like объектов).
            if not best_url:
                streams = getattr(media_obj, "streams", None) or []
                for st in streams:
                    raw = getattr(st, "_data", None) or {}
                    ct = str(
                        raw.get("content_type")
                        or raw.get("content-type")
                        or getattr(st, "content_type", "")
                        or ""
                    ).lower()
                    url_v = getattr(st, "url", None) or raw.get("url")
                    try:
                        br = int(getattr(st, "bitrate", 0) or raw.get("bitrate") or 0)
                    except (TypeError, ValueError):
                        br = 0
                    if not url_v:
                        continue
                    if "mpegurl" in ct or url_v.endswith(".m3u8"):
                        continue
                    is_mp4 = ("mp4" in ct) or (".mp4" in url_v.lower())
                    if is_mp4 and br > best_br:
                        best_br = br
                        best_url = url_v
            return best_url
        except Exception as e:
            log.debug("_best_mp4 failed: %s", e)
            return None

    # Первый проход — photo (приоритет).
    for m in media_list:
        mtype = (getattr(m, "type", None) or getattr(m, "media_type", None) or "").lower()
        if mtype in ("photo", "image"):
            media_url = (
                getattr(m, "media_url_https", None)
                or getattr(m, "media_url", None)
                or getattr(m, "url", None)
            )
            if media_url:
                media_type = "photo"
                break

    # Если фото не нашли — пробуем видео / gif.
    if not media_url:
        for m in media_list:
            mtype = (getattr(m, "type", None) or "").lower()
            if mtype == "video":
                mp4 = _best_mp4(m)
                if mp4:
                    media_url = mp4
                    media_type = "video"
                    break
            if mtype in ("animated_gif", "gif"):
                mp4 = _best_mp4(m)
                if mp4:
                    media_url = mp4
                    media_type = "animation"  # Telegram отдельно: send_animation
                    break

    # Фолбэки: если прямой media нет, пробуем
    #   (a) thumbnail_url карточки (URL-превью)
    #   (b) media из quoted tweet (если пост цитирует твит с картинкой)
    #   (c) media из retweeted_tweet (чистый RT — но мы их фильтруем выше,
    #       так что это фолбэк «на всякий»).
    def _extract_from_media_list(ml) -> tuple[Optional[str], Optional[str]]:
        for m in (ml or []):
            mt = (getattr(m, "type", None) or getattr(m, "media_type", None) or "").lower()
            if mt in ("photo", "image"):
                u = (getattr(m, "media_url_https", None)
                     or getattr(m, "media_url", None)
                     or getattr(m, "url", None))
                if u:
                    return u, "photo"
            elif mt == "video":
                u = _best_mp4(m)
                if u:
                    return u, "video"
            elif mt in ("animated_gif", "gif"):
                u = _best_mp4(m)
                if u:
                    return u, "animation"
        return None, None

    if not media_url:
        # (a) card thumbnail
        try:
            thumb = getattr(t, "thumbnail_url", None)
            if thumb:
                media_url = thumb
                media_type = "photo"
        except Exception:
            pass
    if not media_url and quoted_obj is not None:
        # (b) quoted tweet's media — сначала прямое media, потом card-thumbnail.
        try:
            qm = getattr(quoted_obj, "media", None) or []
            u, mt = _extract_from_media_list(qm)
            if u:
                media_url, media_type = u, mt
        except Exception:
            pass
        if not media_url:
            try:
                q_thumb = getattr(quoted_obj, "thumbnail_url", None)
                if q_thumb:
                    media_url = q_thumb
                    media_type = "photo"
            except Exception:
                pass

    # DUAL MEDIA: если у автора ЕСТЬ своё медиа И в квоте тоже есть медиа —
    # сохраняем его в quote_image_url, чтобы delivery слал два сообщения
    # (пост автора с автор-видео, потом quote-text с quote-видео).
    quote_image_url: Optional[str] = None
    quote_media_type: Optional[str] = None
    if media_url and quoted_obj is not None:
        try:
            qm = getattr(quoted_obj, "media", None) or []
            u, mt = _extract_from_media_list(qm)
            if u and u != media_url:
                quote_image_url = u
                quote_media_type = mt
            if not quote_image_url:
                q_thumb = getattr(quoted_obj, "thumbnail_url", None)
                if q_thumb and q_thumb != media_url:
                    quote_image_url = q_thumb
                    quote_media_type = "photo"
        except Exception:
            pass
    if not media_url:
        # (c) retweeted tweet's media (last resort)
        try:
            rt_obj = getattr(t, "retweeted_tweet", None)
            if rt_obj is not None:
                rm = getattr(rt_obj, "media", None) or []
                u, mt = _extract_from_media_list(rm)
                if u:
                    media_url, media_type = u, mt
        except Exception:
            pass

    image_url = media_url

    created_at = getattr(t, "created_at_datetime", None) or getattr(t, "created_at", None)
    if isinstance(created_at, str):
        # twikit иногда отдаёт "Wed Oct 10 20:19:24 +0000 2018" — Twitter legacy format.
        for fmt in ("%a %b %d %H:%M:%S %z %Y", "%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z"):
            try:
                created_at = datetime.strptime(created_at, fmt)
                break
            except ValueError:
                continue
        if isinstance(created_at, str):
            created_at = datetime.now(timezone.utc)
    if not isinstance(created_at, datetime):
        created_at = datetime.now(timezone.utc)
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)

    likes = int(getattr(t, "favorite_count", 0) or 0)
    rts = int(getattr(t, "retweet_count", 0) or 0)
    replies = int(getattr(t, "reply_count", 0) or 0)

    # Чистый ретвит без комментария: у twikit поле retweeted_tweet с оригиналом.
    is_rt = bool(getattr(t, "retweeted_tweet", None)) and not bool(getattr(t, "is_quote_status", False))
    if not is_rt and text.strip().startswith("RT @"):
        is_rt = True

    url = f"https://x.com/{author_username}/status/{tweet_id}"
    hashtags_raw = getattr(t, "hashtags", None)
    if hashtags_raw and isinstance(hashtags_raw[0], str):
        hashtags = list(hashtags_raw)
    elif hashtags_raw:
        hashtags = [getattr(h, "text", str(h)) for h in hashtags_raw]
    else:
        hashtags = _HASHTAG_RE.findall(text)

    # Достаём автор и текст квоты отдельно (для 2-сообщения режима).
    q_author_uname = None
    q_text_only = None
    if quoted_obj is not None:
        q_author_uname = getattr(getattr(quoted_obj, "user", None), "screen_name", None)
        q_text_only = (getattr(quoted_obj, "full_text", None)
                       or getattr(quoted_obj, "text", "") or "").strip()
        if q_text_only:
            q_text_only = re.sub(r"\s*https?://t\.co/\S+", "", q_text_only).strip()

    # Linked tweet: если в URL-entities есть expanded_url типа
    # `https://x.com/<user>/status/<id>` — это значит автор embed'ит чужой
    # твит через t.co (не native quote). Сохраняем target id, чтобы delivery
    # мог подтянуть его медиа когда у текущего поста медиа нет.
    linked_tweet_id_val = None
    url_entities = getattr(t, "urls", None) or []
    for ue in url_entities:
        expanded = (
            ue.get("expanded_url") if isinstance(ue, dict)
            else getattr(ue, "expanded_url", None)
        )
        if not expanded:
            continue
        m = re.search(r"(?:twitter|x)\.com/[^/]+/status/(\d{5,25})", expanded)
        if m:
            linked_tweet_id_val = m.group(1)
            break

    return RawTweet(
        tweet_id=tweet_id,
        author_username=author_username,
        author_display_name=display,
        text=text.strip(),
        url=url,
        image_url=image_url,
        created_at=created_at,
        likes_count=likes,
        retweets_count=rts,
        replies_count=replies,
        is_retweet_no_comment=is_rt,
        hashtags=hashtags,
        media_type=media_type,
        quote_image_url=quote_image_url,
        quote_media_type=quote_media_type,
        quote_author=q_author_uname,
        quote_text=q_text_only,
        linked_tweet_id=linked_tweet_id_val,
    )


def compute_trust_score(author: AuthorInfo) -> float:
    """0..1. Log10 от followers, возраст, verified, engagement."""
    followers_score = min(1.0, math.log10(max(1, author.followers_count) + 1) / 7.0)
    age_score = min(1.0, author.account_age_days / (365 * 5))
    verified_bonus = 0.2 if author.verified else 0.0
    engagement_bonus = min(0.2, author.recent_engagement_rate * 10.0)
    raw = 0.45 * followers_score + 0.25 * age_score + verified_bonus + engagement_bonus
    return max(0.0, min(1.0, raw))


# Модуль-синглтон.
parser = XParser()
