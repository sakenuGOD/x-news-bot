"""Кнопка «Хочу больше…» — свободный текст → Claude Haiku → boost/suppress кластеров."""

from __future__ import annotations

import logging
from typing import Optional

from aiogram import F, Router
from aiogram.types import CallbackQuery, Message

from core import ai_client
from core.recommender import apply_cluster_weight_update
from db.database import session_scope
from db.models import User

log = logging.getLogger(__name__)

router = Router(name="preferences")


@router.callback_query(F.data == "ctl:more")
async def handle_more_click(cb: CallbackQuery) -> None:
    async with session_scope() as s:
        user = await s.get(User, cb.from_user.id)
        if not user:
            await cb.answer("Сначала /start", show_alert=True)
            return
        user.onboarding_state = "awaiting_more_text"

    await cb.message.answer(
        "Напиши свободно что хочешь видеть чаще (и что меньше).\n"
        "Пример: <i>«хочу больше про AI и стартапы, меньше политики»</i>",
        parse_mode="HTML",
    )
    await cb.answer()


async def apply_preference_text(message: Message, text: str) -> None:
    """Обработка «хочу больше…» — обновляем веса кластеров, auto-add аккаунтов
    по темам которые пользователь попросил усилить, и СРАЗУ подтягиваем свежие
    посты по теме (чтобы «больше моды» дало реальный новый контент, а не
    +1 пост из остатков пула).
    """
    from bot.keyboards import main_menu_kb
    from core.x_parser import parser
    from db.models import FollowedAuthor
    from sqlalchemy import select
    import re as _re

    # Эвристический маппинг нестандартных запросов на кластеры. Claude иногда
    # не может сопоставить «японский стиль одежды» → lifestyle и возвращает
    # пустой boost. Этот словарь подхватывает такие случаи детерминированно,
    # плюс генерирует поисковый запрос на инглише для X search.
    # Эвристика — ТОЛЬКО на случай если Claude полностью обнулился. Ключевое:
    # word-boundary там где нужно, чтобы «япон**ии**» не триггерил AI-матч.
    _HEURISTIC = [
        # (regex lower, cluster, fallback_search_query)
        (r"(япон|\bjapan\b|tokyo|токио|harajuku|uniqlo)", "lifestyle", "japanese fashion"),
        (r"(корей|\bkorean\b|k-?pop|k-?beauty|сеул)", "culture", "korean style"),
        (r"(мод[ау]|\bfashion\b|одежд|outfit|streetwear)", "lifestyle", "fashion trends"),
        (r"(бренд|luxury|gucci|prada|hermes|vuitton)", "lifestyle", "luxury fashion"),
        (r"(beaut|красот|skincare|уход за кож)", "lifestyle", "skincare beauty"),
        # AI — только ПОЛНЫЕ слова («ии» как отдельное слово, не внутри «япон-ии»).
        (r"(\bai\b|\bии\b|\bllm\b|claude|gpt|anthropic|agent\b)", "ai", "AI agents"),
        (r"(startup|стартап|founder|vc)", "business", "startup funding"),
        (r"(crypto|крипт|bitcoin|eth|defi)", "crypto", "crypto market"),
        (r"(\bspace\b|nasa|космос|starship)", "science", "space discovery"),
        (r"(\bgame\b|игр[ыау]|gaming|playstation|xbox|nintendo)", "gaming", "game release"),
        (r"(football|футбол|basketball|nba|nfl|soccer)", "sports", "sports highlight"),
        (r"(music|музык|spotify|album|rap|hip-?hop)", "culture", "music release"),
        (r"(film|movi|фильм|series|сериал)", "culture", "film release"),
        (r"(politic|политик|election|trump|biden)", "politics", "politics breaking"),
    ]

    # Паттерн «убери/исключи @handle» — РАНЬШЕ вызова Claude.
    # Если юзер просит убрать конкретный аккаунт — НЕ гоним suppress по кластеру
    # (раньше «убери @Reuters» давало suppress=news и лента новостей становилась
    # хуже; блокировка только точечная).
    _EXCL_RE = _re.compile(
        r"(?:исключ\w*|убер\w*|без|вычеркн\w*|не показ\w+|удал\w*|блок\w*)\s+@?"
        r"([A-Za-z0-9_]{2,15})",
        _re.IGNORECASE,
    )
    manual_blocked = [m.group(1) for m in _EXCL_RE.finditer(text) if m.group(1)]
    is_pure_block_request = bool(manual_blocked) and not _re.search(
        r"(больш\w+|хоч\w+ еще|ещё|усил\w+|добав\w+|\bmore\b)", text, _re.IGNORECASE,
    )

    async with session_scope() as s:
        user = await s.get(User, message.from_user.id)
        if not user:
            return
        current = user.cluster_weights or {}
        result = await ai_client.process_preference_request(text, current)

        # Добавляем manually-extracted handles в blocked_accounts.
        for h in manual_blocked:
            if h and h.lower() not in [b.lower() for b in (result.blocked_accounts or [])]:
                result.blocked_accounts.append(h)

        # Если запрос ЧИСТО про блокировку («убери @Reuters» без «больше чего-то») —
        # игнорируем suppress от Claude (иначе news-кластер понижается просто
        # потому что Reuters — новостной аккаунт, а юзер не просил меньше новостей).
        if is_pure_block_request:
            if result.suppress:
                log.info("pure block request — discarding Claude suppress=%s", result.suppress)
            result.suppress = []

        lower_text = text.lower()
        claude_queries_raw = list(getattr(result, "search_queries", []) or [])
        if not result.boost and not result.suppress and not is_pure_block_request:
            for pat, cluster, q in _HEURISTIC:
                if _re.search(pat, lower_text):
                    if cluster not in result.boost:
                        result.boost.append(cluster)
                    if q not in claude_queries_raw:
                        claude_queries_raw.append(q)
            if result.boost:
                # Перепишем reply — что реально сделали.
                result.reply = (
                    f"Понял: тема «{text.strip()[:60]}». "
                    f"Усилил кластеры: {', '.join(result.boost)}. "
                    f"Буду искать: {', '.join(claude_queries_raw) or '—'}."
                )

        # Сильнее сдвигаем веса: «хочу больше про X» должно реально затоплять
        # ленту этой темой, а не робко добавлять 1 пост на 100. +0.45 за раз,
        # ослабление -0.40 — юзер это и просил: «реально помимо моей ленты
        # анализирует горячие аккаунты связанные с темой».
        new_weights = apply_cluster_weight_update(
            current, result.boost, result.suppress,
            boost_delta=0.45, suppress_delta=0.40,
        )
        user.cluster_weights = new_weights
        user.onboarding_state = None

        # Применяем blocked_accounts: добавляем в user.blocked_authors (lowercase,
        # без @). Также УДАЛЯЕМ эти аккаунты из FollowedAuthor — чтобы их посты
        # не просачивались в фетче подписок.
        if result.blocked_accounts:
            blocked_now = list(user.blocked_authors or [])
            new_blocked_for_report: list[str] = []
            for h in result.blocked_accounts:
                h_low = h.lower().strip()
                if h_low and h_low not in [b.lower() for b in blocked_now]:
                    blocked_now.append(h_low)
                    new_blocked_for_report.append(h)
            user.blocked_authors = blocked_now
            # Чистим из FollowedAuthor (case-insensitive).
            if new_blocked_for_report:
                from db.models import FollowedAuthor as _FA
                lowered = [h.lower() for h in new_blocked_for_report]
                fa_rows = (await s.execute(
                    select(_FA).where(_FA.user_id == user.telegram_id)
                )).scalars().all()
                for fa in fa_rows:
                    if fa.author_username.lower() in lowered:
                        await s.delete(fa)

        # Всегда сохраняем search_queries (даже если Claude дал boost — юзер
        # может точно попросить «японский стиль», и «fashion trends» без
        # японского акцента не подхватит нужный контент). Плюс — даже если
        # boost/suppress пусто, сам текст пользователя используем как поисковый
        # запрос (переводим только первые 60 символов, без лишнего мусора).
        if text.strip() and not claude_queries_raw and not is_pure_block_request:
            claude_queries_raw.append(text.strip()[:80])

        if claude_queries_raw:
            payload = dict(user.onboarding_payload or {})
            saved = list(payload.get("saved_search_queries") or [])
            for q in claude_queries_raw:
                if isinstance(q, str) and q.strip() and q not in saved:
                    saved.append(q.strip())
            # Было 10 — старые запросы протекали в interest-search. Юзер
            # пишет «про одежду», а в ленте «graphic design» с 2 недель назад.
            # 5 последних = текущий интерес юзера + немного фона для
            # диверсификации, без конфликтующих призраков.
            saved = saved[-5:]
            payload["saved_search_queries"] = saved
            user.onboarding_payload = payload

        paused = bool(user.paused)
        user_id = user.telegram_id

    boosted = ", ".join(result.boost) or "—"
    suppressed = ", ".join(result.suppress) or "—"

    # AUTO-ADD — строго по тому что Claude РЕАЛЬНО предложил для темы запроса.
    # Никаких хардкод-словарей «fashion → voguemagazine»: модель сама подбирает
    # релевантные X-аккаунты из своего знания (для «японский стиль одежды» это
    # voguejapan, hypebeastjp, fashionsnap и т.п. — а не generic voguemagazine).
    # Мы лишь валидируем что handle существует в X через get_author_info.
    added_accounts: list[str] = []
    skipped_accounts: list[str] = []
    suggested = list(getattr(result, "suggested_accounts", []) or [])

    async with session_scope() as s:
        from sqlalchemy import update as _sa_update, func as _sa_func
        rows = (await s.execute(
            select(FollowedAuthor.author_username).where(FollowedAuthor.user_id == user_id)
        )).all()
        already = {r[0].lower() for r in rows}

        log.info("auto-add: user=%s claude-suggested=%s already_count=%d",
                 user_id, suggested, len(already))

        # Список handle которые Claude предложил повторно (уже были): обновляем
        # added_at=NOW чтобы pinned cluster «Свежее от добавленных» сработал.
        # Без этого: юзер третий раз пишет «больше моды», бот находит те же
        # @Highsnobiety/@GQ, пропускает (уже есть), added_at остался с прошлой
        # недели → fresh_cutoff 6h не видит → pin не создаётся.
        refreshed_handles: list[str] = []

        for handle in suggested[:8]:
            if not handle:
                continue
            if handle.lower() in already:
                refreshed_handles.append(handle)
                continue
            # Проверяем что аккаунт реально есть в X.
            info = None
            try:
                info = await parser.get_author_info(handle)
            except Exception as e:
                log.debug("validate @%s failed: %s", handle, e)
            if not info:
                skipped_accounts.append(f"@{handle}")
                log.info("auto-add: skip @%s (not found in X)", handle)
                continue
            # Мягкий фильтр на очевидные dead accounts: должны быть хотя бы
            # минимальные подписчики ИЛИ verified. Не обрезаем слишком агрессивно —
            # нишевые fashion-профили живут с 2-5k followers.
            if not info.verified and info.followers_count < 500:
                skipped_accounts.append(f"@{handle}")
                log.info("auto-add: skip @%s (followers=%d, not verified)",
                         handle, info.followers_count)
                continue
            # Добавляем с weight=2.0 — это специально подобранные под интерес
            # юзера аккаунты, они должны ДОМИНИРОВАТЬ в рекомендациях пока юзер
            # не нащупал что именно нравится. В recommender.score_tweet бонус
            # = (w-1)*0.25 → для 2.0 это +25% к скору, достаточно чтобы новые
            # каналы массово заходили в ленту сразу после «хочу больше X».
            s.add(FollowedAuthor(user_id=user_id, author_username=info.username, weight=2.0))
            already.add(info.username.lower())
            added_accounts.append(f"@{info.username}")
            refreshed_handles.append(info.username)
            log.info("auto-add: user=%s +@%s (followers=%d verified=%s)",
                     user_id, info.username, info.followers_count, info.verified)
            if len(added_accounts) >= 6:
                break

        # Refresh added_at для уже существующих handle — иначе при повторном
        # «хочу больше моды» pinned cluster «Свежее от добавленных» не
        # срабатывает (added_at старый → за пределами 6h fresh_cutoff).
        if refreshed_handles:
            lowered = [h.lower() for h in refreshed_handles]
            await s.execute(
                _sa_update(FollowedAuthor)
                .where(FollowedAuthor.user_id == user_id)
                .where(_sa_func.lower(FollowedAuthor.author_username).in_(lowered))
                .values(added_at=_sa_func.now())
            )
            log.info("auto-add: refreshed added_at for %d handles", len(refreshed_handles))

    # IMMEDIATE FETCH — ключевое отличие от «просто +1 поста». Юзер сказал
    # «больше моды» → сразу идём в X: (1) search по claude-queries, (2) последние
    # посты от added_accounts. Сохраняем в Tweet с embedding — следующий отчёт
    # увидит этот свежий пул и отдаст его. Так «больше моды» реально приносит
    # десятки новых постов, а не надежду на следующий random-fetch.
    fetched_new = 0
    if not is_pure_block_request and (claude_queries_raw or added_accounts):
        try:
            fetched_new = await _immediate_topic_fetch(
                user_id, claude_queries_raw, added_accounts,
                raw_user_text=text,
            )
        except Exception as e:
            log.warning("immediate fetch for user=%s failed: %s", user_id, e)

    blocked_list = result.blocked_accounts or []
    msg_lines = [result.reply, ""]
    if not is_pure_block_request:
        msg_lines.append(f"<b>Усилил:</b> {boosted}")
        msg_lines.append(f"<b>Ослабил:</b> {suppressed}")
    if blocked_list:
        msg_lines.append(f"<b>Больше не показываю:</b> {', '.join('@' + h for h in blocked_list)}")
    if added_accounts:
        msg_lines.append("")
        msg_lines.append(f"<b>Нашёл по теме и добавил:</b> {', '.join(added_accounts)}")
    if fetched_new > 0:
        msg_lines.append(f"<i>Сразу подтянул {fetched_new} свежих постов по теме — "
                         f"будут в следующем отчёте.</i>")
    elif added_accounts:
        msg_lines.append("<i>Новые каналы подключил — начнут появляться в отчётах.</i>")
    elif suggested and not is_pure_block_request:
        msg_lines.append("")
        msg_lines.append("<i>Предложенные аккаунты не прошли проверку (нет в X или слишком "
                         "маленькие) — выдача пойдёт через поиск по теме.</i>")

    await message.answer(
        "\n".join(msg_lines),
        parse_mode="HTML",
        reply_markup=main_menu_kb(paused=paused),
    )


async def _immediate_topic_fetch(
    user_id: int,
    queries: list[str],
    added_account_strings: list[str],
    *,
    raw_user_text: str = "",
) -> int:
    """Сразу после «хочу больше X» — тянем из X посты по запросам и от новых
    каналов, сохраняем с embedding. Возвращаем сколько новых твитов добавили.

    Без этого `apply_cluster_weight_update` только двигает веса, а контент
    появится только когда следующий scheduler-fetch сам случайно наткнётся на
    посты темы. Юзер жаловался: «хочу больше — добавляет +1 пост».

    Валидация релевантности: эмбеддим эталон (сумма text+queries) и дропаем
    посты с cosine similarity < 0.28 — это защищает от кейса «попросил
    японский стиль → bot вернул generic fashion-feed» (жалоба юзера про
    Мет Галу в лент[е]).
    """
    from core.x_parser import parser
    from core import embeddings as emb
    from core.filters import is_low_signal
    from core.report import _upsert_raw_tweets
    from db.database import session_scope
    from db.models import FollowedAuthor
    from sqlalchemy import select as _select, desc as _desc

    all_raw = []
    seen_ids: set[str] = set()

    # 1) X-search по каждому Claude-запросу (top + latest). Большой захват:
    # юзер прямо попросил «больше X» — даём ему реально много свежего контента.
    # Раньше было 40+30=70 на запрос, всего ~350. Теперь до 80 на запрос — это
    # ~600+ постов кандидатов на 5 запросов после дедупа. После similarity-гейта
    # и трэш-фильтра в DB упадёт 60-100 валидных постов темы.
    for q in (queries or [])[:5]:
        if not q or not q.strip():
            continue
        try:
            top = await parser.search_tweets(q, product="Top", count=80)
            for t in top:
                if t.tweet_id not in seen_ids:
                    all_raw.append(t)
                    seen_ids.add(t.tweet_id)
            latest = await parser.search_tweets(q, product="Latest", count=80)
            for t in latest:
                if t.tweet_id not in seen_ids:
                    all_raw.append(t)
                    seen_ids.add(t.tweet_id)
        except Exception as e:
            log.debug("immediate search(%s): %s", q, e)

    # 2) Последние посты от добавленных каналов + recent FollowedAuthor.
    handles = [a.lstrip("@") for a in (added_account_strings or []) if a]
    async with session_scope() as s:
        rows = (await s.execute(
            _select(FollowedAuthor.author_username)
            .where(FollowedAuthor.user_id == user_id)
            .order_by(_desc(FollowedAuthor.added_at))
            .limit(12)
        )).all()
        recent_fa = [r[0] for r in rows if r[0] and r[0] not in handles]
    fetch_handles = handles + recent_fa

    if fetch_handles:
        try:
            # 12 авторов × 12 постов = до 144 кандидатов от подобранных каналов.
            # Раньше: 8×8=64. Юзер видит больше каналов и больше их недавних постов.
            author_tweets = await parser.get_recent_tweets_for_authors(
                fetch_handles[:12], limit_per_author=12,
            )
            for t in author_tweets:
                if t.tweet_id not in seen_ids:
                    all_raw.append(t)
                    seen_ids.add(t.tweet_id)
        except Exception as e:
            log.debug("immediate author-fetch: %s", e)

    # Фильтруем мусор, чтобы не засорить Tweet.
    clean = []
    for t in all_raw:
        low, _ = is_low_signal(t, hype_threshold=0.5)
        if low:
            continue
        clean.append(t)

    if not clean:
        log.info("immediate fetch user=%s: 0 clean tweets (raw=%d, queries=%s)",
                 user_id, len(all_raw), queries)
        return 0

    # Embedding + topic similarity guard.
    # Эталон темы = сырой текст юзера + Claude-queries. Сырой текст даёт
    # более богатый вектор (несколько слов + контекст), чем 3-словные queries
    # поодиночке. Иначе короткие queries дают слабый anchor и валидный fashion-
    # контент отсеивается (в логах видел low_sim_drop=37 из 51).
    anchor_parts: list[str] = []
    if raw_user_text:
        anchor_parts.append(raw_user_text)
    anchor_parts.extend([q for q in (queries or []) if q])
    topic_anchor_text = " ".join(anchor_parts).strip()
    topic_anchor_emb: Optional[list[float]] = None
    if topic_anchor_text:
        anchor_embs = await emb.embed_batch([topic_anchor_text])
        if anchor_embs and anchor_embs[0]:
            topic_anchor_emb = anchor_embs[0]

    texts = [t.text for t in clean]
    embs = await emb.embed_batch(texts)
    pairs_raw = [(rt, e) for rt, e in zip(clean, embs) if e]

    # Similarity-гейт. Было 0.18 — пропускал «hairy guy video», quote-tweet с
    # NSFW, и spam-посты про одежду от farm-аккаунтов. Поднимаем порог до 0.25
    # — это всё ещё пропускает ассоциированный контент (streetwear → fashion
    # news), но режет явные off-topic с единственным совпадающим словом
    # («clothes» в NSFW-цитате).
    pairs: list[tuple] = []
    filtered_low_sim = 0
    if topic_anchor_emb and pairs_raw:
        scored = [(rt, e, emb.cosine_similarity(e, topic_anchor_emb)) for rt, e in pairs_raw]
        scored.sort(key=lambda x: x[2], reverse=True)
        hard_cutoff = 0.25
        for rt, e, sim in scored:
            if sim < hard_cutoff:
                filtered_low_sim += 1
                continue
            pairs.append((rt, e))
    else:
        pairs = pairs_raw

    if not pairs:
        log.info("immediate fetch user=%s: 0 after similarity filter (raw=%d, low_sim=%d)",
                 user_id, len(pairs_raw), filtered_low_sim)
        return 0

    async with session_scope() as s:
        before = len(pairs)
        await _upsert_raw_tweets(s, pairs)

        # Сохраняем upserted tweet_ids в user.onboarding_payload — build_report
        # их достанет и добавит в кластеризацию. Без этого 55 immediate-fetch
        # постов в БД есть, но build_report их не видит (он делает свежий
        # get_home_timeline и не заглядывает в pending Tweet'ы).
        pending_ids = [rt.tweet_id for rt, _ in pairs]
        if pending_ids:
            from db.models import User as _U
            user_row = await s.get(_U, user_id)
            if user_row:
                payload = dict(user_row.onboarding_payload or {})
                existing_pending = list(payload.get("pending_boost_ids") or [])
                # Дедуп + окно: храним не больше 200 id (защита от распухания).
                merged: list[str] = []
                seen_p: set[str] = set()
                for tid in pending_ids + existing_pending:
                    if tid and tid not in seen_p:
                        seen_p.add(tid)
                        merged.append(tid)
                payload["pending_boost_ids"] = merged[:200]
                user_row.onboarding_payload = payload

        log.info(
            "immediate fetch user=%s: upserted %d tweets "
            "(queries=%d handles=%d low_sim_drop=%d, pending=%d)",
            user_id, before, len(queries or []), len(fetch_handles), filtered_low_sim,
            len(pending_ids),
        )
        return before
