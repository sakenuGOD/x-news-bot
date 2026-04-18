"""Inline клавиатуры. Callback data — короткая, чтобы влезала в 64 байта Telegram."""

from __future__ import annotations

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

# callback_data формат:
#   "fb:like:<tweet_id>" / "fb:dis:<tweet_id>"
#   "tr:ru:<tweet_id>" — перевести на русский
#   "tr:en:<tweet_id>" — вернуться к английскому оригиналу
#   "ctl:more" / "ctl:pause" / "ctl:resume"
#   "intv:<hours>"


def feedback_kb(
    tweet_id: str,
    liked: bool | None = None,
    translated: bool = False,
    quote_author: str | None = None,
) -> InlineKeyboardMarkup:
    """Клавиатура под твитом: 👍 / 👎 / язык / (↪ Цитата если есть) / 💬 Комменты.

    liked=True  → 👍 подсвечен
    liked=False → 👎 подсвечен
    liked=None  → ничего не подсвечено (ещё не голосовал)

    translated=True означает что сейчас показан русский перевод; кнопка предложит
    вернуться к оригиналу.

    quote_author — если задан, добавляется кнопка «↪ Цитата: @<author>». При
    клике отправляется второе сообщение с текстом/медиа цитаты. Не шлём цитату
    автоматически как было раньше: юзеру раздражает когда каждый пост
    превращается в пару bubble'ов «автор / цитата» без его запроса.
    """
    like_txt = "✅ 👍" if liked is True else "👍"
    dis_txt = "✅ 👎" if liked is False else "👎"
    if translated:
        lang_txt = "🇬🇧 Оригинал"
        lang_cb = f"tr:en:{tweet_id}"
    else:
        lang_txt = "🇷🇺 Перевод"
        lang_cb = f"tr:ru:{tweet_id}"

    rows: list[list[InlineKeyboardButton]] = [
        [
            InlineKeyboardButton(text=like_txt, callback_data=f"fb:like:{tweet_id}"),
            InlineKeyboardButton(text=dis_txt, callback_data=f"fb:dis:{tweet_id}"),
            InlineKeyboardButton(text=lang_txt, callback_data=lang_cb),
        ],
    ]
    if quote_author:
        # Обрезаем до 20 символов — callback_data limit + красиво в UI.
        qa = quote_author if len(quote_author) <= 20 else quote_author[:19] + "…"
        rows.append([InlineKeyboardButton(
            text=f"↪ Цитата: @{qa}",
            callback_data=f"qt:{tweet_id}",
        )])
    rows.append([
        InlineKeyboardButton(text="💬 Комменты", callback_data=f"cm:{tweet_id}"),
    ])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def batch_controls_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="⚙️ Хочу больше…", callback_data="ctl:more"),
                InlineKeyboardButton(text="⏸ Пауза", callback_data="ctl:pause"),
            ]
        ]
    )


def resume_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[[InlineKeyboardButton(text="▶️ Продолжить", callback_data="ctl:resume")]]
    )


def setup_welcome_kb() -> InlineKeyboardMarkup:
    """Главное меню при первом /start — три явных варианта."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="⚙️ Подключить мой X-аккаунт", callback_data="setup:connect")],
            [InlineKeyboardButton(text="🚀 Попробовать без настройки (демо)", callback_data="setup:demo")],
            [InlineKeyboardButton(text="📖 Как это работает", callback_data="setup:info")],
        ]
    )


def setup_connect_kb() -> InlineKeyboardMarkup:
    """Меню после «Подключить X» — ссылка на инструкцию + ввод токена."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="🌐 Открыть инструкцию (twikit)", url="https://github.com/d60/twikit#readme")],
            [InlineKeyboardButton(text="📥 Ввести auth_token", callback_data="setup:enter_token")],
            [InlineKeyboardButton(text="⬅ Назад", callback_data="setup:back")],
        ]
    )


def setup_ct0_kb() -> InlineKeyboardMarkup:
    """После auth_token спрашиваем ct0 (опционально)."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="📥 Добавить ct0", callback_data="setup:enter_ct0")],
            [InlineKeyboardButton(text="⏭ Пропустить", callback_data="setup:skip_ct0")],
            [InlineKeyboardButton(text="⬅ Назад", callback_data="setup:back")],
        ]
    )


def setup_verified_kb() -> InlineKeyboardMarkup:
    """После успешной верификации X — переход к анализу подписок."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="▶️ Продолжить — указать свой X", callback_data="setup:start_username")],
        ]
    )


def setup_retry_kb() -> InlineKeyboardMarkup:
    """Кнопки после неудачной проверки X — повтор или демо."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="🔁 Ввести токен заново", callback_data="setup:enter_token")],
            [InlineKeyboardButton(text="🚀 Пропустить → демо", callback_data="setup:demo")],
        ]
    )


def main_menu_kb(paused: bool = False) -> InlineKeyboardMarkup:
    """Главное меню.

    📊 Что обсуждают — For You лента X, сгруппированная по темам
                      (аналог X Topics / Stories — система сама решает
                      что для тебя актуально).
    📰 Моя лента     — Following (хронологически, все посты подписок).
    """
    rows: list[list[InlineKeyboardButton]] = [
        [InlineKeyboardButton(text="📊 Что обсуждают", callback_data="rep:new")],
        [InlineKeyboardButton(text="📰 Моя лента", callback_data="rep:feed")],
        [InlineKeyboardButton(text="💬 Хочу больше / меньше", callback_data="ctl:more")],
    ]
    if paused:
        rows.append([InlineKeyboardButton(text="▶️ Продолжить", callback_data="ctl:resume")])
    else:
        rows.append([InlineKeyboardButton(text="⏸ Пауза", callback_data="ctl:pause")])
    rows.append([InlineKeyboardButton(text="⚙ Интервал", callback_data="menu:interval")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def report_overview_kb(clusters_meta: list[tuple[int, str, str, int]]) -> InlineKeyboardMarkup:
    """Плоский обзор отчёта (без супер-топиков): по кнопке на каждую тему.

    clusters_meta: [(cluster_id, emoji, name, count), ...]
    """
    rows: list[list[InlineKeyboardButton]] = []
    for cid, emoji, name, count in clusters_meta:
        short = name if len(name) <= 28 else name[:27] + "…"
        rows.append([InlineKeyboardButton(
            text=f"{emoji} {short} · {count}",
            callback_data=f"rep:topic:{cid}",
        )])
    rows.append([
        InlineKeyboardButton(text="🔁 Обновить отчёт", callback_data="rep:new"),
        InlineKeyboardButton(text="⬅ В меню", callback_data="menu:main"),
    ])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def super_topics_kb(supers: list[tuple[int, str, str, int]]) -> InlineKeyboardMarkup:
    """Супер-категории: [(super_idx, emoji, name, total_posts), ...].

    Клик по супер-категории раскрывает её саб-темы.
    """
    rows: list[list[InlineKeyboardButton]] = []
    for sidx, emoji, name, total in supers:
        short = name if len(name) <= 28 else name[:27] + "…"
        rows.append([InlineKeyboardButton(
            text=f"{emoji} {short} · {total}",
            callback_data=f"rep:super:{sidx}",
        )])
    rows.append([
        InlineKeyboardButton(text="📋 Все темы списком", callback_data="rep:flat"),
        InlineKeyboardButton(text="🔁 Обновить", callback_data="rep:new"),
    ])
    rows.append([InlineKeyboardButton(text="⬅ В меню", callback_data="menu:main")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def super_topic_details_kb(
    sub_clusters_meta: list[tuple[int, str, str, int]],
) -> InlineKeyboardMarkup:
    """После клика на супер-категорию — список её под-тем и кнопка назад."""
    rows: list[list[InlineKeyboardButton]] = []
    for cid, emoji, name, count in sub_clusters_meta:
        short = name if len(name) <= 28 else name[:27] + "…"
        rows.append([InlineKeyboardButton(
            text=f"{emoji} {short} · {count}",
            callback_data=f"rep:topic:{cid}",
        )])
    rows.append([
        InlineKeyboardButton(text="⬅ К категориям", callback_data="rep:supers"),
        InlineKeyboardButton(text="🏠 Меню", callback_data="menu:main"),
    ])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def topic_view_kb(cluster_id: int, has_more: bool = False) -> InlineKeyboardMarkup:
    """Старая клавиатура «в конце темы» (оставил чтобы не ломать существующие вызовы)."""
    rows: list[list[InlineKeyboardButton]] = []
    if has_more:
        rows.append([InlineKeyboardButton(
            text="📜 Показать все посты темы",
            callback_data=f"rep:expand:{cluster_id}",
        )])
    rows.append([
        InlineKeyboardButton(text="⬅ К отчёту", callback_data="rep:back"),
        InlineKeyboardButton(text="🏠 В меню", callback_data="menu:main"),
    ])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def topic_paginator_kb(
    cluster_id: int,
    pos: int,
    total: int,
    tweet_id: str,
    liked: bool | None = None,
    translated: bool = False,
    quote_author: str | None = None,
) -> InlineKeyboardMarkup:
    """Клавиатура пагинатора темы.

    Layout:
      Row 1: [👍] [👎] [🇷🇺 Перевод / 🇬🇧 Оригинал]    ← реакции
      Row 2: [↪ Цитата: @user]                        ← если есть цитата
      Row 3: [💬 Комменты]
      Row 4: [⬅ Пред.] [N/total] [След. ➡]
      Row 5: [⬅ К темам] [🏠 В меню]
    """
    like_txt = "✅ 👍" if liked is True else "👍"
    dis_txt = "✅ 👎" if liked is False else "👎"
    lang_txt = "🇬🇧 Оригинал" if translated else "🇷🇺 Перевод"
    lang_cb = f"tr:en:{tweet_id}" if translated else f"tr:ru:{tweet_id}"

    fb_row = [
        InlineKeyboardButton(text=like_txt, callback_data=f"fb:like:{tweet_id}"),
        InlineKeyboardButton(text=dis_txt, callback_data=f"fb:dis:{tweet_id}"),
        InlineKeyboardButton(text=lang_txt, callback_data=lang_cb),
    ]

    nav_row: list[InlineKeyboardButton] = []
    if pos > 0:
        nav_row.append(InlineKeyboardButton(
            text="⬅ Пред.", callback_data=f"tpc:{cluster_id}:{pos-1}",
        ))
    else:
        nav_row.append(InlineKeyboardButton(text="·", callback_data="noop"))
    nav_row.append(InlineKeyboardButton(
        text=f"{pos+1} / {total}", callback_data="noop",
    ))
    if pos < total - 1:
        nav_row.append(InlineKeyboardButton(
            text="След. ➡", callback_data=f"tpc:{cluster_id}:{pos+1}",
        ))
    else:
        nav_row.append(InlineKeyboardButton(text="·", callback_data="noop"))

    rows: list[list[InlineKeyboardButton]] = [fb_row]
    if quote_author:
        qa = quote_author if len(quote_author) <= 20 else quote_author[:19] + "…"
        rows.append([InlineKeyboardButton(
            text=f"↪ Цитата: @{qa}",
            callback_data=f"qt:{tweet_id}",
        )])
    rows.append([InlineKeyboardButton(text="💬 Комменты", callback_data=f"cm:{tweet_id}")])
    rows.append(nav_row)
    rows.append([
        InlineKeyboardButton(text="⬅ К темам", callback_data="rep:back"),
        InlineKeyboardButton(text="🏠 В меню", callback_data="menu:main"),
    ])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def interval_kb(with_back: bool = True) -> InlineKeyboardMarkup:
    rows = [
        [
            InlineKeyboardButton(text="1 час", callback_data="intv:1"),
            InlineKeyboardButton(text="3 часа", callback_data="intv:3"),
        ],
        [
            InlineKeyboardButton(text="6 часов", callback_data="intv:6"),
            InlineKeyboardButton(text="12 часов", callback_data="intv:12"),
        ],
    ]
    if with_back:
        rows.append([InlineKeyboardButton(text="⬅ В меню", callback_data="menu:main")])
    return InlineKeyboardMarkup(inline_keyboard=rows)
