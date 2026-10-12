"""Детектор регулярных списаний (подписок) и «забытых» подписок.

Алгоритм (подробно — в README):
1. Берём списания (без переводов/наличных), группируем по нормализованному мерчанту.
2. Внутри группы убираем дубли «тот же день + та же сумма» (запоминаем как аномалию).
3. Кластеризуем суммы (допуск ±12% — курсовые колебания, НДС).
4. Склеиваем кластеры, идущие друг за другом во времени, если общий ряд остаётся
   периодическим — так ловится повышение цены (29 900 → 39 900).
5. Для каждого ряда ищем период (неделя / 2 недели / месяц / квартал / год) по медиане
   интервалов; все интервалы должны укладываться в допуск.
6. Уверенность = регулярность интервалов + стабильность суммы + число повторов +
   «чистота» (доля ряда среди всех операций мерчанта).
7. Флаги: повышение цены, пробный период → платный, дубль, мелкая долгая сумма,
   активность прекратилась / нет связанной активности, подписка уже не активна.
   Из флагов считается forgotten_score; >= 0.5 → «похоже, забытая».
"""

from __future__ import annotations

import datetime as dt
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from .models import Subscription, Transaction
from .normalize import merchant_root


@dataclass(frozen=True)
class PeriodSpec:
    label: str
    nominal: float
    tol: float
    min_occurrences: int


PERIODS: tuple[PeriodSpec, ...] = (
    PeriodSpec("weekly", 7, 2, 4),
    PeriodSpec("biweekly", 14, 3, 3),
    PeriodSpec("monthly", 30.44, 5, 3),
    PeriodSpec("quarterly", 91.3, 12, 2),
    PeriodSpec("yearly", 365.25, 20, 2),
)

DAYS_PER_MONTH = 30.44
DAYS_PER_YEAR = 365.25

# категории, где «пользование» обычно оставляет следы в выписке (бар в зале, визиты)
USAGE_EXPECTED_CATEGORIES = frozenset({"health"})
# у счетов за связь/коммуналку «пользование» в выписке не видно — usage-флаги не ставим
NO_USAGE_SIGNAL_CATEGORIES = frozenset({"utilities"})
EXCLUDED_CATEGORIES = frozenset({"transfers", "cash", "salary"})

FORGOTTEN_WEIGHTS: dict[str, float] = {
    "usage_stopped": 0.6,
    "no_related_activity": 0.5,
    "trial_to_paid": 0.5,
    "small_recurring": 0.3,
    "price_increase": 0.15,
}

FLAG_REASONS_RU: dict[str, str] = {
    "usage_stopped": "связанная активность прекратилась, а списания продолжаются",
    "no_related_activity": "нет ни одной связанной операции (посещений, покупок) за период",
    "trial_to_paid": "пробный период перешёл в платную подписку",
    "small_recurring": "мелкая сумма списывается долго — такие легко забыть",
    "price_increase": "цена выросла",
    "duplicate_charge": "двойное списание в один день",
    "inactive": "списаний больше нет — похоже, подписка уже отменена",
}


@dataclass
class DetectorConfig:
    amount_tolerance: float = 0.12
    min_confidence: float = 0.5
    usage_gap_days: int = 60
    trial_ratio: float = 0.25
    price_increase_threshold: float = 0.07
    forgotten_threshold: float = 0.5
    small_long_min_days: int = 120
    small_monthly_threshold: dict[str, float] = field(
        default_factory=lambda: {"UZS": 30_000, "USD": 3, "EUR": 3, "RUB": 300, "KZT": 1500}
    )


@dataclass
class _Series:
    txns: list[Transaction]
    segments: list[list[Transaction]]

    @property
    def first(self) -> dt.date:
        return self.txns[0].date

    @property
    def last(self) -> dt.date:
        return self.txns[-1].date


def _amount_clusters(txns: list[Transaction], tol: float) -> list[list[Transaction]]:
    clusters: list[list[Transaction]] = []
    anchor = 0.0
    for t in sorted(txns, key=lambda x: x.amount):
        if clusters and t.amount <= anchor * (1 + tol):
            clusters[-1].append(t)
        else:
            clusters.append([t])
            anchor = t.amount
    return [sorted(c, key=lambda x: x.date) for c in clusters]


def _fit_period(dates: list[dt.date]) -> tuple[PeriodSpec, list[int]] | None:
    if len(dates) < 2:
        return None
    intervals = [(b - a).days for a, b in zip(dates, dates[1:], strict=False)]
    med = statistics.median(intervals)
    for spec in PERIODS:
        if abs(med - spec.nominal) > spec.tol:
            continue
        ok = sum(abs(i - spec.nominal) <= spec.tol for i in intervals)
        enough = ok == len(intervals) if len(intervals) <= 4 else ok >= 0.8 * len(intervals)
        if enough and len(dates) >= spec.min_occurrences:
            return spec, intervals
        return None
    return None


def _merge_segments(clusters: list[list[Transaction]]) -> list[_Series]:
    """Склеить последовательные во времени кластеры сумм (смена цены)."""
    series = [_Series(txns=list(c), segments=[list(c)]) for c in clusters]
    merged = True
    while merged:
        merged = False
        series.sort(key=lambda s: s.first)
        for i, a in enumerate(series):
            if len(a.txns) < 2:
                continue
            for j in range(i + 1, len(series)):
                b = series[j]
                if len(b.txns) < 2 or not a.last < b.first:
                    continue
                if _fit_period([t.date for t in a.txns + b.txns]) is None:
                    continue
                series[i] = _Series(txns=a.txns + b.txns, segments=a.segments + b.segments)
                del series[j]
                merged = True
                break
            if merged:
                break
    return series


def _cv(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = statistics.fmean(values)
    return statistics.pstdev(values) / mean if mean else 0.0


def _dedupe_same_day(txns: list[Transaction]) -> tuple[list[Transaction], list[Transaction]]:
    seen: set[tuple[dt.date, float]] = set()
    unique: list[Transaction] = []
    dups: list[Transaction] = []
    for t in sorted(txns, key=lambda x: (x.date, x.amount)):
        key = (t.date, round(t.amount, 2))
        if key in seen:
            dups.append(t)
        else:
            seen.add(key)
            unique.append(t)
    return unique, dups


def _hint(sub: Subscription) -> str:
    if not sub.active:
        return "Похоже, уже отменена — проверьте, что доступ закрыт."
    if "duplicate_charge" in sub.flags and not sub.forgotten:
        return "Двойное списание — запросите возврат у сервиса или банка."
    if sub.forgotten:
        if "usage_stopped" in sub.flags or "no_related_activity" in sub.flags:
            return "Отменить? Пользования не видно, а деньги списываются."
        if "trial_to_paid" in sub.flags:
            return "Отменить? Пробный период незаметно стал платным."
        return "Отменить? Похоже на забытую подписку."
    if "price_increase" in sub.flags:
        return "Цена выросла — сравните тарифы или откажитесь."
    if "small_recurring" in sub.flags:
        return "Мелкая, но регулярная — нужна ли она ещё?"
    return "Оставить"


class SubscriptionDetector:
    def __init__(self, config: DetectorConfig | None = None) -> None:
        self.config = config or DetectorConfig()

    # ------------------------------------------------------------------ public
    def detect(self, txns: list[Transaction], as_of: dt.date | None = None) -> list[Subscription]:
        if not txns:
            return []
        as_of = as_of or max(t.date for t in txns)
        debits = [
            t
            for t in txns
            if t.direction == "debit" and (t.category or "") not in EXCLUDED_CATEGORIES
        ]
        by_merchant: dict[str, list[Transaction]] = defaultdict(list)
        for t in debits:
            by_merchant[t.merchant_normalized].append(t)
        all_debits = [t for t in txns if t.direction == "debit"]

        subs: list[Subscription] = []
        for merchant, group in by_merchant.items():
            if len(group) < 2:
                continue
            subs.extend(self._detect_in_group(merchant, group, all_debits, as_of))
        subs.sort(key=lambda s: (not s.active, not s.forgotten, -s.annual_cost))
        return subs

    # ------------------------------------------------------------------ internals
    def _detect_in_group(
        self,
        merchant: str,
        group: list[Transaction],
        all_debits: list[Transaction],
        as_of: dt.date,
    ) -> list[Subscription]:
        cfg = self.config
        unique, dups = _dedupe_same_day(group)
        clusters = [
            c
            for c in _amount_clusters(unique, cfg.amount_tolerance)
            if len({t.date for t in c}) >= 2
        ]
        result: list[Subscription] = []
        for series in _merge_segments(clusters):
            sub = self._evaluate(merchant, series, group, unique, dups, all_debits, as_of)
            if sub is not None:
                result.append(sub)
        return result

    def _evaluate(
        self,
        merchant: str,
        series: _Series,
        group: list[Transaction],
        unique: list[Transaction],
        dups: list[Transaction],
        all_debits: list[Transaction],
        as_of: dt.date,
    ) -> Subscription | None:
        cfg = self.config
        txns = sorted(series.txns, key=lambda t: t.date)
        dates = [t.date for t in txns]
        fit = _fit_period(dates)
        if fit is None:
            return None
        spec, intervals = fit
        med_interval = statistics.median(intervals)
        amounts = [t.amount for t in txns]

        purity = len(txns) / len(unique)
        seg_cv = statistics.fmean(_cv([t.amount for t in seg]) for seg in series.segments)
        if len(txns) == 2 and (purity < 1 or seg_cv > 0.005):
            return None  # два совпадения — мало: нужна «чистая» группа и почти та же сумма

        interval_dev = statistics.fmean(abs(i - med_interval) for i in intervals)
        interval_score = 1 - min(1.0, interval_dev / spec.tol)
        amount_score = 1 - min(1.0, seg_cv / cfg.amount_tolerance)
        count_score = min(1.0, len(txns) / (spec.min_occurrences + 3))
        confidence = 0.35 * interval_score + 0.25 * amount_score + 0.2 * count_score + 0.2 * purity
        if confidence < cfg.min_confidence:
            return None

        currency = Counter(t.currency for t in txns).most_common(1)[0][0]
        category = Counter(t.category or "other" for t in txns).most_common(1)[0][0]
        old = statistics.median(amounts[:2])
        current = statistics.median(amounts[-2:])
        monthly_cost = current * DAYS_PER_MONTH / spec.nominal
        annual_cost = current * DAYS_PER_YEAR / spec.nominal
        period_days = round(med_interval)
        next_expected = dates[-1] + dt.timedelta(days=period_days)
        active = (as_of - dates[-1]).days <= med_interval + spec.tol
        if not active and any(t.date > dates[-1] for t in unique):
            # мерчант продолжает списывать другие суммы — это не отменённая подписка,
            # а случайное совпадение сумм у плавающего счёта
            return None

        flags: list[str] = []
        old_amount: float | None = None
        if current > old * (1 + cfg.price_increase_threshold):
            flags.append("price_increase")
            old_amount = round(old, 2)

        series_dates = set(dates)
        dup_dates = sorted(
            {
                d.date
                for d in dups
                if d.date in series_dates
                and any(abs(d.amount - t.amount) < 0.01 for t in txns if t.date == d.date)
            }
        )
        if dup_dates:
            flags.append("duplicate_charge")

        trial_amount: float | None = None
        window = med_interval + spec.tol
        for t in group:
            gap = (dates[0] - t.date).days
            if 0 < gap <= window and t.amount <= cfg.trial_ratio * amounts[0]:
                flags.append("trial_to_paid")
                trial_amount = t.amount
                break

        # связанная активность: другие операции того же мерчанта или «семейства»
        series_ids = {id(t) for t in txns} | {id(t) for t in dups}
        root = merchant_root(merchant)
        # (для того же мерчанта — только заметно меньшие суммы: бар в зале, а не тот же счёт)
        related = [
            t
            for t in all_debits
            if id(t) not in series_ids
            and t.amount != trial_amount
            and (
                (t.merchant_normalized == merchant and t.amount < 0.5 * current)
                or (
                    t.merchant_normalized != merchant
                    and merchant_root(t.merchant_normalized) == root
                )
            )
        ]
        if category in NO_USAGE_SIGNAL_CATEGORIES:
            pass
        elif related:
            last_related = max(t.date for t in related)
            if (as_of - last_related).days >= cfg.usage_gap_days and (
                dates[-1] - last_related
            ).days >= spec.nominal:
                flags.append("usage_stopped")
        elif category in USAGE_EXPECTED_CATEGORIES:
            flags.append("no_related_activity")

        small = cfg.small_monthly_threshold.get(currency)
        if (
            small is not None
            and monthly_cost < small
            and (dates[-1] - dates[0]).days >= cfg.small_long_min_days
        ):
            flags.append("small_recurring")

        if not active:
            flags.append("inactive")

        score = sum(FORGOTTEN_WEIGHTS.get(f, 0.0) for f in flags) if active else 0.0
        score = round(min(1.0, score), 2)
        sub = Subscription(
            merchant=merchant,
            category=category,
            amount=round(current, 2),
            currency=currency,
            period_days=period_days,
            period_label=spec.label,
            occurrences=len(txns),
            first_date=dates[0],
            last_date=dates[-1],
            next_expected=next_expected,
            monthly_cost=round(monthly_cost, 2),
            annual_cost=round(annual_cost, 2),
            confidence=round(confidence, 2),
            active=active,
            amount_history=[round(a, 2) for a in amounts],
            flags=flags,
            forgotten=active and score >= cfg.forgotten_threshold,
            forgotten_score=score,
            reasons=[self._reason(f, old_amount, current, trial_amount) for f in flags],
            old_amount=old_amount,
            trial_amount=trial_amount,
            duplicate_dates=dup_dates,
        )
        sub.hint = _hint(sub)
        return sub

    @staticmethod
    def _reason(flag: str, old: float | None, new: float, trial: float | None) -> str:
        text = FLAG_REASONS_RU.get(flag, flag)
        if flag == "price_increase" and old:
            text += f": {old:,.0f} → {new:,.0f} (+{(new / old - 1) * 100:.0f}%)".replace(",", " ")
        if flag == "trial_to_paid" and trial is not None:
            text += f" (пробное списание {trial:,.0f})".replace(",", " ")
        return text


def detect_subscriptions(
    txns: list[Transaction],
    config: DetectorConfig | None = None,
    as_of: dt.date | None = None,
) -> list[Subscription]:
    return SubscriptionDetector(config).detect(txns, as_of=as_of)


def burn(subs: list[Subscription]) -> tuple[float, float]:
    """(ежемесячная, годовая) нагрузка активных подписок."""
    active = [s for s in subs if s.active]
    return (
        round(sum(s.monthly_cost for s in active), 2),
        round(sum(s.annual_cost for s in active), 2),
    )
