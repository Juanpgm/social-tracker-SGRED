"""Synthetic corpora for the calibration tests (C1, C3). Everything here is generated, seeded and
fictitious: no real person, account or report is reproduced.

`templated_bulletins` imitates a feed that publishes the same sentence with different numbers
all day long (an automatic seismic bulletin). `mixed_corpus` imitates a day of the monitored
sources: the SHARE of each kind is an assumption (written down in `MIX` and in the README), not a
measurement, and the escalation rate it produces is only as good as that assumption.
"""

import random
from collections.abc import Iterator
from dataclasses import dataclass

PLACES = (
    "Cali", "Yumbo", "Jamundí", "Palmira", "Buenaventura", "Popayán", "Tuluá", "Buga", "Cartago",
    "Pasto", "Armenia", "Pereira", "Manizales", "Ibagué", "Neiva", "Florencia", "Quibdó",
    "Quilichao", "Dagua", "Candelaria", "Florida", "Pradera", "Cerrito", "Restrepo", "Calima",
    "Zarzal", "Roldanillo", "Sevilla", "Caicedonia",
)  # fmt: skip
LINK = "https://official-domain.example/sismos/"


def templated_bulletins(n: int, *, days_span: int, seed: int) -> list[tuple[str, int]]:
    """`n` bulletins `(text, hours since the first)` spread over `days_span` days, all built on
    one sentence: only the magnitude, the place, the time and the depth change."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        text = (
            f"Sismo de magnitud {rng.randint(20, 59) / 10:.1f} con epicentro en "
            f"{rng.choice(PLACES)} a las {rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}, "
            f"profundidad {rng.randint(2, 140)} km"
        )
        out.append((text, rng.randint(0, days_span * 24 - 1)))
    return sorted(out, key=lambda pair: pair[1])


@dataclass(frozen=True)
class Kind:
    name: str
    share: float  # of the whole corpus
    escalated: bool  # whether the cascade is expected to send it to the LLM
    in_scope: bool = True


# The assumed mix. Shares add up to 1. The kinds the rules should escalate add up to 14 %.
MIX = (
    Kind("official_report_linked", 0.04, escalated=False),
    Kind("news_report", 0.08, escalated=False),
    Kind("neutral_chatter", 0.22, escalated=False),
    Kind("secretariat_complaint", 0.07, escalated=False),
    Kind("other_disaster_news", 0.07, escalated=False),
    Kind("off_topic", 0.33, escalated=False, in_scope=False),
    Kind("crime", 0.04, escalated=False, in_scope=False),
    Kind("english", 0.01, escalated=False, in_scope=False),
    Kind("prediction_rumor", 0.04, escalated=True),
    Kind("chain_message", 0.03, escalated=True),
    Kind("false_attribution", 0.02, escalated=True),
    Kind("recycled_caption", 0.02, escalated=True),
    Kind("attributed_report_unlinked", 0.03, escalated=True),
)


def _pick(rng: random.Random, options: tuple[str, ...]) -> str:
    return options[rng.randrange(len(options))]


def _quake(rng: random.Random) -> str:
    return f"{rng.randint(20, 59) / 10:.1f}"


def _text(kind: str, rng: random.Random, n: int) -> str:
    place = _pick(rng, PLACES)
    hour = f"{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}"
    if kind == "official_report_linked":
        return (
            f"El Servicio Geológico Colombiano informó un sismo de magnitud {_quake(rng)} con "
            f"epicentro en {place} a las {hour}. Más información en {LINK}{n}"
        )
    if kind == "news_report":
        return _pick(
            rng,
            (
                f"Un sismo de magnitud {_quake(rng)} se registró hoy en {place} sin reportes "
                f"de daños, según los primeros datos",
                f"Un temblor de {_quake(rng)} grados se sintió esta mañana en {place} y la gente "
                f"salió a la calle sin que se reportaran heridos",
            ),
        )
    if kind == "neutral_chatter":
        return _pick(
            rng,
            (
                f"Ayer sentimos el temblor en {place} y todos los vecinos salieron a la calle "
                f"a conversar un rato",
                f"Qué susto el sismo de anoche, en {place} se movió todo el edificio y los "
                f"perros no paraban de ladrar",
                f"Alguien sabe si el temblor de hoy en {place} alcanzó a dañar las casas del "
                f"barrio donde vive mi mamá",
            ),
        )
    if kind == "secretariat_complaint":
        return _pick(
            rng,
            (
                f"Otra vez la Secretaría de Gestión del Riesgo sin limpiar los caños de "
                f"{place}, cuando llueve se inunda todo",
                f"El DAGRD nunca contesta las llamadas de los vecinos de {place} cuando hay "
                f"una emergencia en el barrio",
                f"Me parece que la Secretaría de Gestión del Riesgo debería explicar mejor "
                f"cómo se prepara {place} para un terremoto",
            ),
        )
    if kind == "other_disaster_news":
        return _pick(
            rng,
            (
                f"Las lluvias de esta semana provocaron un deslizamiento en la vía hacia "
                f"{place} y el tránsito sigue lento",
                f"Una creciente súbita del río dejó varias familias en {place} esperando que "
                f"baje el agua para volver a sus casas",
            ),
        )
    if kind == "off_topic":
        return _pick(
            rng,
            (
                f"El equipo de {place} ganó el partido de anoche con un gol en el último minuto",
                f"Abrieron un restaurante nuevo en {place} con comida típica y precios "
                f"razonables para las familias",
                f"El concierto del sábado en {place} fue una locura, la gente cantó todas las "
                f"canciones de principio a fin",
            ),
        )
    if kind == "crime":
        return f"Hubo un robo a mano armada en un local de {place} y la policía capturó a dos"
    if kind == "english":
        return f"The mayor said that the new bridge in {place} will be open for the people soon"
    if kind == "prediction_rumor":
        return _pick(
            rng,
            (
                f"Mañana habrá un temblor fuerte en {place}, compartan este mensaje",
                f"Se viene un terremoto en {place} este fin de semana, avisen a todos",
                f"Un científico predijo un sismo grande para el sábado en {place}",
            ),
        )
    if kind == "chain_message":
        return (
            f"URGENTE!!! Dicen que va a temblar en {place}, fuentes confiables lo confirman. "
            f"Compartan y reenvíen a todos sus contactos antes de que lo borren"
        )
    if kind == "false_attribution":
        return _pick(
            rng,
            (
                f"La Secretaría de Gestión del Riesgo ordena evacuar {place} de inmediato por "
                f"un sismo que viene, evacuen ya",
                f"Según la UNGRD el sábado habrá un terremoto en {place}, salgan de sus casas",
            ),
        )
    if kind == "recycled_caption":
        return (
            f"Video de 2016 del terremoto, ahora en {place}: miren cómo se cae el edificio, "
            f"esto es de hoy y nadie lo dice"
        )
    if kind == "attributed_report_unlinked":
        return (
            f"La Defensa Civil informó que el sismo de {hour} dejó daños menores en {place} y "
            f"que los equipos recorren las calles"
        )
    raise ValueError(kind)


def mixed_corpus(n: int, *, seed: int) -> Iterator[tuple[str, Kind]]:
    """`n` items `(text, kind)` drawn from `MIX`; every text is made unique by its place, time
    and magnitude or, failing that, by a trailing index in a form that is not a number."""
    rng = random.Random(seed)
    kinds = [kind for kind in MIX for _ in range(round(kind.share * 1000))]
    for index in range(n):
        kind = kinds[rng.randrange(len(kinds))]
        text = _text(kind.name, rng, index)
        # distinct items for the store; the suffix is letters only, so no number is added
        suffix = "".join(chr(97 + int(digit)) for digit in str(index))
        yield f"{text} ref{suffix}" if kind.name != "official_report_linked" else text, kind
