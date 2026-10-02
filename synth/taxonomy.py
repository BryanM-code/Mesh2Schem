"""Architectural-category classifier for 3D-asset text metadata (caption, name, tags).

A render caption names its subject first: "A two-story brick house with a garage" is a
house, not a garage, and "Off-road vehicle with an enclosed cabin" is a vehicle, not a
cabin. So classification reads the HEAD PHRASE -- the caption up to its first "with",
"featuring", full stop... -- before anything else, and an object noun in the head (desk,
car, cup) rejects the object outright. Name and tags are only consulted when there is no
caption, and then only with architecture evidence.
"""
import re

CATEGORIES = [
    ("skyscraper", r"skyscrapers?|high[- ]?rises?|highrise|super[- ]?tall|tower blocks?|"
                   r"(?:glass|office|residential|apartment|corporate|modern|commercial) (?:tower|high[- ]rise)s?"),
    ("apartment", r"apartments?(?: (?:building|block|complex|house|tower))?|block of flats|"
                  r"condominiums?|tenements?|housing (?:block|estate)|"
                  r"panel (?:building|house)s?|khrushchyovka|plattenbau|dormitor(?:y|ies)|"
                  r"residential (?:building|block|tower)s?|multi[- ]?family|"
                  r"(?:multi|many)[- ]?(?:story|storey|level) residential"),
    ("townhouse", r"town ?houses?|terraced? houses?|row ?houses?|brownstones?|semi[- ]detached|duplex(?:es)?"),
    ("office", r"office (?:building|block|tower|complex)s?|corporate (?:building|headquarters)s?|"
               r"headquarters|business (?:center|centre|building)s?|bank buildings?"),
    ("commercial", r"shops?|stores?|supermarkets?|shopping (?:mall|center|centre)s?|malls?|restaurants?|"
                   r"caf[eé]s?|diners?|pubs?|taverns?|inns?|hotels?|motels?|(?:gas|petrol|filling) stations?|"
                   r"market (?:hall|building)s?|pharmac(?:y|ies)|drugstores?|bakery|bakeries|cinemas?|"
                   r"storefronts?|boutiques?|dealerships?|fast[- ]food restaurants?|casinos?"),
    ("institutional", r"schools?|schoolhouses?|universit(?:y|ies)|college buildings?|hospitals?|clinics?|"
                      r"librar(?:y|ies)|museums?|town halls?|city halls?|courthouses?|court houses?|"
                      r"police stations?|fire ?stations?|firehouses?|post offices?|government buildings?|"
                      r"parliament|capitol|(?:train|railway|railroad|bus|metro|subway) stations?|"
                      r"station buildings?|terminal buildings?|embass(?:y|ies)|opera houses?|theat(?:er|re)s?|"
                      r"observator(?:y|ies)|stadiums?|arenas?|gymnasiums?|exhibition halls?"),
    # Open garden/park structures: not "institutional", and not enclosed buildings.
    ("pavilion", r"pavilions?|pergolas?|bandstands?|rotundas?"),
    ("religious", r"churche?s?|cathedrals?|chapels?|mosques?|temples?|synagogues?|shrines?|monaster(?:y|ies)|"
                  r"abbe(?:y|ys)|basilicas?|pagodas?|stupas?|minarets?|kirks?|convents?"),
    ("fortification", r"castles?|fortress(?:es)?|citadels?|palaces?|ch[aâ]teaux?|bastions?|"
                      r"strongholds?|fortifications?|kremlin|forts?"),
    ("tower", r"lighthouses?|light houses?|clock ?towers?|bell ?towers?|belfr(?:y|ies)|campaniles?|"
              r"water ?towers?|watch ?towers?|observation towers?|(?:tv|radio|television) towers?|towers?"),
    ("industrial", r"factor(?:y|ies)|warehouses?|barns?|silos?|windmills?|watermills?|mills?|hangars?|"
                   r"garages?|sheds?|workshops?|power (?:plant|station)s?|depots?|stables?|greenhouses?|"
                   r"granar(?:y|ies)|boathouses?|sawmills?|foundr(?:y|ies)|refiner(?:y|ies)"),
    ("house", r"houses?|homes?|cottages?|bungalows?|villas?|cabins?|chalets?|farm ?houses?|mansions?|"
              r"residences?|lodges?|huts?|shacks?|ranch(?:es)?|manors?|dachas?|izbas?|dwellings?|"
              r"residential propert(?:y|ies)|family residences?"),
]
_CAT_RE = [(c, re.compile(r"\b(?:" + p + r")\b", re.I)) for c, p in CATEGORIES]

# Where a caption's subject ends and its description starts.
HEAD_SPLIT = re.compile(r"\s(?:with|featuring|including|showcasing|showing|having|topped|surrounded|"
                        r"consisting|comprising|containing|situated|set|sitting|standing|atop|beside|"
                        r"next|flanked|adorned|equipped|made|built|that|which|where|whose|on|in|under|"
                        r"against|inside|within|overlooking|characterized|displaying|decorated)\b"
                        r"|[.;:]|,\s(?:a|an|the)\s", re.I)

# Object nouns that, in the head phrase, mean the subject is not a building.
OBJECT_HEAD = re.compile(
    r"\b(?:desks?|chairs?|tables?|beds?|stretchers?|cabinets?|shelf|shelves|bookcases?|benches?|sofas?|"
    r"couch(?:es)?|stools?|lamps?|lanterns?|carts?|trolley|cars?|vehicles?|suv|sedans?|trucks?|vans?|buses|"
    r"bus|motorcycles?|motorbikes?|bikes?|bicycles?|wagons?|jeeps?|tractors?|boats?|ships?|yachts?|"
    r"airplanes?|aircraft|helicopters?|trains?|locomotives?|trams?|rockets?|spaceships?|robots?|"
    r"characters?|figures?|figurines?|statues?|busts?|sculptures?|monuments?|memorials?|obelisks?|"
    r"fountains?|bridges?|gates?|gazebos?|arches?|crosses|cross|tombstones?|graves?|"
    r"cups?|mugs?|bottles?|jars?|vases?|pots?|pottery|bowls?|plates?|boxes|box|crates?|barrels?|"
    r"containers?|signs?|kiosks?|booths?|vending machines?|machines?|computers?|terminals?|"
    r"helmets?|weapons?|swords?|guns?|rifles?|skulls?|clocks?|watches|phones?|books?|coins?|"
    r"trees?|plants?|flowers?|rocks?|stones?|tents?|yurts?|teepees?|igloos?|mailbox(?:es)?|"
    r"trailers?|caravans?|ferris wheel|carousel|playground|swing|slide|pool table|piano|organ|"
    r"altars?|fonts?|pulpits?|thrones?|chess|board game|toys?|dolls?|puppets?|masks?|"
    r"waitress(?:es)?|waiters?|persons?|people|man|men|woman|women|girls?|boys?|workers?|chefs?|"
    r"strainers?|filters?|valves?|pumps?|reactors?|"
    r"cranes?|pylons?|poles?|antennas?|turbines?|engines?|generators?|tanks?|"
    r"platforms?|stages?|docks?|piers?|fences?|walls?|doors?|windows?|facades?|façades?|"
    r"roofs?|columns?|pillars?|staircases?|stairs|chimneys?|domes?|spires?|entrances?|doorways?|portals?)\b",
    re.I)

STRONG_CUE = re.compile(r"\b(?:buildings?|roofs?|rooftop|roofline|facades?|façades?|stor(?:y|ey|ies|eys)|"
                        r"floors|architectur\w*|gables?|gabled|chimneys?|dormers?|eaves|masonry|"
                        r"balcon(?:y|ies)|porch|entrance|foundation|spires?|steeples?|belfr\w+|"
                        r"crenellat\w*|turrets?|battlements?)\b", re.I)

_MULTIPLIER = (r"(?:two|three|four|five|six|several|multiple|many|numerous|various|pair of|group of|"
               r"set of|cluster of|row of|collection of|series of|complex of|array of|variety of)")
# Several whole buildings anywhere, or several anything at the very start: a scene. A
# cathedral "with two towers" is still one building.
MULTI = re.compile(r"\b" + _MULTIPLIER + r"\b[^.;]{0,50}?\b(?:buildings|houses|homes|cottages|"
                   r"skyscrapers|high-rises|huts|cabins|villas|bungalows)\b", re.I)
MULTI_HEAD = re.compile(r"^(?:a |an |the )?" + _MULTIPLIER + r"\b[^.;]{0,50}?\b(?:towers|structures|"
                        r"churches|castles|buildings|houses|homes|blocks)\b", re.I)
NOT_FINISHED = re.compile(r"under construction|skeletal|structural (?:frame|model)|wire ?frame|"
                          r"framing only|scaffold|wooden frame model|foundation only|massing model|"
                          r"partially constructed|schematic model|exposed structural|"
                          r"white model|clay model|block ?out|blockout", re.I)

HOUSE_SUB = [
    ("house:traditional", r"cottage|cabin|chalet|farm ?house|\bhut\b|shack|\blog\b|thatch|rustic|rural|"
                          r"tudor|timber[- ]fram|half[- ]timber|medieval|victorian|colonial|georgian|"
                          r"historic|\bold\b|traditional|stone house|dacha|izba|manor|lodge"),
    ("house:villa", r"villa|mansion|estate|luxur"),
    ("house:modern", r"modern|contemporary|minimalist|flat roof|glass|concrete"),
    ("house:suburban", r"suburban|family house|family home|two[- ]stor(?:y|ey) house|garage|porch|driveway|"
                       r"gable|siding|bungalow|ranch"),
]
_HOUSE_SUB_RE = [(c, re.compile(p, re.I)) for c, p in HOUSE_SUB]

# Checked against the head phrase + name (what the object IS), not the whole caption.
HEAD_EXCLUDE = {
    "blocky_source": r"minecraft|voxel|pixel ?art|pixelated|\blego\b|8[- ]bit|made (?:up )?of (?:small )?cubes|"
                     r"block[- ]based|cube[- ]based",
    "toy_or_miniature": r"dollhouse|doll house|birdhouse|bird house|doghouse|dog house|\btoys?\b|"
                        r"figurine|toy blocks|building blocks|wooden blocks|unit ?blocks|playset|play ?set|"
                        r"gingerbread|\bcake\b|music box|papercraft|paper model|cardboard|keychain|"
                        r"snow ?globe|ornament|souvenir|ceramic|porcelain|\bminiature\b",
    "interior": r"\binterior\b|\brooms?\b|living room|kitchen|bedroom|bathroom|lobby|hallway|corridor|"
                r"\boffice space\b|inside of",
    "scene": r"\bcity\b|cityscape|\bvillage\b|\bstreets?\b|neighbou?rhood|district|diorama|\bscene\b|"
             r"landscape|\bisland\b|\bmap\b|collection|\bset of\b|\bpack\b|\bkit\b|modular|"
             r"row of|complex of|compound|settlement|courtyard|plaza|\bsquare\b",
    "part": r"^(?:a |an |the )?(?:single |wooden |stone |metal |white |old |large |small |classical |ornate )?"
            r"(?:door|window|wall|facade|façade|roof|column|staircase|stairs|fence|gate|balcony|"
            r"chimney|arch|pillar|porch|dome|spire|entrance|doorway|portal)\b",
}
_HEAD_EXCLUDE_RE = {k: re.compile(p, re.I) for k, p in HEAD_EXCLUDE.items()}
# Toy block sets: "a wooden castle made of rectangular prisms, cylinders...".
_TOY_BUILD = re.compile(r"wooden blocks?|block castle|made of (?:light[- ]colou?red |unfinished )?(?:wood(?:en)? )?"
                        r"(?:rectangular )?(?:blocks|prisms|cubes)|made of rectangular prisms|"
                        r"geometric (?:wooden )?blocks|building blocks|toy", re.I)
# Checked anywhere: these describe how the object was made or drawn.
ANY_EXCLUDE = {
    "cutaway": r"cutaway|cross[- ]section|floor ?plan|sectional view|interior view|inside view",
    "ruin_or_wreck": r"\bruins?\b|ruined|destroyed|abandoned|wreck|collapsed|damaged|burnt|bombed|dilapidated",
}
_ANY_EXCLUDE_RE = {k: re.compile(p, re.I) for k, p in ANY_EXCLUDE.items()}

# Furnished interiors (a diner's booths, an apartment's floor layout): several furniture or
# room words in one caption means the model is mostly inside, often without a roof.
_INTERIOR_WORD = re.compile(r"\b(?:sofas?|couch(?:es)?|beds?|bedrooms?|kitchens?|kitchenette|bathrooms?|"
                            r"living (?:room|area)|dining (?:room|area|table)|furniture|furnished|booths?|"
                            r"counters?|shelving|shelves|racks?|cabinetry|workstations?|cubicles?|"
                            r"armchairs?|coffee table|tv|television|mannequins?|stools?|desks?|rooms)\b", re.I)

STYLE = {
    "stylized": r"cartoon|cartoonish|\btoon\b|stylized|stylised|chibi|fantasy|fairy|whimsical|magical|"
                r"gnome|hobbit|mushroom house|candy|sci[- ]fi|futuristic|alien|cyberpunk|steampunk|"
                r"post[- ]apocalyptic|haunted|spooky|halloween|christmas|anime",
    "low_poly": r"low[- ]?poly|lowpoly|flat[- ]shaded|simple geometric|simplified",
    "scan": r"photogrammetr|\bscan\b|\bscanned\b|3d scan|reality ?capture|\bdrone\b|lidar|agisoft|"
            r"metashape|polycam|\bluma\b|point cloud",
}
_STYLE_RE = {k: re.compile(p, re.I) for k, p in STYLE.items()}

# Rips of commercial game assets: the uploader's CC licence is not theirs to give.
GAME_RIP = re.compile(
    r"fortnite|\bgta\b|grand theft auto|roblox|\bsims\b|zelda|pok[eé]mon|animal crossing|stardew|"
    r"\bhalo\b|call of duty|counter[- ]strike|cs ?:? ?go|de_dust|genshin|fallout|skyrim|witcher|"
    r"world of warcraft|\bwow\b|league of legends|overwatch|ripped|\brip\b|fan ?art|from the game|"
    r"resident evil|silent hill|half[- ]life|assassin'?s creed|red dead|cyberpunk 2077|"
    r"final fantasy|mario|sonic|nintendo|playstation|xbox|valorant|apex legends|pubg|"
    r"team fortress|tf2|garry'?s mod|gmod|source engine|unreal tournament|heroes 3|"
    r"harry potter|star wars|disney|pixar|marvel|simpsons|transformers",
    re.I)

_NUM = {"one": 1, "single": 1, "two": 2, "double": 2, "three": 3, "four": 4, "five": 5, "six": 6,
        "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15,
        "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50}
_STOREY_RE = re.compile(r"\b(\d{1,3}|" + "|".join(_NUM) + r")[- ](?:story|storey|stories|storeys|floor|level)s?\b",
                        re.I)
_MULTI_STOREY_RE = re.compile(r"\bmulti[- ]?(?:story|storey|level|floor)\b|\bmany floors\b|\bnumerous floors\b",
                              re.I)
_GENERIC_RE = re.compile(r"\bbuildings?\b|\barchitectural model\b", re.I)


# Coarse groups for reporting a dataset's balance: every category belongs to exactly one.
# Institutional buildings are split by subcategory so civic buildings (town halls, courts,
# stations...) are counted on their own.
GROUPS = {
    "house": "residential", "townhouse": "residential", "apartment": "residential",
    "midrise_generic": "mid_rise", "skyscraper": "high_rise",
    "office": "commercial", "commercial": "commercial",
    "institutional": "institutional", "religious": "religious", "tower": "tower",
    "fortification": "fortification", "industrial": "industrial", "pavilion": "other_structure",
    "building_generic": "unspecified",
}

INSTITUTIONAL_SUB = [
    ("civic", r"town halls?|city halls?|courthouses?|court houses?|police stations?|fire ?stations?|"
              r"firehouses?|post offices?|government|parliament|capitol|embass(?:y|ies)|"
              r"(?:train|railway|railroad|bus|metro|subway) stations?|station buildings?|terminal buildings?"),
    ("education", r"schools?|schoolhouses?|universit(?:y|ies)|college"),
    ("health", r"hospitals?|clinics?"),
    ("culture", r"librar(?:y|ies)|museums?|opera|theat(?:er|re)s?|observator(?:y|ies)|stadiums?|arenas?|"
                r"gymnasiums?|exhibition"),
]
_INSTITUTIONAL_SUB_RE = [(c, re.compile(r"\b(?:" + p + r")", re.I)) for c, p in INSTITUTIONAL_SUB]


def subcategory(category, text):
    """Finer label within a category where one is defined (institutional), else None."""
    if category != "institutional":
        return None
    head = head_phrase(text or "")
    for basis in (head, text or ""):
        for c, rx in _INSTITUTIONAL_SUB_RE:
            if rx.search(basis):
                return c
    return "institutional_other"


def group(category, sub=None):
    """Coarse architectural group of a category (and subcategory)."""
    if category == "institutional" and sub == "civic":
        return "civic"
    return GROUPS.get(category, "unspecified")


def storeys(text):
    """Largest explicit storey count in the text, or None. 'multi-story' reads as 4."""
    best = None
    for m in _STOREY_RE.finditer(text or ""):
        tok = m.group(1).lower()
        n = int(tok) if tok.isdigit() else _NUM.get(tok)
        if n and n <= 200:
            best = n if best is None else max(best, n)
    if best is None and _MULTI_STOREY_RE.search(text or ""):
        best = 4
    return best


def head_phrase(text):
    m = HEAD_SPLIT.search(text or "")
    return (text[:m.start()] if m else text or "").strip()


def _earliest(text, with_pos=False):
    hits = []
    for order, (c, rx) in enumerate(_CAT_RE):
        m = rx.search(text)
        if m:
            hits.append((m.start(), order, c))
    if not hits:
        return (None, -1) if with_pos else None
    best = min(hits)
    return (best[2], best[0]) if with_pos else best[2]


def _object_after(head, cat_pos):
    """True when an object noun follows the building noun in the head: "school desk",
    "hospital bed", "church bench" -- the building word is a modifier there."""
    return any(m.start() > cat_pos for m in OBJECT_HEAD.finditer(head))


def classify(caption, name, tags, description="", sketchfab_cats=()):
    """Return (category, house_sub, flags, storeys). category None = not a building."""
    caption = caption or ""
    fallback = " ".join([name or "", " ".join(tags or [])])
    flags = {}
    if caption:
        head = head_phrase(caption)
        basis = caption
        cat, cat_pos = _earliest(head, with_pos=True)
        if cat is not None and _object_after(head, cat_pos):
            object_head, cat = True, None
        else:
            object_head = bool(OBJECT_HEAD.search(head)) and cat is None
        if cat is None and not object_head:
            if _GENERIC_RE.search(head):
                cat = "building_generic"
            elif STRONG_CUE.search(caption):
                # A vague head ("A rectangular two-level structure"): trust the body only
                # when it describes a building.
                cat = _earliest(caption)
                if cat is None and re.search(r"\bstructure\b", head, re.I):
                    cat = "building_generic"
        flags["object_head"] = object_head
    else:
        head = name or ""
        basis = fallback
        arch = ("architecture" in (sketchfab_cats or ())) or bool(STRONG_CUE.search(fallback))
        cat = _earliest(fallback) if arch else None
        if cat is None and arch and _GENERIC_RE.search(fallback):
            cat = "building_generic"
        flags["object_head"] = bool(OBJECT_HEAD.search(name or "")) and _earliest(name or "") is None

    n_storeys = storeys(basis)
    if cat == "building_generic" and n_storeys:
        cat = "skyscraper" if n_storeys >= 15 else ("midrise_generic" if n_storeys >= 4 else cat)
    if cat in ("apartment", "office", "midrise_generic", "commercial") and n_storeys and n_storeys >= 15:
        cat = "skyscraper"

    sub = None
    if cat == "house":
        for c, rx in _HOUSE_SUB_RE:
            if rx.search(basis):
                sub = c
                break
        sub = sub or "house:unspecified"

    head_and_name = " ".join([head, name or ""])
    all_text = " ".join([caption, fallback, (description or "")[:400]])
    for k, rx in _HEAD_EXCLUDE_RE.items():
        flags[k] = bool(rx.search(head if k == "part" else head_and_name))
    # How the object was MADE shows up anywhere in the caption ("a castle in a voxel-based
    # style", "a castle made of wooden blocks"), not just in the subject.
    flags["blocky_source"] = flags["blocky_source"] or bool(_HEAD_EXCLUDE_RE["blocky_source"].search(basis))
    flags["toy_or_miniature"] = flags["toy_or_miniature"] or bool(_TOY_BUILD.search(basis))
    for k, rx in _ANY_EXCLUDE_RE.items():
        flags[k] = bool(rx.search(basis))
    for k, rx in _STYLE_RE.items():
        flags[k] = bool(rx.search(all_text))
    flags["furnished_interior"] = len(set(m.group(0).lower() for m in _INTERIOR_WORD.finditer(basis))) >= 3         or bool(re.search(r"\blayout\b", head, re.I))
    flags["multi_building"] = bool(MULTI.search(basis) or MULTI_HEAD.search(basis))
    flags["not_finished"] = bool(NOT_FINISHED.search(all_text))
    flags["game_rip"] = bool(GAME_RIP.search(" ".join([name or "", " ".join(tags or []),
                                                       (description or "")[:600]])))
    return cat, sub, flags, n_storeys
