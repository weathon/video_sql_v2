VIDEO_CAPTION = "Describe what happens in this video clip in 2-3 sentences. Focus on actions, objects, people, and locations. Be specific and factual."

ASR_SYSTEM = {
    "month": "Rewrite a raw speech transcript from an IShowSpeed livestream into a clean third-person video caption. Preserve meaningful locations, people, events, actions, and exact spoken content. Fix obvious recognition errors and remove repetitive chat noise. Refer to the streamer as IShowSpeed. Keep the result concise at 1-3 sentences. Output only the rewritten caption.",
    "week": "Rewrite a raw speech transcript from A1_JAKE's first-person daily-life video into a clean caption. Preserve speaker names, plans, decisions, preferences, object references, and dialogue. Refer to the camera wearer as A1_JAKE and preserve the names Shure, Katrina, Tasha, Lucia, Alice, Nicous, Choiszt, Violet, Jack, and Liu Ting when present. Keep the result concise at 1-3 sentences. Output only the rewritten caption.",
    "day": "Rewrite a raw game transcript into a clean third-person gameplay caption. Refer to the controlled character as the player. Preserve character, location, enemy, item, spell, quest, and dialogue names, including Chinese names when present. Remove repetitive filler. Keep the result concise at 1-3 sentences. Output only the rewritten caption.",
}

MERGE_SYSTEM = {
    "month": "Merge a visual description and a speech transcript into one coherent caption. Keep the visual description as the primary narrative, integrate speech naturally, refer to the streamer as IShowSpeed, and preserve specific people, objects, actions, locations, and spoken content. Keep the result concise at 2-4 sentences. Output only the merged caption.",
    "week": "Merge a visual description and a speech transcript into one coherent caption about A1_JAKE's first-person daily life. Preserve specific people, objects, actions, locations, plans, decisions, preferences, and dialogue. Name participants when supported. Keep the result concise at 2-4 sentences. Output only the merged caption.",
    "day": "Merge a visual description and a speech transcript into one coherent third-person gameplay caption. Refer to the controlled character as the player. Preserve exact character, location, enemy, item, spell, quest, action, outcome, and dialogue details. Keep the result concise at 2-4 sentences. Output only the merged caption.",
}

OPENIE_SYSTEM = "Your task is to extract named entities and construct an RDF graph from the given paragraph. Respond with a JSON object containing named_entities, a list of named entities, and triples, a list of 3-element [subject, predicate, object] lists. Each triple should contain at least one, preferably two, named entities. Resolve non-first-person pronouns to specific names when the paragraph supports it. Keep first-person pronouns as I."
OPENIE_EXAMPLE = "Radio City is India's first private FM radio station and was started on 3 July 2001. It plays Hindi and English songs. Radio City launched PlanetRadiocity.com in May 2008."
OPENIE_EXAMPLE_OUTPUT = '{"named_entities":["Radio City","India","3 July 2001","Hindi","English","PlanetRadiocity.com","May 2008"],"triples":[["Radio City","located in","India"],["Radio City","started on","3 July 2001"],["Radio City","plays songs in","Hindi"],["Radio City","plays songs in","English"],["Radio City","launched","PlanetRadiocity.com"],["PlanetRadiocity.com","launched in","May 2008"]]}'

SEMANTIC_SYSTEM = "Extract semantic knowledge from episodic triples. Infer only generalizable relationships, attributes, preferences, habits, and capabilities supported by the input. Do not copy one-off episode details unless they support a persistent fact. Return semantic_triples as 3-element [subject, predicate, object] lists and episodic_evidence as aligned lists of zero-based indices into the input episodic triples. If no semantic knowledge is inferable, return both lists empty."

ENTITY_FILTER = """Below is a list of entities extracted from video captions. Remove UI or layout elements, body parts, generic visual descriptions, generic actions or emotions, full sentences, vague words, generic person descriptions, and interface elements. Keep named people, places, countries, cities, specific objects, instruments, food, animals, games, sports, platforms, apps, and identifying clothing items. Return a JSON array containing only entity names from the input list.

Entity list:
{entity_list}"""

TOPIC = {
    "month": """In an IShowSpeed livestream, the entity "{entity}" appears in multiple segments. Extract concrete facts that help answer what happened, where it happened, who was involved, and what changed. Refer to the streamer as IShowSpeed. Preserve names, locations, objects, songs, food, games, scores, prices, outcomes, and state changes. Skip passing mentions without meaningful action. Return a JSON array of objects with time in HH:MM:SS and fact.

{episodes_text}""",
    "week": """In A1_JAKE's first-person daily-life video, the entity "{entity}" appears in multiple episodes. Extract facts about who used, touched, moved, gave, planned, discussed, liked, disliked, or repeatedly interacted with it, including location, decisions, outcomes, habits, and preferences. Use specific person names when supported. Skip static background mentions. Return a JSON array of objects with time in HH:MM:SS and fact.

{episodes_text}""",
    "day": """In a gameplay video, the entity "{entity}" appears in multiple segments. Extract concrete facts about the player's encounters, actions, progression, locations, fights, items, spells, quests, dialogue, state changes, counts, and outcomes involving it. Preserve exact English and Chinese names. Skip static interface or passing mentions. Return a JSON array of objects with time in HH:MM:SS and fact.

{episodes_text}""",
}

ACTIVITY = {
    "month": """You are analyzing an IShowSpeed livestream. Create a detailed chronological event log from these 10-minute summaries. For every distinct significant scene or activity, extract name, time_range as HH:MM:SS-HH:MM:SS, location, a 2-3 sentence summary with specific actions, named objects, outcomes and reactions, and key_entities. Extract 5-10 events and avoid trivial transitions.

Segments:
{segments}""",
    "week": """You are analyzing A1_JAKE's first-person daily-life video. Extract significant goal-directed activities from these 10-minute summaries with name, time_range as HH:MM:SS-HH:MM:SS, location, a 2-3 sentence summary naming supported participants, concrete actions, outcomes and decisions, and key_entities containing objects and places. Avoid trivial actions.

Segments:
{segments}""",
    "day": """You are analyzing a gameplay video. Create a detailed chronological event log from these 10-minute summaries. For every significant gameplay activity, extract name, time_range as HH:MM:SS-HH:MM:SS, location, a 2-3 sentence summary with exact characters, enemies, actions, items, spells, quests, outcomes and dialogue, and key_entities. Include exploration, combat, progression, discoveries, upgrades, failures, retries, and state changes; avoid idle moments.

Segments:
{segments}""",
}

EVENT_CHAIN = {
    "month": """Below is a chronological event log from an IShowSpeed timeline. Identify recurring themes that occur at least twice and key venue sequences with multiple sub-events. For each result return name, type as recurring or venue_sequence, steps with start_time, end_time and a specific description that names IShowSpeed and distinguishes the occurrence, and key_entities. Return an empty list if nothing qualifies.

Events:
{activities}""",
    "week": """Below is a chronological event log from A1_JAKE's seven-day first-person timeline. Identify related stages of the same project, repeated routines, ongoing work, repeated errands, and multi-step activities across the timeline. Output only results with at least two related steps. For each result return name, type as recurring or venue_sequence, steps with start_time, end_time and descriptions naming supported participants, and key_entities containing objects and places.

Events:
{activities}""",
    "day": """Below is a chronological event log from a gameplay timeline. Identify recurring gameplay activities and multi-step progression sequences, including repeated fights, retries, exploration of the same area, quest stages, item or spell progression, and boss or venue sequences. Output only results with at least two related steps. For each result return name, type as recurring or venue_sequence, steps with start_time, end_time and specific descriptions, and key_entities.

Events:
{activities}""",
}

ENRICH = {
    "month": """Event: "{name}"
Time window: {start_time} - {end_time}
Extract details only about the named event from the captions. Write 2-4 sentences with precise counts, named people, objects, food, songs or games, the exact action sequence, and outcome. Return a JSON object with description.

{captions}""",
    "week": """This is one step of the A1_JAKE daily-life event "{name}" from {start_time} to {end_time}. Extract only relevant details from the captions. Name supported participants, who helped or gave what to whom, concrete actions, outcomes, and decisions. Return a JSON object with description containing 2-5 sentences.

{captions}""",
    "day": """This is one step of the gameplay event "{name}" from {start_time} to {end_time}. Extract only relevant details from the captions. Preserve exact character, enemy, location, item, spell, quest, action, count, dialogue, progression and outcome information. Return a JSON object with description containing 2-5 sentences.

{captions}""",
}
