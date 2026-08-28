VIDEO_CAPTION = "Describe what happens in this video clip in 2-3 sentences. Focus on actions, objects, people, and locations. Be specific and factual."

MMLIFELONG_ASR_SYSTEM = """You are a professional video captioner. Your task is to rewrite raw speech transcripts from a livestream into clean, third-person video captions.

# Background
- The streamer is **IShowSpeed** (also known as "Speed"), a popular American YouTuber/streamer.
- These are IRL (In Real Life) livestream broadcasts spanning Feb 28 – Apr 20, 2025 (~51 days).
- He travels across multiple cities and countries during this period.

# Travel itinerary and locations
- **China**: Shanghai (Yu Garden, basketball court), Chongqing (light rail, river cruise, barbershop, parks), Changsha (shopping centers), Shenzhen, Beijing (Forbidden City, Great Wall), Nanjing
- **Mongolia**: visited museums, fan interactions
- **Hong Kong**: Victoria Harbour, gaming areas
- **London, UK**: 1V1 football challenge
- **Shaolin**: hiking at the foot of the mountain (warm-up, running, duck-walking, frog-jumping)

# Recurring people
- **Jackson Wang**: played basketball with IShowSpeed in Shanghai
- **Coco**: another creator whose China trip video IShowSpeed watched on Discord
- **Rosso, Alessa, Olivia**: companions/followers in Monster Hunter gameplay
- "Uncle in floral/colorful shirt": recurring fan encountered in multiple cities

# Common activities
- Street interactions with fans, trying local food, visiting landmarks
- Singing (especially on subways across different cities)
- Gaming (FRAGPUNK, Monster Hunter, Split or Steal)
- Dancing, talent shows, sports (football, basketball, tennis)
- Reacting to content on Discord, indoor chatting

# Common ASR Errors to Fix
- "Tongue King" / "Tongue King China" / "Tong King" → "Chongqing"
- "Gongzo" / "Guanzhou" → "Guangzhou"
- "Jia Kang" / "Jia Kang Gua" → likely a misheard Chinese name or phrase
- "Cyber City" (referring to Chongqing) → "Chongqing"
- "Chang sha" / "Chansha" → "Changsha"
- "Shenzen" → "Shenzhen"
- "Yu garden" / "You garden" → "Yu Garden (Shanghai)"

# Guidelines
1. Rewrite in **third person** ("IShowSpeed does X" not "I do X").
2. Filter out repetitive chat interaction noise ("spam the W's", "everybody subscribe", "like the stream").
3. Preserve meaningful content: locations, people met, events, actions, things said.
4. Fix obvious ASR errors, especially Chinese/Mongolian place names and person names.
5. If the transcript is purely filler/noise with no meaningful content, output a single short sentence summarizing the mood or setting.
6. Keep output concise: 1-3 sentences.

# Output
Output ONLY the rewritten caption. No JSON, no explanations."""

EGOLIFE_CAPTION_SYSTEM = """# Role and Objective

You are an expert egocentric video captioner. Your task is to transform raw video caption and transcript segments into concise, item-focused, first-person captions that clearly describe key actions, objects, and interactions from the camera wearer's perspective.

# Input Data Format

The input will be a JSON array where each element is an object with these required fields:

- `start`: Start timestamp in seconds (integer or float)
- `end`: End timestamp in seconds (integer or float)
- `text`: Caption or transcript content (string)
- `type`: Specifies either `caption` (visual action) or `transcript` (spoken dialogue)

# Caption Handling

- Captions describe what the camera wearer does; refine or merge them for clarity and focus on items/actions.
- Emphasize objects, tools in use, and the wearer's interactions.

# Transcript Handling

- Transcripts are lines of spoken dialogue.
- Preserve other speakers' names (e.g., `Shure:`).
- For the camera wearer, remove the name (e.g., `Jake:`) and present as direct first-person speech.

# Guidelines

1. Use first-person language (I/me/my).
2. Highlight key objects, tools, or interactions.
3. Keep captions concise, direct, and focused on items or actions.
4. Merge overlapping or redundant content.
5. Ensure captions remain in timestamp order and correspond to the original sequence.
6. Favor clarity and succinctness over extensive narration.

# Output Requirement

Output **only** the final consolidated caption string in plain text. Do **not** include checklists, explanations, timestamps, JSON, or any other formatting.

# Examples

**Example 1**
Input:
```json
[
  {"start": 120200, "end": 120202, "text": "Reach down and open the drawer.", "type": "caption"},
  {"start": 120203, "end": 120205, "text": "Grab the scissors.", "type": "caption"},
  {"start": 120206, "end": 120207, "text": "Jake: These should work.", "type": "transcript"}
]
```

Output:
```
I open the drawer and grab the scissors. "These should work," I say.
```

**Example 2**
Input:
```json
[
  {"start": 130310, "end": 130311, "text": "Set the plate on the table.", "type": "caption"},
  {"start": 130312, "end": 130313, "text": "Jake: Make sure it's centered.", "type": "transcript"},
  {"start": 130315, "end": 130316, "text": "Shure: Got it.", "type": "transcript"}
]
```

Output:
```
I place the plate on the table. "Make sure it's centered," I say. Shure replies, "Got it."
```
"""

MMLIFELONG_MERGE_SYSTEM = """You are an expert video captioner. Your task is to merge a visual description and a speech transcript into a single, coherent caption.

# Input
You will receive:
- **Visual**: A description of what is visually happening in the video segment.
- **Transcript**: What is being said/spoken during the same segment.

# Guidelines
1. Combine both sources into one fluent paragraph.
2. Keep the visual description as the primary narrative.
3. Integrate speech content naturally (e.g., "The narrator explains '...' while the camera shows").
4. If transcript is empty, return the visual description as-is.
5. If visual description is empty but transcript exists, describe the speech.
6. Be concise: 2-4 sentences.

# Output
Output ONLY the merged caption text. No JSON, no explanations."""

ASR_SYSTEM = {
    "month": MMLIFELONG_ASR_SYSTEM,
    "week": EGOLIFE_CAPTION_SYSTEM,
    "day": "Rewrite a raw game transcript into a clean third-person gameplay caption. Refer to the controlled character as the player. Preserve character, location, enemy, item, spell, quest, and dialogue names, including Chinese names when present. Remove repetitive filler. Keep the result concise at 1-3 sentences. Output only the rewritten caption.",
}

MERGE_SYSTEM = {
    "month": MMLIFELONG_MERGE_SYSTEM,
    "week": EGOLIFE_CAPTION_SYSTEM,
    "day": "Merge a visual description and a speech transcript into one coherent third-person gameplay caption. Refer to the controlled character as the player. Preserve exact character, location, enemy, item, spell, quest, action, outcome, and dialogue details. Keep the result concise at 2-4 sentences. Output only the merged caption.",
}

OPENIE_SYSTEM = """Your task is to extract named entities and construct an RDF (Resource Description Framework) graph from the given paragraph.

Respond with a JSON object containing two keys:
1. "named_entities": A list of named entities found in the paragraph.
2. "triples": A list of RDF triples, each as a 3-element list [subject, predicate, object].

Pay attention to the following requirements:
- Each triple should contain at least one, but preferably two, of the named entities.
- When resolving pronouns, if the pronoun refers to the first-person (e.g., I, me, my), keep it as "I" instead of replacing with terms like "speaker" or "narrator". For other pronouns, clearly resolve them to their specific names to maintain clarity.
"""
OPENIE_EXAMPLE = """Radio City
Radio City is India's first private FM radio station and was started on 3 July 2001.
It plays Hindi, English and regional songs.
Radio City recently forayed into New Media in May 2008 with the launch of a music portal - PlanetRadiocity.com that offers music related news, videos, songs, and other music-related features."""
OPENIE_EXAMPLE_OUTPUT = """{
    "named_entities": ["Radio City", "India", "3 July 2001", "Hindi", "English", "May 2008", "PlanetRadiocity.com"],
    "triples": [
        ["Radio City", "located in", "India"],
        ["Radio City", "is", "private FM radio station"],
        ["Radio City", "started on", "3 July 2001"],
        ["Radio City", "plays songs in", "Hindi"],
        ["Radio City", "plays songs in", "English"],
        ["Radio City", "forayed into", "New Media"],
        ["Radio City", "launched", "PlanetRadiocity.com"],
        ["PlanetRadiocity.com", "launched in", "May 2008"],
        ["PlanetRadiocity.com", "is", "music portal"],
        ["PlanetRadiocity.com", "offers", "news"],
        ["PlanetRadiocity.com", "offers", "videos"],
        ["PlanetRadiocity.com", "offers", "songs"]
    ]
}"""

SEMANTIC_SYSTEM = """You are tasked with extracting semantic knowledge from episodic triples. 
Your goal is to infer generalizable information that extends beyond the specific episode. 
Focus on capturing valid semantic triples that can guide reasoning about behavior, relationships, or preferences.

# What to Extract:
1. **Relationships**: social bonds or roles between entities  that persist over time
   (e.g., "Alice is a friend with Bob", "Jason is a teacher of Alice").
2. **Attributes & Preferences**: tendencies, likes/dislikes, personality-like traits, or behavioral habits
   (e.g., "Alice prefers not having dessert", "Bob enjoys music").
3. **Habits & Capabilities**: actions or patterns that suggest what an entity often does, can do, or tends to do
   (e.g., "Alice often helps friends", "Jason can give advice").
4. **Conceptual Knowledge**: directly useful facts that support reasoning, but avoid overly broad taxonomic statements
   (e.g., "Alice's office is near Cafe X", "Bob's gym is closed on Sundays").

# What to Avoid:
- **One-off events or transient states** (e.g., “ate pizza yesterday”, “was late once”) unless explicitly declared as a preference/role
- **Broad taxonomy or trivia unrelated to behavior** (e.g., “a laptop is electronics”, “Paris is in France”)
- **Speculative or mind-reading inferences without textual support** (e.g., motives, beliefs not evidenced)

# Important Notes:
- Prefer to base semantic triples on multiple supporting episodes. 
- BUT if a single episode clearly reflects a role, preference, habit, or capability, it is valid to include it.
- Each semantic triple MUST have at least one supporting episodic triple.
- Reduce duplication. If multiple episodic triples support the same or very similar semantic knowledge, merge them into one semantic triple rather than repeating.
- The `episodic_evidence[i]` list must always point to the indices that support `semantic_triples[i]`.
- Aim for broad coverage: extract as many valid semantic triples as reasonably supported by the input.

# Output Format:
Return ONLY a JSON object with the following two keys:
- `semantic_triples` (List[List[str]]): Each item is a triple [subject, predicate, object].
- `episodic_evidence` (List[List[int]]) : Each item is a list of **0-based** indices pointing to the input episodic triples 
   that support the corresponding semantic triple at the same position.
- The two lists MUST have the same length and aligned order.
- If no semantic knowledge is inferable, return:
  {"semantic_triples": [], "episodic_evidence": []}
"""
SEMANTIC_EXAMPLE_INPUT = """Episodic triples:
0. ["Alice", "talks to", "Bob"],
1. ["Alice", "laughs with", "Bob"],
2. ["Alice", "doesn't eat cake", "at restaurant"],
3. ["Alice", "shares personal stories with", "Bob"],
4. ["Alice", "brings coffee to", "Bob"],
5. ["Jason", "talks to", "Alice"],
6. ["Alice", "declines dessert", "at friend's house"]
"""
SEMANTIC_EXAMPLE_OUTPUT = """{
  "semantic_triples": [
    ["Alice", "is a friend with", "Bob"],
    ["Alice", "prefers", "not having dessert"]
  ],
  "episodic_evidence": [
    [0, 1, 3],
    [2, 6]
  ]
}
"""

ENTITY_FILTER = """Below is a list of entities extracted from video captions. Remove UI or layout elements, body parts, generic visual descriptions, generic actions or emotions, full sentences, vague words, generic person descriptions, and interface elements. Keep named people, places, countries, cities, specific objects, instruments, food, animals, games, sports, platforms, apps, and identifying clothing items. Return a JSON array containing only entity names from the input list.

Entity list:
{entity_list}"""

TOPIC_SYSTEM = "You are a precise video fact extractor. Output valid JSON only."

TOPIC = {
    "month": """In a livestream video by IShowSpeed, the entity "{entity}" appears in multiple segments at different times.

Below are the 30-second episode captions where "{entity}" is mentioned:

{episodes_text}

Extract facts about "{entity}" that would help answer questions about the livestream, such as:
- WHAT happened with "{entity}"? (specific actions, events, outcomes)
- WHERE did it happen? (city, venue, location — important for travel streams)
- WHO was involved? (IShowSpeed, fans, specific people by name/description)
- WHAT changed? (state changes, before/after)

Focus on:
1. **Specific events**: what exactly happened, not vague descriptions
2. **Location context**: which city or place this occurred in
3. **People involved**: IShowSpeed, specific fans, celebrities, staff
4. **Concrete details**: song names, food names, game names, scores, prices

Always refer to the streamer as "IShowSpeed". Never use "he", "they", "the man".

Skip trivial mentions where "{entity}" appears in passing without meaningful action.

Good facts:
- "IShowSpeed sang {entity} while riding the bus in Chongqing"
- "A fan wearing a {entity} jersey approached IShowSpeed on the street in Shanghai"
- "IShowSpeed tried {entity} flavor ice cream at the dessert shop in Chengdu"

Bad facts (skip these):
- "IShowSpeed looked at {entity}" (no meaningful action)
- "{entity} was visible in the background" (no action)

Output a JSON array (empty array [] if no meaningful facts):
[
  {{"time": "01:30:00", "fact": "descriptive fact about {entity}"}},
  {{"time": "02:15:00", "fact": "another fact"}}
]""",
    "week": """In a first-person video, the entity "{entity}" appears in multiple episodes at different times.
The wearer is A1_JAKE (also referred to as "I"), living with housemates: Shure, Katrina, Tasha, Lucia, Alice, and sometimes visitors (Nicous, Choiszt, Jack, Violet, Lucy).

Below are the 30-second episode captions where "{entity}" is mentioned:

{episodes_text}

Extract facts about "{entity}" that help answer these types of questions:

1. **Who & When** (EntityLog): Who used/touched/moved "{entity}" first? Whose "{entity}" is this? Where was "{entity}" before?
2. **What happened** (EventRecall): When was "{entity}" first/last mentioned or discussed? What happened with "{entity}" last time?
3. **Relationships** (RelationMap): Who helped whom with "{entity}"? Who gave/passed "{entity}" to whom? Who was together when using "{entity}"?
4. **Plans & Decisions** (TaskMaster): Who suggested/planned to buy/make/use "{entity}"? What was decided about "{entity}"?
5. **Habits & Preferences** (HabitInsight): Who always/usually/often uses "{entity}"? Who likes/dislikes "{entity}"? What does someone typically do with "{entity}"?

Rules:
- Always use specific person names (A1_JAKE, Shure, Katrina, Tasha, Lucia, Alice, Nicous, Choiszt, Jack). NEVER use "I", "he", "she", "they", "we".
- Include the WHERE (which room, location) when relevant.
- Record preferences and habits ("Tasha refused the {entity}", "Shure always drinks {entity} in the morning").
- Record plans and discussions ("They decided to order {entity} for lunch", "A1_JAKE suggested buying {entity}").
- Skip trivial mentions where "{entity}" appears in background without meaningful action.

Good facts:
- "Shure handed the {entity} to A1_JAKE in the kitchen"
- "Lucia and A1_JAKE set up the {entity} together in the living room"
- "Katrina said she doesn't like {entity}"
- "A1_JAKE suggested ordering {entity} for dinner, everyone agreed"
- "Tasha was the first person to use the {entity} this morning"

Bad facts (skip these):
- "A1_JAKE picked up the {entity}" (no meaningful interaction)
- "The {entity} was on the table" (static description, no action)

Output a JSON array (empty array [] if no meaningful facts):
[
  {{"time": "DAY1 17:45", "fact": "descriptive fact with person names, location, and action"}},
  {{"time": "DAY2 11:00", "fact": "another fact"}}
]""",
    "day": """In a gameplay video, the entity "{entity}" appears in multiple segments. Extract concrete facts about the player's encounters, actions, progression, locations, fights, items, spells, quests, dialogue, state changes, counts, and outcomes involving it. Preserve exact English and Chinese names. Skip static interface or passing mentions. Return a JSON array of objects with time in HH:MM:SS and fact.

{episodes_text}""",
}

ACTIVITY = {
    "month": """You are analyzing a livestream video by IShowSpeed. Below are 10-minute summary segments.

Create a detailed chronological event log. For EVERY distinct scene or activity, extract:
- name: specific name (e.g., "Eating hot pot at Haidilao restaurant", "Basketball 1v1 with fan near Oriental Pearl Tower")
- time_range: "HH:MM:SS-HH:MM:SS"
- location: specific place name (e.g., "Chengdu night market", "wrestling arena", "in the car")
- summary: 2-3 sentences. Must include:
  * Specific actions (NOT "interacted with fans" — instead "high-fived 3 fans, signed a jersey, took selfies")
  * Named objects (food names, song names, game names, gift items)
  * Outcomes and reactions
- key_entities: specific objects, food, songs, games mentioned

BAD: "IShowSpeed explored the area and met fans."
GOOD: "IShowSpeed ate spicy duck tongue and Sichuan peppercorn chicken at the hot pot restaurant. He struggled with chopsticks and drank three glasses of water from the spiciness."

Extract 5-10 events from these segments. Focus on significant moments (receiving a gift, singing a song, playing a game, visiting a venue), not trivial transitions.

Segments:
{segments}
""",
    "week": """You are analyzing a first-person daily life video. The narrator is A1_JAKE, living with housemates: Shure, Katrina, Tasha, Lucia, Alice, and sometimes visitors (Nicous, Choiszt, Violet, Jack, Liu Ting).

Below are 10-minute summary segments from {date}.

For each major activity that happened on this day, extract:
- name: a short descriptive name (e.g., "Making a cake", "Shopping at supermarket")
- time_range: approximate start and end time (e.g., "11:00:00-12:30:00")
- summary: 2-3 sentences describing what happened. Must include:
  * WHO specifically participated (by name, never "he/she/they/I/we")
  * WHO helped whom, WHO worked together
  * WHAT concrete actions were taken (not vague words like "coordinated", "handled", "managed")
  * WHAT was the outcome or decision made
- key_entities: important objects or places involved (do NOT include person names here — only objects and places)

Return a JSON list:
[
  {{
    "name": "activity name",
    "time_range": "HH:MM:SS-HH:MM:SS",
    "summary": "detailed description with person names, specific actions, and outcomes",
    "key_entities": ["object1", "place1"]
  }}
]

BAD summary: "A1_JAKE coordinated box stacking and workspace assembly planning with the group."
GOOD summary: "A1_JAKE and Shure carried boxes upstairs and assembled a display rack. Lucia asked about placement and helped hold the frame. Katrina suggested making the rack longer. The rack was installed with a whiteboard on top."

Focus on significant, goal-directed activities (cooking, shopping, meetings, performances, crafts), not trivial actions. Aim for 5-15 activities per day.

Segments:
{segments}
""",
    "day": """You are analyzing a gameplay video. Create a detailed chronological event log from these 10-minute summaries. For every significant gameplay activity, extract name, time_range as HH:MM:SS-HH:MM:SS, location, a 2-3 sentence summary with exact characters, enemies, actions, items, spells, quests, outcomes and dialogue, and key_entities. Include exploration, combat, progression, discoveries, upgrades, failures, retries, and state changes; avoid idle moments.

Segments:
{segments}""",
}

EVENT_CHAIN = {
    "month": """Below is a chronological event log from one IShowSpeed livestream broadcast.

Identify RECURRING THEMES — things that happen 2+ times in this broadcast. Examples:
- Singing the same song at different moments
- Fans giving gifts (list each gift separately)
- Eating different foods at different places
- Playing the same game multiple rounds
- Getting in and out of a car/bus
- Fan encounters with specific items (e.g., Messi jerseys)
- Doing push-ups, backflips, or physical challenges
- Taking photos with fans at different locations

For each recurring theme, list EVERY occurrence as a separate step with:
- start_time, end_time: exact time range
- description: WHAT specifically happened THIS time (not a generic summary)
  * Always say "IShowSpeed" (never "he/they/the streamer")
  * Include specific details that distinguish this occurrence from others

Also identify KEY VENUE SEQUENCES — visiting a major location with multiple sub-events:
- e.g., "Shaolin Temple visit" with steps: arrived → watched kung fu demo → tried martial arts → took photos → left by bus

Return JSON:
[
  {{
    "name": "theme or venue name",
    "type": "recurring" or "venue_sequence",
    "steps": [
      {{"start_time": "HH:MM:SS", "end_time": "HH:MM:SS", "description": "specific description"}}
    ],
    "key_entities": ["entity1", "entity2"]
  }}
]

Return [] if nothing qualifies. Be thorough — don't miss recurring events.

Events:
{activities}
""",
    "week": """Below are daily activities from a 7-day first-person video. The participants are: A1_JAKE, Shure, Katrina, Tasha, Lucia, Alice, and visitors Nicous, Choiszt.

Find activities that are related across different time periods. This includes:
- Different stages of the same project (e.g., "discussed cake plan" DAY1 → "bought ingredients" DAY3 → "baked the cake" DAY5)
- Repeated occurrences of the same activity (e.g., "brewed coffee" on DAY2, DAY3, DAY4)
- Ongoing work done intermittently (e.g., "room decoration" on DAY1 morning and DAY2 evening)
- Same type of errand repeated (e.g., "shopping at supermarket" on DAY1, DAY3, DAY5)

Only output events that have 2 or more related activities across different time periods. Ignore standalone activities. Be thorough and find as many as possible.

For each event, output:
- name: specific name
- steps: list of {{day, start_time, end_time, description}}
  * description must name specific people involved (never "he/she/they/I/we")
  * description must state concrete actions and outcomes, not vague summaries
  * description should explain how this step relates to the previous step (continuation, escalation, completion, etc.)
- key_entities: important objects and places only (do NOT include person names)

BAD description: "A1_JAKE handled coffee preparation while the group reviewed the shopping list."
GOOD description: "Shure brewed pour-over coffee for everyone in the courtyard. A1_JAKE served espresso shots. Lucia and Katrina discussed which coffee beans to buy for the next day."

Return JSON list.

Daily activities:
{activities}
""",
    "day": """Below is a chronological event log from a gameplay timeline. Identify recurring gameplay activities and multi-step progression sequences, including repeated fights, retries, exploration of the same area, quest stages, item or spell progression, and boss or venue sequences. Output only results with at least two related steps. For each result return name, type as recurring or venue_sequence, steps with start_time, end_time and specific descriptions, and key_entities.

Events:
{activities}""",
}

WEEK_EVENT_CHAIN_ROUND2 = """Below are daily activities from a 7-day first-person video, and a list of cross-time event chains already found.

Your task: find ADDITIONAL cross-time event chains that were MISSED in the first round. Do not repeat any chain already found.

Look specifically for:
- Recurring daily routines (meals, cleaning, waking up, sleeping)
- Equipment/device management across days (cameras, computers, glasses, phones)
- Social activities that repeat (games, outings, group discussions)
- Food/cooking activities across days (hot pot, cake, coffee, barbecue)
- Any other related activities across different time periods

Already found chains (do NOT repeat these):
{found_names}

Daily activities:
{daily_text}

For each NEW event chain, output:
- name: specific name
- steps: list of {{day, start_time, end_time, description}}
  * description must name specific people involved (never "he/she/they/I/we")
  * description must state concrete actions and outcomes
- key_entities: important objects and places only (do NOT include person names)

Return JSON list. If no additional chains found, return [].
"""

ENRICH = {
    "month": """Event chain: "{name}"
Time window: {start_time} - {end_time}

Below are 30-second captions from this time window. Extract details ONLY about "{name}".

Captions:
{captions}

Write 2-4 sentences with maximum specificity:
- Count things precisely (how many times, how many items)
- Name every person, object, food, song, game involved
- Describe the exact sequence of actions
- Note the outcome or result

BAD: "IShowSpeed sang a song in the car."
GOOD: "IShowSpeed sang 'Sunshine Rainbow Little White Horse' twice while sitting in the back seat of the limousine. The driver and two staff members clapped along. He forgot the lyrics midway through the second attempt."

Output: {{"description": "your detailed description"}}""",
    "week": """You are analyzing a first-person daily life video. The narrator is A1_JAKE, living with housemates: Shure, Katrina, Tasha, Lucia, Alice, and sometimes visitors (Nicous, Choiszt).

This is one step of the event chain "{name}".
Time: {day} {start_time} - {end_time}

Below are the 30-second episode captions from this time period. Extract ONLY information relevant to "{name}". Ignore unrelated content.

Captions:
{captions}

Write 2-5 sentences describing what happened in this step, focusing on:
- WHO specifically participated (by name, never "I/he/she/they/we")
- WHO helped whom, WHO gave what to whom
- WHAT concrete actions were taken and what was the outcome
- WHAT decisions were made

Output a single JSON string (just the description text):
{{"description": "detailed description here"}}""",
    "day": """This is one step of the gameplay event "{name}" from {start_time} to {end_time}. Extract only relevant details from the captions. Preserve exact character, enemy, location, item, spell, quest, action, count, dialogue, progression and outcome information. Return a JSON object with description containing 2-5 sentences.

{captions}""",
}
