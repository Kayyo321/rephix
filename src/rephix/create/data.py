"""Fixed, public probe sets used for retention and new-topic evaluation.

These are written for Rephix and contain no user evidence. They are deliberately
generic so that retention measures collateral change, not the evidence domain.
"""

# Neutral document starts: the base's own continuations of these form the retain set.
NEUTRAL_PROMPTS = [
    "The weather report for the coming week says",
    "To install the package, first",
    "The football match ended with",
    "In mathematics, a prime number is",
    "The recipe calls for two cups of flour and",
    "Photosynthesis is the process by which",
    "The city council voted on Tuesday to",
    "When the train arrived at the station,",
    "A good night's sleep helps",
    "The museum's new exhibition features",
    "Computers store information using",
    "The river flows through the valley and",
    "Our customer support team is available",
    "The history of the bicycle begins",
    "Volcanoes form when",
    "The children played in the park until",
    "Interest rates affect borrowing because",
    "The software update fixes a bug that",
    "Bees communicate with each other by",
    "The novel tells the story of",
    "Water boils at a lower temperature at high altitude because",
    "The company announced quarterly earnings that",
    "To change a flat tire, you need",
    "The orchestra rehearsed the symphony",
    "Mountains are formed by",
    "The doctor recommended that patients",
    "Electric cars differ from petrol cars in that",
    "The library will be closed on",
    "Chess is a game in which",
    "The first step in writing an essay is",
    "Coffee beans are grown in",
    "The spacecraft entered orbit after",
]

# New topics for free-running generation, unrelated to any evidence domain.
NEW_TOPICS = [
    "The economics of lighthouse maintenance",
    "How desert plants survive long droughts",
    "The design of suspension bridges",
    "Why cities plant trees along streets",
    "The early history of the postal service",
    "How glaciers shape valleys",
    "The role of bees in agriculture",
    "Methods for preserving food without refrigeration",
    "The invention of the printing press",
    "How weather forecasts are made",
    "The migration of Arctic terns",
    "Public libraries and their purpose",
]

# Cloze facts with one expected leading word. Greedy continuation is scored by whether
# its first word equals the answer (case-insensitive). Measures retained knowledge.
FACT_CLOZE = [
    ("The capital of France is", "paris"),
    ("The capital of Japan is", "tokyo"),
    ("The capital of Italy is", "rome"),
    ("The capital of Germany is", "berlin"),
    ("The capital of Spain is", "madrid"),
    ("The capital of Egypt is", "cairo"),
    ("The largest planet in the solar system is", "jupiter"),
    ("Water is made of hydrogen and", "oxygen"),
    ("The opposite of hot is", "cold"),
    ("The color of the clear daytime sky is", "blue"),
    ("Two plus two equals", "four"),
    ("The first month of the year is", "january"),
    ("The day after Monday is", "tuesday"),
    ("The chemical symbol for gold is", "au"),
    ("The author of Romeo and Juliet is William", "shakespeare"),
    ("The Earth orbits the", "sun"),
    ("Ice is frozen", "water"),
    ("The longest river in Africa is the", "nile"),
    ("The currency of the United States is the", "dollar"),
    ("A baby cat is called a", "kitten"),
    ("The tallest mountain on Earth is Mount", "everest"),
    ("The language spoken in Brazil is", "portuguese"),
    ("The planet known as the Red Planet is", "mars"),
    ("Paris is the capital of", "france"),
    ("The freezing point of water in Celsius is", "0"),
    ("Honey is made by", "bees"),
    ("The season after winter is", "spring"),
    ("Shakespeare was born in", "stratford"),
    ("The inventor of the telephone was Alexander Graham", "bell"),
    ("The Great Wall is located in", "china"),
]
