## Hard constraints
jr1_HARD_CONSTRAINTS = [
    {"type": "hard", "text": "destination: Marseille, France", "user_skipped": False},
    {"type": "hard", "text": "origin: Frankfurt, Germany", "user_skipped": False},
    {"type": "hard", "text": "travel_dates: 2026-07-31 to 2026-08-07", "user_skipped": False},
    {"type": "hard", "text": "travelers: 5 adults", "user_skipped": False},
    {"type": "hard", "text": "budget: 7500 EUR total", "user_skipped": False},
    {"type": "hard", "text": "accommodation: Hotel", "user_skipped": False},
    {"type": "hard", "text": "transport: Flight", "user_skipped": False},
    {"type": "hard", "text": "interests: boys vacation, parties, adventures, fun", "user_skipped": False},
]

jr2_HARD_CONSTRAINTS = [
    {"type": "hard", "text": "destination: Talinn, Estonia", "user_skipped": False},
    {"type": "hard", "text": "origin: Frankfurt, Germany", "user_skipped": False},
    {"type": "hard", "text": "travel_dates: 2026-09-20 to 2026-09-24", "user_skipped": False},
    {"type": "hard", "text": "travelers: 2 adults", "user_skipped": False},
    {"type": "hard", "text": "budget: 1500 EUR total", "user_skipped": False},
    {"type": "hard", "text": "accommodation: Hotel", "user_skipped": False},
    {"type": "hard", "text": "transport: Flight", "user_skipped": False},
    {"type": "hard", "text": "interests: culture, nature, activities, couple", "user_skipped": False},
]

jr3_HARD_CONSTRAINTS = [
    {"type": "hard", "text": "destination: Lisbon, Portugal", "user_skipped": False},
    {"type": "hard", "text": "origin: Munich, Germany", "user_skipped": False},
    {"type": "hard", "text": "travel_dates: 2026-07-01 to 2026-07-10", "user_skipped": False},
    {"type": "hard", "text": "travelers: 1 adult", "user_skipped": False},
    {"type": "hard", "text": "budget: 1500 EUR total", "user_skipped": False},
    {"type": "hard", "text": "accommodation: Hotel", "user_skipped": False},
    {"type": "hard", "text": "transport: Flight", "user_skipped": False},
    {"type": "hard", "text": "interests: summer hide away, workation, surfing, socializing", "user_skipped": False},
]

jr4_HARD_CONSTRAINTS = [
    {"type": "hard", "text": "destination: Johannesburg, South Africa", "user_skipped": False},
    {"type": "hard", "text": "origin: Berlin, Germany", "user_skipped": False},
    {"type": "hard", "text": "travel_dates: 2026-08-15 to 2026-08-25", "user_skipped": False},
    {"type": "hard", "text": "travelers: 4 adults", "user_skipped": False},
    {"type": "hard", "text": "budget: 8000 EUR total", "user_skipped": False},
    {"type": "hard", "text": "accommodation: Hotel", "user_skipped": False},
    {"type": "hard", "text": "transport: Flight", "user_skipped": False},
    {"type": "hard", "text": "interests: safari, food, bonding with teenager children", "user_skipped": False},
]

jr5_HARD_CONSTRAINTS = [
    {"type": "hard", "text": "destination: Managua, Nicaragua", "user_skipped": False},
    {"type": "hard", "text": "origin: Frankfurt, Germany", "user_skipped": False},
    {"type": "hard", "text": "travel_dates: 2026-07-12 to 2026-07-22", "user_skipped": False},
    {"type": "hard", "text": "travelers: 1 adult", "user_skipped": False},
    {"type": "hard", "text": "budget: 2500 EUR total", "user_skipped": False},
    {"type": "hard", "text": "accommodation: Hotel", "user_skipped": False},
    {"type": "hard", "text": "transport: Flight", "user_skipped": False},
    {"type": "hard", "text": "interests: backpacking, low budget, adventures, meeting other travelers", "user_skipped": False},
]

jr6_HARD_CONSTRAINTS = [
    {"type": "hard", "text": "destination: Hanoi, Vietnam", "user_skipped": False},
    {"type": "hard", "text": "origin: Basel, Switzerland", "user_skipped": False},
    {"type": "hard", "text": "travel_dates: 2026-09-01 to 2026-09-10", "user_skipped": False},
    {"type": "hard", "text": "travelers: 2 adults", "user_skipped": False},
    {"type": "hard", "text": "budget: 3500 EUR total", "user_skipped": False},
    {"type": "hard", "text": "accommodation: Hotel", "user_skipped": False},
    {"type": "hard", "text": "transport: Flight", "user_skipped": False},
    {"type": "hard", "text": "interests: young couple, romantic, culture, spirituality, getting to know each other better", "user_skipped": False},
]

## Queries
jr1 = """Hey, I'm looking to plan a boys trip to Marseille, France! There will be 5 of us flying from Frankfurt, Germany, and we're planning to go from July 31st to August 7th, 2026. We're all about having a great time — parties, adventures, and just a lot of fun. Our total budget is 7500 EUR. We'd like to fly there and stay in a hotel. Can you help us put together an epic itinerary?"""

jr2 = """My partner and I are planning a short trip to Tallinn, Estonia from September 20th to 24th, 2026. We'll be flying from Frankfurt, Germany, just the two of us. We're really into culture, nature, and doing activities together as a couple. Our total budget is 1500 EUR. We'd like a hotel and a flight. Could you suggest a nice itinerary for us?"""

jr3 = """I'm planning a solo workation in Lisbon, Portugal from July 1st to July 10th. I'll be flying from Munich, Germany. I'm looking for a place where I can work remotely but also enjoy some surfing, soak up the summer vibe, and meet new people. My budget is 1500 EUR in total. Please book me a flight and a hotel. Can you help plan this out?"""

jr4 = """We're a group of 4 adults — parents with their teenage kids — and we'd love to go on a safari trip to Johannesburg, South Africa! We're flying from Berlin, Germany and plan to be there from August 15th to August 25th, 2026. It's really about bonding as a family, experiencing wildlife, and enjoying great local food. Our total budget is 8000 EUR. We'll need flights and hotel accommodation. Could you help us plan this adventure?"""

jr5 = """I'm heading out on a solo backpacking adventure to Managua, Nicaragua from July 12th to July 22nd, 2026, flying out of Frankfurt, Germany. I love traveling on a budget, meeting fellow travelers, and going on adventures off the beaten path. My total budget is 2500 EUR — though I'd like to keep costs as low as possible. I need a flight and a hotel. Can you help me plan this trip?"""

jr6 = """My partner and I are planning a romantic trip to Hanoi, Vietnam from September 1st to September 10th, 2026. We're flying from Basel, Switzerland. It's kind of a special trip for us — we're still getting to know each other and would love to explore local culture and spirituality together. Our total budget is 3500 EUR. We'd like to fly there and stay in a hotel. Could you create a meaningful itinerary for us?"""

