import os
import time
from datetime import datetime
import numpy as np
import pandas as pd
import requests

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
STEAMSPY_URL = "https://steamspy.com/api.php"
STEAM_STORE_URL = "https://store.steampowered.com/api/appdetails"
np.random.seed(42)

GENRES = {
    "Action": (19.99, 59.99),
    "RPG": (29.99, 59.99),
    "Strategy": (14.99, 49.99),
    "Indie": (4.99, 24.99),
    "Adventure": (14.99, 49.99),
    "Simulation": (14.99, 39.99),
    "Sports": (29.99, 59.99),
    "Racing": (19.99, 49.99),
    "Puzzle": (2.99, 14.99),
    "Casual": (2.99, 9.99),
}

SALE_CALENDAR = [
    (2, 5, 12, "Lunar New Year Sale", (15, 40)),
    (3, 15, 22, "Spring Sale", (15, 35)),
    (6, 23, 30, "Summer Sale", (30, 75)),
    (7, 1, 7, "Summer Sale", (30, 75)),
    (10, 25, 31, "Halloween Sale", (20, 50)),
    (11, 20, 27, "Autumn Sale", (25, 60)),
    (12, 19, 31, "Winter Sale", (30, 75)),
]

GENRE_SALE_PROBABILITY = {
    "Action": 0.85,
    "RPG": 0.80,
    "Strategy": 0.75,
    "Indie": 0.90,
    "Adventure": 0.80,
    "Simulation": 0.70,
    "Sports": 0.65,
    "Racing": 0.70,
    "Puzzle": 0.90,
    "Casual": 0.95,
}

def fetch_steamspy_top_games():
    try:
        resp = requests.get(
            STEAMSPY_URL,
            params={"request": "top100forever"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        games = []
        for appid_str, info in data.items():
            appid = int(appid_str)
            name = info.get("name", f"Game {appid}")
            genre_raw = info.get("genre", "")
            genre = genre_raw.split(",")[0].strip() if genre_raw else "Action"
            price_raw = info.get("price", "0")
            price = int(price_raw) / 100.0 if price_raw else 0.0

            if price > 0:
                games.append({
                    "appid": appid,
                    "name": name,
                    "genre": genre if genre in GENRES else "Action",
                    "base_price": price,
                    "release_year": 2020,
                })
        return games if len(games) >= 30 else None
    except Exception:
        return None

def enrich_from_steam(games, max_games=80):
    enriched = 0
    for game in games[:max_games]:
        try:
            resp = requests.get(
                STEAM_STORE_URL,
                params={"appids": game["appid"], "cc": "us"},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()

            app_data = data.get(str(game["appid"]), {}).get("data", {})
            if app_data:
                genres = app_data.get("genres", [])
                if genres:
                    g = genres[0].get("description", "Action")
                    game["genre"] = g if g in GENRES else "Action"

                release = app_data.get("release_date", {}).get("date", "")
                if release:
                    try:
                        year = int(release.split(",")[-1].strip())
                        if 2010 <= year <= 2026:
                            game["release_year"] = year
                    except ValueError:
                        pass

                price_info = app_data.get("price_overview", {})
                if price_info:
                    game["base_price"] = price_info.get("initial", 0) / 100.0

                enriched += 1
            time.sleep(1.5)
        except Exception:
            time.sleep(2)
    return games

def generate_fallback_games(n=150):
    name_parts = {
        "Action": (
            ["Dark", "Shadow", "Steel", "Iron", "Blood", "War", "Fury", "Storm", "Blade", "Fire"],
            ["Souls", "Rising", "Warfare", "Force", "Strike", "Hunter", "Legends", "Assault", "Core", "Reign"],
        ),
        "RPG": (
            ["Elder", "Dragon", "Crystal", "Mystic", "Eternal", "Sacred", "Divine", "Ancient", "Lost", "Arcane"],
            ["Scrolls", "Quest", "Legacy", "Realms", "Chronicles", "Saga", "Odyssey", "Kingdoms", "Dawn", "Fate"],
        ),
        "Strategy": (
            ["Empire", "Civilization", "Total", "Age", "Command", "Rise", "Dawn", "Grand", "Supreme", "Iron"],
            ["Wars", "Tactics", "Conquest", "Nations", "Defense", "Protocol", "Dominion", "Order", "Alliance", "Front"],
        ),
        "Indie": (
            ["Hollow", "Celeste", "Pixel", "Neon", "Tiny", "Paper", "Stardew", "Dream", "Night", "Hyper"],
            ["Knight", "Dreams", "World", "Light", "Garden", "Trail", "Drift", "Echo", "Haven", "Loop"],
        ),
        "Adventure": (
            ["Uncharted", "Lost", "Hidden", "Secret", "Tomb", "Broken", "Forgotten", "Wild", "Strange", "Monkey"],
            ["Island", "Tales", "Passage", "Journey", "Mystery", "Expedition", "Roads", "Shores", "Keys", "Ages"],
        ),
        "Simulation": (
            ["City", "Farm", "Flight", "Planet", "Train", "Euro", "Space", "Power", "Truck", "Bus"],
            ["Builder", "Simulator", "Tycoon", "Manager", "Express", "Station", "Craft", "Control", "Life", "Works"],
        ),
        "Sports": (
            ["Pro", "Super", "Ultimate", "Championship", "World", "League", "Grand", "Premier", "AllStar", "MVP"],
            ["Soccer", "Basketball", "Tennis", "Golf", "Boxing", "Wrestling", "Racing", "Cricket", "Hockey", "League"],
        ),
        "Racing": (
            ["Speed", "Turbo", "Nitro", "Fast", "Drift", "Road", "Track", "Asphalt", "Grid", "Burn"],
            ["Racer", "Rush", "Circuit", "Fury", "Legends", "Championship", "Masters", "Rally", "Rivals", "Pursuit"],
        ),
        "Puzzle": (
            ["Portal", "Tetris", "Brain", "Logic", "Mind", "Color", "Shape", "Hex", "Block", "Cube"],
            ["Quest", "Twist", "Maze", "Flow", "Shift", "Spark", "Link", "Pop", "Dash", "Grid"],
        ),
        "Casual": (
            ["Happy", "Cozy", "Sweet", "Garden", "Kitchen", "Pet", "Bubble", "Candy", "Match", "Zen"],
            ["Life", "Tales", "Party", "Time", "World", "Village", "Island", "Home", "Cafe", "Day"],
        ),
    }

    genre_list = list(GENRES.keys())
    genre_weights = [0.20, 0.12, 0.10, 0.18, 0.10, 0.08, 0.06, 0.05, 0.06, 0.05]

    used_names = set()
    games = []

    for i in range(n):
        genre = np.random.choice(genre_list, p=genre_weights)
        price_min, price_max = GENRES[genre]

        parts = name_parts[genre]
        while True:
            name = f"{np.random.choice(parts[0])} {np.random.choice(parts[1])}"
            if name not in used_names:
                used_names.add(name)
                break

        base_price = round(np.random.uniform(price_min, price_max), 2)
        base_price = round(int(base_price)) - 0.01 if base_price > 1 else base_price

        release_year = int(np.random.choice(
            range(2016, 2026),
            p=[0.03, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.15, 0.15, 0.13],
        ))

        games.append({
            "appid": 100000 + i * 10,
            "name": name,
            "genre": genre,
            "base_price": base_price,
            "release_year": release_year,
        })
    return games

def generate_price_history(games_df):
    records = []
    today = datetime(2026, 7, 1)

    for _, game in games_df.iterrows():
        appid = game["appid"]
        genre = game["genre"]
        base_price = game["base_price"]
        release_year = game["release_year"]
        sale_prob = GENRE_SALE_PROBABILITY.get(genre, 0.80)

        for year in range(release_year, 2027):
            age = year - release_year
            age_factor = min(1.0 + age * 0.15, 2.0)

            for month, day_start, day_end, sale_name, discount_range in SALE_CALENDAR:
                sale_date = datetime(year, month, day_start)
                if sale_date < datetime(release_year, 1, 1) or sale_date > today:
                    continue

                if np.random.random() > sale_prob:
                    continue

                min_disc, max_disc = discount_range
                discount = np.random.uniform(min_disc, max_disc) * age_factor
                discount = int(round(min(discount, 90) / 5) * 5)
                discount = max(discount, 5)

                sale_price = round(base_price * (1 - discount / 100), 2)

                records.append({
                    "appid": appid,
                    "date": sale_date.strftime("%Y-%m-%d"),
                    "price": sale_price,
                    "discount_pct": discount,
                    "base_price": base_price,
                    "sale_event": sale_name,
                })

            n_random = np.random.choice(
                [0, 1, 1, 2, 2, 3],
                p=[0.15, 0.25, 0.25, 0.15, 0.15, 0.05],
            )
            for _ in range(n_random):
                month = np.random.choice([1, 3, 4, 5, 8, 9])
                day = np.random.randint(1, 28)
                sale_date = datetime(year, month, day)

                if sale_date < datetime(release_year, 1, 1) or sale_date > today:
                    continue

                discount = np.random.uniform(10, 40) * age_factor
                discount = int(round(min(discount, 85) / 5) * 5)
                discount = max(discount, 5)

                sale_price = round(base_price * (1 - discount / 100), 2)

                records.append({
                    "appid": appid,
                    "date": sale_date.strftime("%Y-%m-%d"),
                    "price": sale_price,
                    "discount_pct": discount,
                    "base_price": base_price,
                    "sale_event": "Midweek/Weekend Deal",
                })

    return pd.DataFrame(records)

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    raw_games = fetch_steamspy_top_games()

    if raw_games:
        raw_games = enrich_from_steam(raw_games)
        games_df = pd.DataFrame(raw_games)
    else:
        raw_games = generate_fallback_games(150)
        games_df = pd.DataFrame(raw_games)

    history_df = generate_price_history(games_df)

    games_path = os.path.join(DATA_DIR, "games.csv")
    history_path = os.path.join(DATA_DIR, "price_history.csv")

    games_df.to_csv(games_path, index=False, encoding="utf-8-sig")
    history_df.to_csv(history_path, index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    main()
