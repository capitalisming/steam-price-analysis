import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

COLORS = {
    "bg": "#1b2838",
    "card": "#2a475e",
    "accent": "#66c0f4",
    "green": "#4caf50",
    "orange": "#f57c00",
    "text": "#c7d5e0",
    "grid": "#3a5a7a",
}

GENRE_COLORS = {
    "Action": "#e53935",
    "RPG": "#8e24aa",
    "Strategy": "#1e88e5",
    "Indie": "#43a047",
    "Adventure": "#fb8c00",
    "Simulation": "#00897b",
    "Sports": "#fdd835",
    "Racing": "#f4511e",
    "Puzzle": "#5e35b1",
    "Casual": "#ec407a",
}

MONTH_NAMES_RU = [
    "Янв", "Фев", "Мар", "Апр", "Май", "Июн",
    "Июл", "Авг", "Сен", "Окт", "Ноя", "Дек",
]

def setup_style():
    plt.rcParams.update({
        "figure.facecolor": COLORS["bg"],
        "axes.facecolor": COLORS["card"],
        "axes.edgecolor": COLORS["grid"],
        "axes.labelcolor": COLORS["text"],
        "text.color": COLORS["text"],
        "xtick.color": COLORS["text"],
        "ytick.color": COLORS["text"],
        "grid.color": COLORS["grid"],
        "grid.alpha": 0.3,
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
    })

def load_data():
    games = pd.read_csv(os.path.join(DATA_DIR, "games.csv"))
    history = pd.read_csv(
        os.path.join(DATA_DIR, "price_history.csv"),
        parse_dates=["date"],
    )
    history = history.merge(
        games[["appid", "genre", "release_year", "name"]],
        on="appid",
        how="left",
    )
    return games, history

def plot_discount_frequency(games, history):
    fig, ax = plt.subplots(figsize=(12, 6))

    per_game = history.groupby("appid").size().reset_index(name="n_discounts")
    per_game = per_game.merge(games[["appid", "genre", "release_year"]], on="appid")
    per_game["years_active"] = 2026 - per_game["release_year"]
    per_game["per_year"] = (
        per_game["n_discounts"] / per_game["years_active"].clip(lower=1)
    )

    genre_freq = per_game.groupby("genre")["per_year"].mean().sort_values(ascending=True)
    colors = [GENRE_COLORS.get(g, COLORS["accent"]) for g in genre_freq.index]

    bars = ax.barh(genre_freq.index, genre_freq.values,
                   color=colors, edgecolor="none", height=0.6)

    for bar, val in zip(bars, genre_freq.values):
        ax.text(val + 0.1, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}", va="center", fontsize=10, color=COLORS["text"])

    ax.set_xlabel("Среднее кол-во скидок в год")
    ax.set_title("Частота скидок по жанрам", pad=15)
    ax.grid(axis="x", alpha=0.2)
    ax.set_xlim(0, genre_freq.max() * 1.15)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "1_discount_frequency.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

def plot_discount_depth(games, history):
    fig, ax = plt.subplots(figsize=(12, 6))

    genre_order = (
        history.groupby("genre")["discount_pct"]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )
    data_by_genre = [
        history[history["genre"] == g]["discount_pct"].values
        for g in genre_order
    ]

    bp = ax.boxplot(
        data_by_genre,
        tick_labels=genre_order,
        patch_artist=True,
        widths=0.6,
        showfliers=False,
        showmeans=True,
        meanprops=dict(marker="D", markerfacecolor=COLORS["accent"], markersize=6),
        medianprops=dict(color="white", linewidth=2),
    )

    for patch, genre in zip(bp["boxes"], genre_order):
        patch.set_facecolor(GENRE_COLORS.get(genre, COLORS["accent"]))
        patch.set_alpha(0.8)
    for w in bp["whiskers"]:
        w.set_color(COLORS["text"])
        w.set_alpha(0.5)
    for c in bp["caps"]:
        c.set_color(COLORS["text"])
        c.set_alpha(0.5)

    medians = [np.median(d) for d in data_by_genre]
    for i, med in enumerate(medians):
        ax.text(i + 1, med + 2, f"{med:.0f}%", ha="center", fontsize=9,
                color="white", fontweight="bold")

    ax.set_ylabel("Глубина скидки (%)")
    ax.set_title("Распределение глубины скидок по жанрам", pad=15)
    ax.grid(axis="y", alpha=0.2)
    plt.xticks(rotation=30, ha="right")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "2_discount_depth.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

def plot_seasonality(games, history):
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [1, 1]}
    )

    history_m = history.copy()
    history_m["month"] = history_m["date"].dt.month
    months = range(1, 13)

    monthly_disc = history_m.groupby("month")["discount_pct"].mean()

    bar_colors = []
    for m in months:
        val = monthly_disc.get(m, 0)
        if val >= 40:
            bar_colors.append("#e53935")
        elif val >= 30:
            bar_colors.append("#fb8c00")
        elif val >= 20:
            bar_colors.append("#66c0f4")
        else:
            bar_colors.append("#4a6a8a")

    bars1 = ax1.bar(
        list(months),
        [monthly_disc.get(m, 0) for m in months],
        color=bar_colors, edgecolor="none", width=0.7,
    )
    for bar, m in zip(bars1, months):
        val = monthly_disc.get(m, 0)
        if val > 0:
            ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.8,
                     f"{val:.0f}%", ha="center", fontsize=9, color=COLORS["text"])

    ax1.set_xticks(list(months))
    ax1.set_xticklabels(MONTH_NAMES_RU)
    ax1.set_ylabel("Средний дисконт (%)")
    ax1.set_title("Сезонность распродаж Steam", pad=15)
    ax1.grid(axis="y", alpha=0.2)

    annotations = {
        6: "Summer\nSale", 7: "Summer\nSale", 12: "Winter\nSale",
        11: "Autumn\nSale", 10: "Halloween", 2: "Lunar\nNY",
    }
    for m, label in annotations.items():
        val = monthly_disc.get(m, 0)
        if val > 0:
            ax1.annotate(
                label, xy=(m, val), xytext=(m, val + 6),
                ha="center", fontsize=7, color=COLORS["accent"],
                fontweight="bold", alpha=0.8,
            )

    monthly_cnt = history_m.groupby("month").size()

    ax2.bar(
        list(months),
        [monthly_cnt.get(m, 0) for m in months],
        color=COLORS["accent"], alpha=0.7, edgecolor="none", width=0.7,
    )
    for m in months:
        val = monthly_cnt.get(m, 0)
        if val > 0:
            ax2.text(m, val + 5, str(val), ha="center", fontsize=8,
                     color=COLORS["text"])

    ax2.set_xticks(list(months))
    ax2.set_xticklabels(MONTH_NAMES_RU)
    ax2.set_ylabel("Кол-во скидочных событий")
    ax2.set_xlabel("Месяц")
    ax2.grid(axis="y", alpha=0.2)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "3_seasonality.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

def plot_price_anchor(games, history):
    fig, ax = plt.subplots(figsize=(12, 7))

    game_stats = history.groupby("appid").agg(
        avg_discount=("discount_pct", "mean"),
        max_discount=("discount_pct", "max"),
        base_price=("base_price", "first"),
    ).reset_index()
    game_stats = game_stats.merge(games[["appid", "genre"]], on="appid")

    for genre in game_stats["genre"].unique():
        mask = game_stats["genre"] == genre
        ax.scatter(
            game_stats.loc[mask, "base_price"],
            game_stats.loc[mask, "avg_discount"],
            c=GENRE_COLORS.get(genre, COLORS["accent"]),
            label=genre, alpha=0.6, s=40, edgecolors="none",
        )

    x = game_stats["base_price"].values
    y = game_stats["avg_discount"].values
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, p(x_line), "--", color="white", linewidth=2, alpha=0.8,
            label=f"Тренд (k={z[0]:.2f})")

    corr = np.corrcoef(x, y)[0, 1]
    ax.text(
        0.02, 0.98, f"Корреляция: r = {corr:.3f}",
        transform=ax.transAxes, fontsize=11, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4",
                  facecolor=COLORS["bg"], edgecolor=COLORS["accent"], alpha=0.9),
    )

    ax.set_xlabel("Базовая цена ($)")
    ax.set_ylabel("Средний дисконт (%)")
    ax.set_title("Ценовой якорь: зависимость скидки от базовой цены", pad=15)
    ax.legend(
        loc="lower right", fontsize=8, ncol=2,
        facecolor=COLORS["bg"], edgecolor=COLORS["grid"],
        labelcolor=COLORS["text"],
    )
    ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "4_price_anchor.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

def plot_price_degradation(games, history):
    fig, ax = plt.subplots(figsize=(12, 7))

    h = history.copy()
    h["game_age"] = h["date"].dt.year - h["release_year"]
    h = h[h["game_age"] >= 0]

    top_genres = h["genre"].value_counts().head(5).index.tolist()

    for genre in top_genres:
        gd = h[h["genre"] == genre]
        age_stats = (
            gd.groupby("game_age")["discount_pct"]
            .agg(["mean", "max"])
            .reset_index()
        )
        age_stats = age_stats[age_stats["game_age"] <= 9]

        color = GENRE_COLORS.get(genre, COLORS["accent"])
        ax.plot(
            age_stats["game_age"], age_stats["mean"],
            "-o", color=color, linewidth=2.5, markersize=6,
            label=f"{genre} (средний)", alpha=0.9,
        )
        ax.fill_between(
            age_stats["game_age"],
            age_stats["mean"] * 0.85,
            age_stats["max"],
            color=color, alpha=0.1,
        )

    ax.set_xlabel("Возраст игры (лет после релиза)")
    ax.set_ylabel("Дисконт (%)")
    ax.set_title("Деградация цен: как скидки растут с возрастом игры", pad=15)
    ax.legend(
        facecolor=COLORS["bg"], edgecolor=COLORS["grid"],
        labelcolor=COLORS["text"],
    )
    ax.grid(alpha=0.2)
    ax.set_xticks(range(0, 10))
    ax.set_xlim(-0.5, 9.5)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "5_price_degradation.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

def plot_optimal_window(games, history):
    fig, ax1 = plt.subplots(figsize=(12, 7))

    h = history.copy()
    h["month"] = h["date"].dt.month
    total_games = games["appid"].nunique()

    monthly_stats = []
    for m in range(1, 13):
        md = h[h["month"] == m]
        games_on_sale = md["appid"].nunique()
        avg_disc = md["discount_pct"].mean() if len(md) > 0 else 0
        n_years = md["date"].dt.year.nunique() if len(md) > 0 else 1
        prob = min((games_on_sale / n_years) / total_games * 100, 100)
        monthly_stats.append({
            "month": m,
            "avg_discount": avg_disc,
            "probability": prob,
        })

    sdf = pd.DataFrame(monthly_stats)

    bar_colors = []
    for _, row in sdf.iterrows():
        score = row["avg_discount"] * 0.6 + row["probability"] * 0.4
        if score >= 30:
            bar_colors.append("#4caf50")
        elif score >= 20:
            bar_colors.append("#66c0f4")
        else:
            bar_colors.append("#4a6a8a")

    ax1.bar(
        sdf["month"], sdf["avg_discount"],
        color=bar_colors, alpha=0.7, width=0.5,
        label="Средний дисконт (%)",
    )
    ax1.set_ylabel("Средний дисконт (%)", color=COLORS["text"])
    ax1.set_xlabel("Месяц")
    ax1.set_xticks(range(1, 13))
    ax1.set_xticklabels(MONTH_NAMES_RU)

    ax2 = ax1.twinx()
    ax2.plot(
        sdf["month"], sdf["probability"],
        "-s", color="#ff7043", linewidth=2.5, markersize=8,
        label="Вероятность скидки (%)", zorder=5,
    )
    ax2.set_ylabel("Вероятность скидки (%)", color="#ff7043")
    ax2.tick_params(axis="y", labelcolor="#ff7043")
    ax2.spines["right"].set_color("#ff7043")

    sdf["score"] = sdf["avg_discount"] * 0.6 + sdf["probability"] * 0.4
    best = sdf.loc[sdf["score"].idxmax()]

    ax1.annotate(
        f"★ Лучший месяц!\nСкидка: {best['avg_discount']:.0f}%",
        xy=(best["month"], best["avg_discount"]),
        xytext=(best["month"] + 1.5, best["avg_discount"] + 8),
        fontsize=10, fontweight="bold", color="#4caf50",
        arrowprops=dict(arrowstyle="->", color="#4caf50", lw=2),
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor=COLORS["bg"], edgecolor="#4caf50", alpha=0.9,
        ),
    )

    ax1.set_title("Оптимальное окно покупки: когда лучше всего покупать?", pad=15)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(
        h1 + h2, l1 + l2, loc="upper left",
        facecolor=COLORS["bg"], edgecolor=COLORS["grid"],
        labelcolor=COLORS["text"],
    )
    ax1.grid(axis="y", alpha=0.2)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "6_optimal_window.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    setup_style()
    games, history = load_data()

    plot_discount_frequency(games, history)
    plot_discount_depth(games, history)
    plot_seasonality(games, history)
    plot_price_anchor(games, history)
    plot_price_degradation(games, history)
    plot_optimal_window(games, history)

if __name__ == "__main__":
    main()
