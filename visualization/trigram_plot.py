import matplotlib.pyplot as plt
import pandas as pd
from textwrap3 import fill

# Load data
imdb_df = pd.read_csv("../trigrams_output/all_imdb_review_trigrams.csv")
chatgpt_df = pd.read_csv("../trigrams_output/aireviews_chatgpt_trigrams.csv")
deepseek_df = pd.read_csv("../trigrams_output/aireviews_deepseek_trigrams.csv")
gemini_df = pd.read_csv("../trigrams_output/aireviews_gemini_trigrams.csv")
gemini_ctx_df = pd.read_csv("../trigrams_output/aireviews_gemini_context_variation_trigrams.csv")


# Helper to extract top-1 trigram for question2
def top_trigram(df, label):
    top = df[df["Question"] == "question2"]
    top = top.sort_values("Count", ascending=False).drop_duplicates(subset="Movie")
    return top[["Movie", "Trigram", "Count"]].rename(columns={
        "Movie": "Title",
        "Trigram": f"{label} Trigram",
        "Count": f"{label} Count"
    })

# Prepare each source
imdb_top = imdb_df.sort_values("Count", ascending=False).drop_duplicates(subset="Title")
imdb_top = imdb_top[["Title", "Trigram", "Count"]].rename(columns={"Trigram": "IMDb Trigram", "Count": "IMDb Count"})
chatgpt_top = top_trigram(chatgpt_df, "ChatGPT")
deepseek_top = top_trigram(deepseek_df, "DeepSeek")
gemini_top = top_trigram(gemini_df, "Gemini")
gemini_ctx_top = top_trigram(gemini_ctx_df, "GeminiCtx")

# Merge everything
df = imdb_top.merge(chatgpt_top, on="Title").merge(deepseek_top, on="Title")
df = df.merge(gemini_top, on="Title").merge(gemini_ctx_top, on="Title")

# Wrap movie titles to fit better in cells
def wrap(title, width=25):
    manual_wraps = {
        "Crouching Tiger, Hidden Dragon": "Crouching Tiger,\nHidden Dragon",
        "Brokeback Mountain": "Brokeback\nMountain",
        "The Shawshank Redemption": "The Shawshank\nRedemption"
    }
    return manual_wraps.get(title, fill(title, width=width))

def wrap_trigram(text, width=18):
    return fill(str(text), width=width)

# Headers
main_headers = ["Movie", "IMDb", "", "ChatGPT", "", "DeepSeek", "", "Gemini", "", "Gemini (detailed)", ""]
sub_headers = [
    "", "Trigram", "Freq",
    "Trigram", "Freq",
    "Trigram", "Freq",
    "Trigram", "Freq",
    "Trigram", "Freq"
]
table_data = [main_headers, sub_headers]

# Data rows
for _, row in df.iterrows():
    table_data.append([
        wrap(row["Title"]),
        wrap_trigram(row["IMDb Trigram"]), row["IMDb Count"],
        wrap_trigram(row["ChatGPT Trigram"]), row["ChatGPT Count"],
        wrap_trigram(row["DeepSeek Trigram"]), row["DeepSeek Count"],
        wrap_trigram(row["Gemini Trigram"]), row["Gemini Count"],
        wrap_trigram(row["GeminiCtx Trigram"]), row["GeminiCtx Count"]
    ])



# Check for rows that aren't length 11
# for i, row in enumerate(table_data):
#     if len(row) != 11:
#         print(f"Row {i} length = {len(row)}: {row}")
#
# Plot & export
fig, ax = plt.subplots(figsize=(26, 0.9 * len(df) + 3))
ax.axis('off')
table = ax.table(cellText=table_data, cellLoc='center', loc='center', edges='closed')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.0, 1.5)

# Style headers
for (r, c), cell in table.get_celld().items():
    if r in (0, 1):
        cell.set_text_props(weight='bold', ha='center', va='center')
        # merge cell
        if r == 0 and c != 0:
            if c % 2 == 1:
                cell.visible_edges = 'TBL'
            else:
                cell.visible_edges = 'TRB'
                cell.get_text().set_text("")


plt.tight_layout()
plt.savefig("top1_trigram_table.png", dpi=300, bbox_inches='tight')
plt.show()

