from PIL import Image, ImageFont, ImageDraw
import os
import csv
from typography_text_generator import generate_typography

def get_project_root():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(current_dir)

# Focused small-viewport typography renderer
# Requirements implemented:
# - Remove unrelated SD/data-processing code (file simplified)
# - Break lines at commas and after every 4 words
# - Render into a small square image (default 512x512)
# - Use a small font and center the rendered block exactly in the image

DEFAULT_SIZE = 512
DEFAULT_FONT_PATH = "/Library/Fonts/Arial Unicode.ttf"
DEFAULT_FONT_SIZE = 28
DEFAULT_MARGIN = 16
DEFAULT_LINE_SPACING = 6


def custom_split(text: str, words_per_line: int = 4):
    """
    Split text by commas first (commas force a break and are removed),
    then split remaining segments into chunks of `words_per_line`.
    """
    MAX_CHARS = 25
    segments = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        words = part.split()
        cur_words = []
        cur_chars = 0
        for w in words:
            w_len = len(w)
            if not cur_words:
                # start new line with this word (even if it exceeds MAX_CHARS)
                cur_words.append(w)
                cur_chars = w_len
            else:
                # check words_per_line constraint first
                if len(cur_words) >= words_per_line:
                    segments.append(" ".join(cur_words))
                    cur_words = [w]
                    cur_chars = w_len
                else:
                    # check char limit if we add this word (with a space)
                    if cur_chars + 1 + w_len <= MAX_CHARS:
                        cur_words.append(w)
                        cur_chars += 1 + w_len
                    else:
                        # push current line, start new line with this word
                        segments.append(" ".join(cur_words))
                        cur_words = [w]
                        cur_chars = w_len
        if cur_words:
            segments.append(" ".join(cur_words))
    return segments if segments else [""]


def render_small_centered(text: str, out_path: str, size: int = DEFAULT_SIZE, font_path: str = DEFAULT_FONT_PATH, font_size: int = DEFAULT_FONT_SIZE, margin: int = DEFAULT_MARGIN, line_spacing: int = DEFAULT_LINE_SPACING):
    """
    Render text into a small square image centered both horizontally and vertically.
    Uses precise measurements (textbbox) to center the block.
    """
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        font = ImageFont.load_default()

    img = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    lines = custom_split(text, words_per_line=4)

    # Measure each line's bbox to compute width and height
    bboxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
    widths = [bbox[2] - bbox[0] for bbox in bboxes]
    heights = [bbox[3] - bbox[1] for bbox in bboxes]

    total_height = sum(heights) + line_spacing * (len(lines) - 1)
    start_y = (size - total_height) // 2

    y = start_y
    for i, line in enumerate(lines):
        w = widths[i]
        h = heights[i]
        x = (size - w) // 2
        draw.text((x, y), line, fill=(0, 0, 0), font=font)
        y += h + line_spacing

    # ensure output directory exists
    out_dir = os.path.dirname(out_path) or get_project_root()
    os.makedirs(out_dir, exist_ok=True)
    img.save(out_path)
    return out_path


def process_txt_file(txt_path: str, output_root: str = "typo_images", size: int = DEFAULT_SIZE, font_size: int = DEFAULT_FONT_SIZE):
    """
    Read a .txt file line by line, convert each line to a typography image using
    `generate_typography`, and save into output_root/<basename_without_ext>/<index>.png.
    Resume support: existing files are skipped.
    """
    if not os.path.exists(txt_path):
        raise FileNotFoundError(txt_path)

    basename = os.path.splitext(os.path.basename(txt_path))[0]
    out_dir = os.path.join(output_root, basename)
    os.makedirs(out_dir, exist_ok=True)

    with open(txt_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    for idx, line in enumerate(lines, start=1):
        out_file = os.path.join(out_dir, f"{idx}.png")
        if os.path.exists(out_file):
            print(f"[skip] exists: {out_file}")
            continue

        raw = line.strip()
        if not raw:
            print(f"[skip] empty line #{idx}")
            continue

        try:
            processed = generate_typography(raw)
            with open('typo_text.txt', 'a') as file:
                file.write(processed)
                file.write('\n')
        except Exception as e:
            print(f"[error] generate_typography failed for line #{idx}: {e}")
            # stop processing to allow the user to inspect/repair
            break

        if not processed:
            processed = raw

        try:
            render_small_centered(processed, out_file, size=size, font_size=font_size)
            print(f"[saved] {out_file}")
        except Exception as e:
            print(f"[error] rendering failed for line #{idx}: {e}")
            break


def process_csv_file(csv_path: str, output_root: str = "typo_images", size: int = DEFAULT_SIZE, font_size: int = DEFAULT_FONT_SIZE):
    """
    Read a .csv file, extract 'goal' column values, convert each to a typography image using
    `generate_typography`, and save into output_root/<basename_without_ext>/<index>.png.
    Resume support: existing files are skipped.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    basename = os.path.splitext(os.path.basename(csv_path))[0]
    out_dir = os.path.join(output_root, basename)
    os.makedirs(out_dir, exist_ok=True)

    goals = []
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'goal' in row and row['goal'].strip():
                    goals.append(row['goal'].strip())
    except Exception as e:
        print(f"[error] Failed to read CSV: {e}")
        return

    print(f"[info] Processing {len(goals)} goals from {csv_path}")

    for idx, goal in enumerate(goals, start=1):
        out_file = os.path.join(out_dir, f"{idx}.png")
        if os.path.exists(out_file):
            print(f"[skip] exists: {out_file}")
            continue

        try:
            processed = generate_typography(goal)
            with open('typo_text.txt', 'a') as file:
                file.write(processed)
                file.write('\n')
        except Exception as e:
            print(f"[error] generate_typography failed for goal #{idx}: {e}")
            # stop processing to allow the user to inspect/repair
            break

        if not processed:
            processed = goal

        try:
            render_small_centered(processed, out_file, size=size, font_size=font_size)
            print(f"[saved] {out_file}")
        except Exception as e:
            print(f"[error] rendering failed for goal #{idx}: {e}")
            break

def process_harmbench(csv_path: str, output_root: str = "typo_images", size: int = DEFAULT_SIZE, font_size: int = DEFAULT_FONT_SIZE):
    """
    Read a .csv file, extract 'Behavior' column values, convert each to a typography image using
    `generate_typography`, and save into output_root/<basename_without_ext>/<index>.png.
    Resume support: existing files are skipped.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    basename = os.path.splitext(os.path.basename(csv_path))[0]
    out_dir = os.path.join(output_root, basename)
    os.makedirs(out_dir, exist_ok=True)

    goals = []
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'Behavior' in row and row['Behavior'].strip():
                    goals.append(row['Behavior'].strip())
    except Exception as e:
        print(f"[error] Failed to read CSV: {e}")
        return

    print(f"[info] Processing {len(goals)} goals from {csv_path}")

    for idx, goal in enumerate(goals, start=1):
        out_file = os.path.join(out_dir, f"{idx}.png")
        if os.path.exists(out_file):
            print(f"[skip] exists: {out_file}")
            continue

        try:
            processed = generate_typography(goal)
            with open('typo_text.txt', 'a') as file:
                file.write(processed)
                file.write('\n')
        except Exception as e:
            print(f"[error] generate_typography failed for goal #{idx}: {e}")
            # stop processing to allow the user to inspect/repair
            break

        if not processed:
            processed = goal

        try:
            render_small_centered(processed, out_file, size=size, font_size=font_size)
            print(f"[saved] {out_file}")
        except Exception as e:
            print(f"[error] rendering failed for goal #{idx}: {e}")
            break

if __name__ == "__main__":
    # # simple CLI for quick single-text rendering
    # user_text = "A person in a garage workshop carefully measures powders into glass beakers, wearing thick gloves and a mask, with a chemistry book open nearby."
    # out_file = os.path.join(get_project_root(), "output_typography_small.jpg")
    # render_small_centered(user_text, out_file, size=512, font_size=28, margin=16)
    # print(f"Saved: {out_file}")
    # txt_file_path = "gpt4_generated_questions/03-Malware_Generation.txt"
    # process_txt_file(txt_file_path)

    # process txt files
    txt_filename_list = [
        # "01-Illegal_Activity.txt",
        # "02-HateSpeech.txt",
        # "03-Malware_Generation.txt",
        # "04-Physical_Harm.txt",
        # "05-EconomicHarm.txt",
        # "06-Fraud.txt",
        # "07-Sex.txt",
        # "08-Political_Lobbying.txt",
        # "09-Privacy_Violence.txt",
        # "10-Legal_Opinion.txt",
        # "11-Financial_Advice.txt",
        # "12-Health_Consultation.txt",
        # "13-Gov_Decision.txt"
    ]

    # target_folder = "gpt4_generated_questions"

    # for filename in txt_filename_list:
    #     input_txt_path = f"{target_folder}/{filename}" 
    #     process_txt_file(input_txt_path)

    csv_input_path = "harmbench_behaviors.csv"
    process_harmbench(csv_input_path)
