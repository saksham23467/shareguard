"""ShareGuard synthetic screen generator (v0).
Renders synthetic screens with planted secrets and writes per-secret and
per-rendering ground truth. Text is rendered at 3x horizontal resolution and
filtered into RGB subpixels as an approximation of ClearType (NOT real Windows
DirectWrite output). All fonts used are SIL OFL. All data is synthetic."""
import argparse, csv, hashlib, json, os, random, string
import numpy as np
from PIL import Image, ImageDraw, ImageFont

G = "/usr/share/fonts/truetype/sand-box/google/"
FONTS = {  # name -> (path, licence)
    "CascadiaCode": (G + "Cascadia Code/CascadiaCode-VariableFont_wght.ttf", "OFL-1.1"),
    "JetBrainsMono": (G + "JetBrains Mono/JetBrainsMono-VariableFont_wght.ttf", "OFL-1.1"),
    "NotoSansMono": (G + "Noto Sans Mono/NotoSansMono-VariableFont_wdth,wght.ttf", "OFL-1.1"),
    "NotoSans": (G + "Noto Sans/NotoSans-VariableFont_wdth,wght.ttf", "OFL-1.1"),
}
MONO = ["CascadiaCode", "JetBrainsMono", "NotoSansMono"]
W, H = 1280, 720
THEMES = {"light": ((255, 255, 255), (30, 30, 30), (173, 214, 255)),
          "dark": ((30, 30, 30), (212, 212, 212), (38, 79, 120))}
SYNTAX = [(86, 156, 214), (206, 145, 120), (78, 201, 176), (197, 134, 192)]
SCALES = [1.0, 1.25, 1.5]

# ---------------- secret generators ----------------
FIRST = ["Aarav","Priya","Rohan","Ananya","Vikram","Sneha","Arjun","Kavya","Rahul","Meera",
 "Siddharth","Ishita","Karthik","Lakshmi","Venkatesh","Divya","Harpreet","Gurleen","Mohammed","Fatima",
 "Subramanian","Aishwarya","Nikhil","Pooja","Emma","Liam","Sofia","Noah","Chen","Yuki"]
LAST = ["Sharma","Iyer","Reddy","Nair","Bansal","Gupta","Patel","Singh","Chatterjee","Mukherjee",
 "Venkataraman","Krishnamurthy","Deshpande","Kulkarni","Rao","Menon","Khan","Das","Joshi","Smith",
 "Garcia","Müller","Wang","Tanaka"]
STREETS = ["MG Road","Linking Road","Anna Salai","Park Street","Residency Road","Sector 18","Banjara Hills Rd 12","Koramangala 5th Block"]
CITIES = ["Bengaluru 560034","Mumbai 400050","Chennai 600002","Kolkata 700016","Hyderabad 500034","Noida 201301","Pune 411004","Delhi 110017"]
_d = [[0,1,2,3,4,5,6,7,8,9],[1,2,3,4,0,6,7,8,9,5],[2,3,4,0,1,7,8,9,5,6],[3,4,0,1,2,8,9,5,6,7],[4,0,1,2,3,9,5,6,7,8],
      [5,9,8,7,6,0,4,3,2,1],[6,5,9,8,7,1,0,4,3,2],[7,6,5,9,8,2,1,0,4,3],[8,7,6,5,9,3,2,1,0,4],[9,8,7,6,5,4,3,2,1,0]]
_p = [[0,1,2,3,4,5,6,7,8,9],[1,5,7,6,2,8,3,0,9,4],[5,8,0,3,7,9,6,1,4,2],[8,9,1,6,0,4,3,5,2,7],[9,4,5,3,1,2,6,8,7,0],
      [4,2,8,6,5,7,3,9,0,1],[2,7,9,3,8,0,6,4,1,5],[7,0,4,6,9,1,3,2,5,8]]
_inv = [0,4,3,2,1,5,6,7,8,9]
def verhoeff_digit(num):
    c = 0
    for i, ch in enumerate(reversed(num)):
        c = _d[c][_p[(i + 1) % 8][int(ch)]]
    return str(_inv[c])
def luhn_digit(num):
    s = 0
    for i, ch in enumerate(reversed(num)):
        d = int(ch)
        if i % 2 == 0:
            d *= 2; d = d - 9 if d > 9 else d
        s += d
    return str((10 - s % 10) % 10)
AN = string.ascii_letters + string.digits
def rs(r, n, al=AN): return "".join(r.choice(al) for _ in range(n))
def g_aws(r): return "AKIA" + rs(r, 16, string.ascii_uppercase + string.digits)
def g_ghp(r): return "ghp_" + rs(r, 36)
def g_apikey(r): return "sk-" + rs(r, 48)
def g_jwt(r): return "eyJ" + rs(r, 30, AN + "-_") + "." + rs(r, 40, AN + "-_") + "." + rs(r, 22, AN + "-_")
def g_password(r): return rs(r, r.randint(10, 18), AN + "!@#$%^&*")
def g_aadhaar(r):
    b = str(r.randint(2, 9)) + rs(r, 10, string.digits); n = b + verhoeff_digit(b)
    return f"{n[:4]} {n[4:8]} {n[8:]}"
def g_pan(r): return rs(r, 3, string.ascii_uppercase) + r.choice("PCHFT") + rs(r, 1, string.ascii_uppercase) + rs(r, 4, string.digits) + rs(r, 1, string.ascii_uppercase)
def g_card(r):
    b = r.choice(["4", "51", "52", "53", "6521"]); b += rs(r, 15 - len(b), string.digits); n = b + luhn_digit(b)
    return " ".join(n[i:i+4] for i in range(0, 16, 4))
def g_phone(r): return f"+91 {r.choice('6789')}{rs(r,4,string.digits)} {rs(r,5,string.digits)}"
def g_name(r): return f"{r.choice(FIRST)} {r.choice(LAST)}"
def g_email(r): return f"{r.choice(FIRST).lower()}.{r.choice(LAST).lower()}{r.randint(1,99)}@example.com"
def g_address(r): return f"{r.randint(1,999)}, {r.choice(STREETS)}, {r.choice(CITIES)}"
def g_dob(r): return f"{r.randint(1,28):02d}/{r.randint(1,12):02d}/{r.randint(1965,2006)}"
GEN = {"aws_key": g_aws, "github_pat": g_ghp, "api_key": g_apikey, "jwt": g_jwt, "password": g_password,
       "aadhaar": g_aadhaar, "pan": g_pan, "card": g_card, "phone": g_phone, "name": g_name,
       "email": g_email, "address": g_address, "dob": g_dob}
DECOYS = ["import os", "def main():", "return None", "for i in range(10):", "print('hello world')",
          "Build succeeded in 3.2s", "git status", "On branch main", "npm run dev", "Meeting at 4 pm tomorrow",
          "Thanks, will check and revert", "SELECT * FROM users LIMIT 10;", "commit 3f9a2c1", "version: 2.1.0",
          "timeout = 30", "region = ap-south-1", "Order #45821 shipped", "See attached slides"]

# ---------------- templates ----------------
# Each template yields list of (line_text, [(secret_type, secret_text, char_start)]) .
def t_terminal(r):
    lines = ["$ cat .env"]
    for key, t in [("AWS_ACCESS_KEY_ID", "aws_key"), ("GITHUB_TOKEN", "github_pat"), ("OPENAI_API_KEY", "api_key"),
                   ("DB_PASSWORD", "password"), ("SESSION_JWT", "jwt")]:
        if r.random() < 0.8:
            s = GEN[t](r); pre = f"{key}="; lines.append((pre + s, [(t, s, len(pre))]))
        lines.append(r.choice(DECOYS))
    lines.append("$ " + r.choice(DECOYS))
    return lines
def t_code(r):
    lines = ["# config.py", "import os", ""]
    for var, t in [("API_KEY", "api_key"), ("AWS_KEY", "aws_key"), ("password", "password"), ("token", "github_pat")]:
        if r.random() < 0.8:
            s = GEN[t](r); pre = f'{var} = "'; lines.append((pre + s + '"', [(t, s, len(pre))]))
        lines.append(r.choice(DECOYS))
    return lines
def t_form(r):
    lines = ["Customer KYC - Review"]
    for lab, t in [("Name:", "name"), ("DOB:", "dob"), ("Aadhaar No:", "aadhaar"), ("PAN:", "pan"),
                   ("Card Number:", "card"), ("Phone:", "phone"), ("Email:", "email"), ("Address:", "address")]:
        if r.random() < 0.85:
            s = GEN[t](r); pre = f"{lab} "; lines.append((pre + s, [(t, s, len(pre))]))
    lines.append("Status: Pending verification")
    return lines
def t_chat(r):
    lines = [f"Chat - #{r.choice(['support','team','billing'])}"]
    for _ in range(r.randint(4, 7)):
        who = g_name(r).split()[0] + ": "
        t = r.choice(["phone", "email", "address", "card", "password", "aadhaar", "name", None, None])
        if t is None:
            lines.append(who + r.choice(DECOYS)); continue
        s = GEN[t](r)
        pre = who + r.choice({"phone": ["call me on ", "my number is "], "email": ["mail it to ", "send to "],
            "address": ["ship to ", "I stay at "], "card": ["use card ", "charge "], "password": ["temp pwd is ", "login with "],
            "aadhaar": ["aadhaar is ", "uid "], "name": ["please add ", "cc "]}[t])
        lines.append((pre + s + r.choice(["", " thanks", " pls"]), [(t, s, len(pre))]))
    return lines
TEMPLATES = {"terminal": t_terminal, "code": t_code, "form": t_form, "chat": t_chat}

# ---------------- rendering ----------------
def lcd_filter(mask3):
    """mask3: HxW*3 float coverage. 5-tap FIR (like FreeType's default LCD filter), then pack to HxWx3."""
    k = np.array([0x08, 0x4D, 0x56, 0x4D, 0x08], np.float32); k /= k.sum()
    p = np.pad(mask3, ((0, 0), (2, 2)), mode="edge")
    f = sum(k[i] * p[:, i:i + mask3.shape[1]] for i in range(5))
    return f.reshape(mask3.shape[0], -1, 3)

def render(spec, scroll_px=0, highlight=None):
    """Returns RGB image and per-line pixel boxes (x0,y0,x1,y1) in screen coords and char-level x offsets."""
    bg, fg, hl = THEMES[spec["theme"]]
    size = int(round(spec["font_px"] * spec["scale"]))
    font = ImageFont.truetype(FONTS[spec["font"]][0], size)
    lh = int(size * 1.5); x0 = int(24 * spec["scale"]); y = int(20 * spec["scale"]) - scroll_px
    # coverage layers at 3x horizontal for fg colours
    boxes = []; draws = []
    for i, ln in enumerate(spec["lines"]):
        text = ln if isinstance(ln, str) else ln[0]
        draws.append((text, x0, y, i)); boxes.append((x0, y, x0 + int(font.getlength(text)) + 2, y + lh)); y += lh
    img = np.zeros((H, W, 3), np.float32); img[:] = bg
    for (text, x, yy, i) in draws:
        if highlight is not None and i in highlight:
            bx = boxes[i]; img[max(0, bx[1]):max(0, min(H, bx[3])), max(0, bx[0]):min(W, bx[2])] = hl
    colour_layers = {}
    for (text, x, yy, i) in draws:
        if yy + lh < 0 or yy > H: continue
        col = fg
        if spec["syntax"] and not isinstance(spec["lines"][i], str): col = SYNTAX[i % len(SYNTAX)]
        m = colour_layers.setdefault(col, Image.new("L", (W * 3, H), 0))
        d = ImageDraw.Draw(m)
        # draw at 3x horizontal by drawing with a 3x-wide font transform
        f3 = ImageFont.truetype(FONTS[spec["font"]][0], size)
        tmp = Image.new("L", (int(f3.getlength(text)) * 3 + 12, lh), 0)
        big = Image.new("L", (int(f3.getlength(text)) + 4, lh), 0)
        ImageDraw.Draw(big).text((0, (lh - size) // 2), text, font=f3, fill=255)
        # horizontal 3x supersample of the glyph image: render at 3x size, then squash vertically
        f_hi = ImageFont.truetype(FONTS[spec["font"]][0], size * 3)
        hi = Image.new("L", (int(f_hi.getlength(text)) + 12, lh * 3), 0)
        ImageDraw.Draw(hi).text((0, (lh - size) // 2 * 3), text, font=f_hi, fill=255)
        hi = hi.resize((hi.width, lh), Image.BOX)
        m.paste(hi, (x * 3, yy), hi)
    for col, m in colour_layers.items():
        cov = lcd_filter(np.asarray(m, np.float32) / 255.0)
        img = img * (1 - cov) + np.array(col, np.float32) * cov
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)), boxes, font

def crop_hash(im, box):
    x0, y0, x1, y1 = [int(v) for v in box]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if x1 <= x0 or y1 <= y0: return ""
    return hashlib.blake2b(im.crop((x0, y0, x1, y1)).tobytes(), digest_size=16).hexdigest()

def split_of(template_id):
    h = int(hashlib.md5(template_id.encode()).hexdigest(), 16) % 100
    return "train" if h < 50 else ("calib" if h < 75 else "test")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--screens", type=int, default=400)
    ap.add_argument("--out", default="out"); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); r = random.Random(a.seed)
    os.makedirs(f"{a.out}/frames", exist_ok=True)
    sec_f = open(f"{a.out}/secrets.csv", "w", newline=""); ren_f = open(f"{a.out}/renderings.csv", "w", newline="")
    sw = csv.writer(sec_f); rw = csv.writer(ren_f)
    sw.writerow(["secret_id", "screen_id", "template_id", "split", "secret_type", "text", "font", "font_licence",
                 "font_px", "theme", "scale", "syntax", "line_index"])
    rw.writerow(["rendering_id", "secret_id", "screen_id", "frame_file", "first_frame", "last_frame",
                 "x0", "y0", "x1", "y1", "line_x0", "line_y0", "line_x1", "line_y1", "line_pixel_hash", "event", "visible_fraction"])
    n_sec = n_ren = 0
    for s in range(a.screens):
        kind = r.choice(list(TEMPLATES)); tseed = r.randint(0, 59)  # 60 layout seeds per kind -> 240 templates
        template_id = f"{kind}_{tseed:02d}"; tr = random.Random(hash(template_id) & 0xffffffff)
        # template fixes layout skeleton; per-screen randomness fills values
        lines = TEMPLATES[kind](random.Random(tseed * 7919 + s))
        spec = {"lines": lines, "theme": r.choice(list(THEMES)), "scale": r.choice(SCALES),
                "font": r.choice(MONO) if kind != "chat" else r.choice(MONO + ["NotoSans"]),
                "font_px": r.choice([11, 12, 13, 14, 16]), "syntax": kind == "code" and r.random() < 0.7}
        screen_id = f"s{s:05d}"; split = split_of(template_id)
        secrets = []
        for li, ln in enumerate(lines):
            if isinstance(ln, str): continue
            for (t, txt, cstart) in ln[1]:
                sid = f"{screen_id}_k{len(secrets)}"; secrets.append((sid, li, txt, cstart))
                sw.writerow([sid, screen_id, template_id, split, t, txt, spec["font"], FONTS[spec["font"]][1],
                             spec["font_px"], spec["theme"], spec["scale"], int(spec["syntax"]), li]); n_sec += 1
        # replay: initial, highlight a random secret line, partial scroll, resize (font +1px)
        events = [("initial", 0, None, spec)]
        sl = [x[1] for x in secrets]
        if sl and r.random() < 0.6: events.append(("highlight", 0, {r.choice(sl)}, spec))
        if r.random() < 0.6: events.append(("scroll", r.randint(3, 40), None, spec))
        if r.random() < 0.4: events.append(("resize", 0, None, dict(spec, font_px=spec["font_px"] + 1)))
        frame = 0
        for ei, (ev, scroll, hlset, sp) in enumerate(events):
            dur = r.randint(20, 200); im, boxes, font = render(sp, scroll, hlset)
            ff = f"frames/{screen_id}_e{ei}_{ev}.png"; im.save(f"{a.out}/{ff}", optimize=False, compress_level=1)
            for (sid, li, txt, cstart) in secrets:
                ln = sp["lines"][li][0]; bx = boxes[li]
                sx0 = bx[0] + font.getlength(ln[:cstart]); sx1 = bx[0] + font.getlength(ln[:cstart + len(txt)])
                vis_top, vis_bot = max(0, bx[1]), min(H, bx[3]); vf = max(0, vis_bot - vis_top) / (bx[3] - bx[1])
                if vf <= 0: continue
                rw.writerow([f"{sid}_r{ei}", sid, screen_id, ff, frame, frame + dur - 1, int(sx0), bx[1], int(np.ceil(sx1)), bx[3],
                             *bx, crop_hash(im, bx), ev, round(vf, 3)]); n_ren += 1
            frame += dur
        if s % 50 == 0: print(s, n_sec, n_ren, flush=True)
    json.dump({"screens": a.screens, "secrets": n_sec, "renderings": n_ren, "seed": a.seed,
               "renderer": "PIL/FreeType 3x-horizontal supersample + 5-tap LCD filter (ClearType approximation, not DirectWrite)",
               "fonts": {k: v[1] for k, v in FONTS.items()}}, open(f"{a.out}/manifest.json", "w"), indent=1)
    print("done", n_sec, n_ren)
if __name__ == "__main__": main()
