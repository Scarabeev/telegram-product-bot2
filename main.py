import logging
from telegram import Update, InputFile
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import requests
from bs4 import BeautifulSoup
import io

BOT_TOKEN = "8155729751:AAFqopt5CqVuhpYdOriH4_7Gds0MBn6ugtk"

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model.to(device)

logging.basicConfig(level=logging.INFO)

def image_to_query(image: Image.Image):
    labels = ["зубная паста", "шампунь", "детское мыло", "ноутбук", "телевизор", "мобильный телефон"]
    inputs = clip_processor(text=labels, images=image, return_tensors="pt", padding=True).to(device)
    outputs = clip_model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    predicted_index = torch.argmax(probs).item()
    return labels[predicted_index]

def search_wb(query):
    try:
        url = f"https://search.wb.ru/exactmatch/ru/common/v4/search?query={query}&resultset=catalog&sort=popular&spp=0&curr=rub"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers)
        data = resp.json()['data']['products'][:5]
        return [{
            'site': 'Wildberries',
            'name': p['name'],
            'price': p['priceU'] / 100,
            'rating': p.get('reviewRating', 0)
        } for p in data]
    except: return []

def search_yandex_market(query):
    try:
        url = f"https://market.yandex.ru/search?text={query}"
        headers = {"User-Agent": "Mozilla/5.0"}
        soup = BeautifulSoup(requests.get(url, headers=headers).text, 'html.parser')
        items = soup.select('article[data-autotest-id="product-snippet"]')
        results = []
        for i, item in enumerate(items[:5]):
            name = item.select_one('h3').text.strip() if item.select_one('h3') else f"Товар {i+1}"
            price = item.select_one('[data-autotest-value]')
            price = int(''.join(filter(str.isdigit, price.text))) / 100 if price else 0
            results.append({'site': 'Я.Маркет', 'name': name, 'price': price, 'rating': 4.6})
        return results
    except: return []

def search_ozon(query):
    try:
        url = f"https://www.ozon.ru/search/?text={query}"
        headers = {"User-Agent": "Mozilla/5.0"}
        soup = BeautifulSoup(requests.get(url, headers=headers).text, 'html.parser')
        items = soup.select('div.tsBodyL')
        results = []
        for i, item in enumerate(items[:5]):
            name = item.select_one('a')
            price_tag = item.find_next('span', {'class': 'ui-uikit-body-xl'})
            name = name.text.strip() if name else f"Ozon товар {i+1}"
            price = int(''.join(filter(str.isdigit, price_tag.text))) / 100 if price_tag else 0
            results.append({'site': 'Ozon', 'name': name, 'price': price, 'rating': 4.5})
        return results
    except: return []

def analyze_and_format(products):
    if not products:
        return "❗️Ничего не найдено."
    products.sort(key=lambda x: x['price'])
    best = products[0]
    cheaper = products[1:4]
    more_expensive = products[-2:] if len(products) > 3 else []
    msg = f"✅ *Лучшее предложение:*\n{best['name']} — {best['price']}₽ ({best['site']})\n\n"
    if cheaper:
        msg += "💰 *Дешевле:*\n" + "\n".join([f"{p['name']} — {p['price']}₽ ({p['site']})" for p in cheaper]) + "\n\n"
    if more_expensive:
        msg += "💎 *Дороже:*\n" + "\n".join([f"{p['name']} — {p['price']}₽ ({p['site']})" for p in more_expensive])
    return msg

def search_all_sources(query):
    wb = search_wb(query)
    ozon = search_ozon(query)
    ym = search_yandex_market(query)
    all_results = wb + ozon + ym
    return analyze_and_format(all_results)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Привет! Я бот-помощник по поиску выгодных товаров. Отправь мне фото товара или его название.")

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.message.text.strip()
    await update.message.reply_text("Ищу лучшие предложения...")
    result = search_all_sources(query)
    await update.message.reply_text(result, parse_mode="Markdown")

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    photo_file = await update.message.photo[-1].get_file()
    image_bytes = await photo_file.download_as_bytearray()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    query = image_to_query(image)
    await update.message.reply_text(f"🔍 Определено: {query}\nИщу лучшие предложения...")
    result = search_all_sources(query)
    await update.message.reply_text(result, parse_mode="Markdown")

def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.run_polling()

if __name__ == "__main__":
    main()
