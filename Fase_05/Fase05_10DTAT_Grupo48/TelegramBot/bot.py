import os

from dotenv import load_dotenv
from google import genai

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    ApplicationBuilder,
    MessageHandler,
    ContextTypes,
    filters,
    CommandHandler
)
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

gauth = GoogleAuth()
gauth.LoadCredentialsFile("credentials.json")

if gauth.credentials is None:
    gauth.LocalWebserverAuth()
elif gauth.access_token_expired:
    gauth.Refresh()
else:
    gauth.Authorize()
gauth.SaveCredentialsFile("credentials.json")

drive = GoogleDrive(gauth)
file_list = drive.ListFile({'q': "'1BsWJJSTAhfeVrjY8sQJFO7gQkuJ1Ju3Q' in parents and trashed=false"}).GetList()
for file in file_list:
    if "BASE" in file['title']:
        file.GetContentFile(file['title']) # Baixa o CSV localmente

# from src.pede_cleaning import build_unified, cleaning_report
# OUT_PARQUET = ROOT / "data_processed" / "pede_unificado.parquet"
# df = build_unified(root=ROOT, save_parquet=OUT_PARQUET)
# cleaning_report(df)

# carrega .env
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# cliente Gemini
client = genai.Client(api_key=GOOGLE_API_KEY)

#Comando /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):

    await update.message.reply_text(
        "🤖 Sou o bot do grupo 48, vulgo melhor grupo da Postech FIAP.\n"
        "🚀 Como podemos auxiliar a Passos Mágicos hoje?"
    )

# responder mensagens
async def responder(update: Update, context: ContextTypes.DEFAULT_TYPE):

    pergunta = update.message.text

    try:
        #Avise que está pensando
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id,
            action=ChatAction.TYPING
        )

        resposta = client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=pergunta
        )

        await update.message.reply_text(
            resposta.text
        )

    except Exception as e:

        await update.message.reply_text(
            f"Erro: {str(e)}"
        )

# telegram
app = (ApplicationBuilder()
    .token(TELEGRAM_TOKEN)
    .connect_timeout(30)
    .read_timeout(30)
    .write_timeout(30)
    .pool_timeout(30)
.build()
)

app.add_handler(
    MessageHandler(
        filters.TEXT & ~filters.COMMAND,
        responder
    )
)

app.add_handler(
    CommandHandler("start", start)
)

print("Bot Gemini iniciado...")

app.run_polling(close_loop=False)