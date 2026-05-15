import os
import chromadb
from google import genai

from dotenv import load_dotenv

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    ApplicationBuilder,
    MessageHandler,
    ContextTypes,
    filters,
    CommandHandler
)

# ---------------- ENV ---------------- #

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# ---------------- CHROMADB ---------------- #

chroma_client = chromadb.CloudClient(
    api_key=os.getenv("CHROMA_API_KEY"),
    tenant=os.getenv("CHROMA_TENANT"),
    database=os.getenv("CHROMA_DATABASE")
)

collection = chroma_client.get_or_create_collection(name="passos_magicos_dados")

# ---------------- GEMINI ---------------- #

gemini_client = genai.Client(
    api_key=GOOGLE_API_KEY
)

SYSTEM_PROMPT = """
Você é um assistente virtual da ONG Passos Mágicos.

REGRAS:
- Responda de forma simples e fácil de entender.
- Use linguagem amigável e acolhedora.
- Limite as respostas a no máximo 2 parágrafos curtos.
- Não use Markdown.
- Não use listas longas.
- Não use símbolos como #, *, _, ``` ou tabelas.
- Use poucos emojis quando fizer sentido.
- Seja direto e claro.
- Caso não saiba a resposta, diga isso honestamente.
- Priorize contexto educacional e social.
"""

# ------------ COMANDOS ------------ #

#Comando /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):

    await update.message.reply_text(
        """🤖 Olá! Eu sou o assistente inteligente da Passos Mágicos.

        Posso ajudar com:
        📊 Indicadores educacionais
        📈 Análises de desempenho
        🎯 Risco de evasão
        🧠 Explicações sobre INDE, IPV e fases
        📚 Metodologia da ONG

        Exemplos:
        • "Qual a média do INDE em 2024?"
        • "O que significa IPV?"
        • "Qual pedra possui maior risco de evasão?"

        🚀 Como posso ajudar?
        """
    )

# responder mensagens
async def responder(update: Update, context: ContextTypes.DEFAULT_TYPE):
    pergunta = update.message.text
    try:
        # avisa digitando
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id,
            action=ChatAction.TYPING
        )
        # BUSCA CONTEXTO NO CHROMA
        resultados = collection.query(
            query_texts=[pergunta],
            n_results=4
        )
        documentos = resultados["documents"][0]
        contexto = "\n".join(documentos)
        # PROMPT FINAL
        prompt = f"""
        {SYSTEM_PROMPT}
        CONTEXTO:
        {contexto}
        PERGUNTA:
        {pergunta}
        """
        # GEMINI
        resposta = gemini_client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=prompt
        )
        texto_resposta = resposta.text
        # fallback caso venha vazio
        if not texto_resposta:

            texto_resposta = (
                "⚠️ Não consegui gerar uma resposta no momento."
            )

        await update.message.reply_text(
            texto_resposta
        )

    except Exception as e:

        print("\nERRO:")
        print(e)

        await update.message.reply_text(
            "⚠️ O assistente está temporariamente indisponível.\n"
            "Tente novamente em alguns instantes"
        )

# ------------ Telegram App e Handlers ------------ #
# Telegram App
app = (ApplicationBuilder()
    .token(TELEGRAM_TOKEN)
    .connect_timeout(30)
    .read_timeout(30)
    .write_timeout(30)
    .pool_timeout(30)
.build()
)

handlers = [
    CommandHandler("start", start),
    MessageHandler(filters.TEXT & ~filters.COMMAND, responder)
]

for h in handlers:
    app.add_handler(h)

print("Bot Gemini iniciado...")

app.run_polling(close_loop=False)