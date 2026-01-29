import os
from dotenv import load_dotenv
from telegram import Bot
import asyncio

# Load environment variables
load_dotenv()

# Telegram setup
telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')

if not telegram_token or not telegram_chat_id:
    print("Error: TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set in .env")
    exit(1)

bot = Bot(token=telegram_token)

async def send_test_message():
    message = "Test message from Kraken Trading Bot! If you see this, Telegram is working. 🚀"
    await bot.send_message(chat_id=telegram_chat_id, text=message)
    print("Test message sent to Telegram!")

# Run the test
asyncio.run(send_test_message())