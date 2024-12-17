from flask import Flask, request, jsonify, session
import os
from dotenv import load_dotenv
from groq import Groq
from deep_translator import GoogleTranslator
from pymongo import MongoClient
from datetime import datetime
import uuid
from concurrent.futures import ThreadPoolExecutor
import cachetools
from threading import Lock
import asyncio
import aiohttp
from motor.motor_asyncio import AsyncIOMotorClient
from asgiref.wsgi import WsgiToAsgi
from hypercorn.asyncio import serve
from hypercorn.config import Config
import re

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
mongo_url = os.getenv("MONGO_URL")

# Async MongoDB setup
client = AsyncIOMotorClient(mongo_url, maxPoolSize=50)
db = client['myanmar_chatbot']
conversations = db['conversations']
sessions = db['sessions']

# Enhanced cache setup
translation_cache = cachetools.TTLCache(maxsize=2000, ttl=7200)
response_cache = cachetools.TTLCache(maxsize=1000, ttl=3600)
translation_lock = Lock()

class MyanmarChatbot:
    def __init__(self):
        self.client = Groq(api_key=groq_api_key)
        self.executor = ThreadPoolExecutor(max_workers=20)
        self.session = None
        
    async def init_session(self):
        if self.session is None:
            self.session = aiohttp.ClientSession()
        
    async def close_session(self):
        if self.session:
            await self.session.close()
    
    def detect_language(self, text):
        """
        Detect if text is primarily Myanmar or English
        Returns: 'my' for Myanmar, 'en' for English
        """
        myanmar_pattern = re.compile('[\u1000-\u109F]')
        myanmar_chars = len(myanmar_pattern.findall(text))
        return 'my' if myanmar_chars > len(text) * 0.1 else 'en'

    async def cached_translate(self, text, source, target):
        """Async cached translation"""
        if source == target:
            return text
            
        await self.init_session()
        cache_key = f"{text}:{source}:{target}"
        
        with translation_lock:
            if cache_key in translation_cache:
                return translation_cache[cache_key]
        
        try:
            translator = GoogleTranslator(source=source, target=target)
            result = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                translator.translate,
                text
            )
            
            with translation_lock:
                translation_cache[cache_key] = result
            return result
        except Exception as e:
            print(f"Translation error: {str(e)}")
            return None

    async def parallel_translate(self, texts, source, target):
        """Parallel translation of multiple texts"""
        if source == target:
            return texts
        tasks = [self.cached_translate(text, source, target) for text in texts]
        return await asyncio.gather(*tasks)

    async def get_conversation_history(self, session_id):
        """Async conversation history retrieval"""
        result = await conversations.find_one(
            {'session_id': session_id},
            {'messages': {'$slice': -5}}
        )
        return result.get('messages', []) if result else []

    async def save_conversation(self, session_id, query, response, is_myanmar):
        """Async conversation saving - only save Myanmar conversations"""
        if not is_myanmar:
            return
            
        try:
            new_messages = [
                {
                    "role": "user",
                    "content": query,
                    "timestamp": datetime.utcnow()
                },
                {
                    "role": "assistant",
                    "content": response,
                    "timestamp": datetime.utcnow()
                }
            ]

            await conversations.update_one(
                {'session_id': session_id},
                {
                    '$push': {
                        'messages': {
                            '$each': new_messages,
                            '$slice': -10
                        }
                    }
                },
                upsert=True
            )
        except Exception as e:
            print(f"MongoDB error: {str(e)}")

    async def get_llm_response(self, english_query, conversation_history):
        """Async LLM response with context"""
        try:
            cache_key = f"{english_query}:{hash(str(conversation_history))}"
            
            if cache_key in response_cache:
                return response_cache[cache_key]
            
            messages = [
                {"role": "system", "content": "You are a helpful assistant. Provide clear and concise responses."}
            ]
            messages.extend(conversation_history)
            messages.append({"role": "user", "content": english_query})
            
            completion = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: self.client.chat.completions.create(
                    model="llama3-8b-8192",
                    messages=messages,
                    temperature=0.7,
                    max_tokens=4096,
                    top_p=1,
                    stream=False
                )
            )
            
            response = completion.choices[0].message.content
            response_cache[cache_key] = response
            return response
            
        except Exception as e:
            print(f"LLM error: {str(e)}")
            return None

    async def chat(self, user_input, session_id):
        """Async optimized chat function with language detection"""
        try:
            # Detect input language
            input_language = self.detect_language(user_input)
            is_myanmar = input_language == 'my'
            
            # Convert input to English if necessary
            english_query = user_input
            if is_myanmar:
                english_query = await self.cached_translate(user_input, 'my', 'en')
                if not english_query:
                    return "သင်၏မေးခွန်းကို ဘာသာပြန်ရာတွင် အမှားအယွင်းရှိနေပါသည်။"
            
            # Get and translate conversation history if it's a Myanmar conversation
            history = []
            if is_myanmar:
                history = await self.get_conversation_history(session_id)
                if history:
                    history_texts = [msg["content"] for msg in history[-2:]]
                    translated_history = await self.parallel_translate(history_texts, 'my', 'en')
                    history = [
                        {"role": msg["role"], "content": trans}
                        for msg, trans in zip(history[-2:], translated_history)
                    ]
            
            # Get LLM response
            english_response = await self.get_llm_response(english_query, history)
            if not english_response:
                return "ပြန်လည်ဖြေကြားရာတွင် အမှားအယွင်းရှိနေပါသည်။" if is_myanmar else "An error occurred while generating the response."
            
            # Translate response to Myanmar if necessary
            final_response = english_response
            if is_myanmar:
                myanmar_response = await self.cached_translate(english_response, 'en', 'my')
                if not myanmar_response:
                    return "ဖြေကြားချက်ကို ဘာသာပြန်ရာတွင် အမှားအယွင်းရှိနေပါသည်။"
                final_response = myanmar_response
            
            # Save conversation only if it's in Myanmar
            if is_myanmar:
                await self.save_conversation(session_id, user_input, final_response, is_myanmar)
            
            return final_response
            
        except Exception as e:
            print(f"Chat error: {str(e)}")
            return "စနစ်တွင် အမှားတစ်ခုဖြစ်ပွားခဲ့သည်။" if is_myanmar else "A system error occurred."

# Initialize chatbot globally
chatbot = None

async def init_chatbot():
    """Initialize chatbot with async session"""
    global chatbot
    chatbot = MyanmarChatbot()
    await chatbot.init_session()
    return chatbot

async def get_chatbot():
    """Get or create chatbot instance"""
    global chatbot
    if chatbot is None:
        chatbot = await init_chatbot()
    return chatbot

async def create_or_get_session():
    """Async session management"""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
        await sessions.update_one(
            {'session_id': session['session_id']},
            {'$setOnInsert': {'created_at': datetime.utcnow()}},
            upsert=True
        )
    return session['session_id']

@app.route('/chat', methods=['POST'])
async def chat():
    try:
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({
                'error': 'Message is required.'
            }), 400
            
        user_message = data['message']
        session_id = await create_or_get_session()
        
        bot = await get_chatbot()
        response = await bot.chat(user_message, session_id)
        
        return jsonify({
            'response': response,
            'session_id': session_id
        })
        
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        return jsonify({
            'error': 'An error occurred. Please try again.'
        }), 500

@app.route('/clear-history', methods=['POST'])
async def clear_history():
    try:
        session_id = session.get('session_id')
        
        if not session_id:
            return jsonify({
                'error': 'No session found.'
            }), 400
            
        await conversations.delete_one({'session_id': session_id})
        session['session_id'] = str(uuid.uuid4())
            
        return jsonify({
            'message': 'Conversation history cleared.',
            'new_session_id': session['session_id']
        })
        
    except Exception as e:
        print(f"Error in clear-history endpoint: {str(e)}")
        return jsonify({
            'error': 'An error occurred. Please try again.'
        }), 500

async def cleanup():
    """Cleanup function to close sessions"""
    if chatbot:
        await chatbot.close_session()

if __name__ == '__main__':
    config = Config()
    config.bind = ["0.0.0.0:5000"]
    
    # Create ASGI app
    asgi_app = WsgiToAsgi(app)
    
    # Run the async server
    asyncio.run(serve(asgi_app, config))
