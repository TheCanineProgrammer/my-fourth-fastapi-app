from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Optional
import os
import openai
import pandas as pd
from rapidfuzz import process, fuzz
import logging
import dask.dataframe as dd

# ------------------------------
# Logging configuration
# ------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logging.getLogger("datasets").setLevel(logging.WARNING)

# ------------------------------
# OpenAI API setup
# ------------------------------
#openai.api_key = "trb-efb1e474594463bbea-27e8-4578-b6ff-b2aed5414b57"
#openai.api_base = "https://turbo.torob.com/v1"

storage_options = {
    "key": "ZTeZ9y0SEj3gazBrpl6w",
    "secret": "h1K1gAxb5uggNo6iNCWUg02w3JfBvwEECNJzPa8y",
    "client_kwargs": {"endpoint_url": "https://newminio.darkube.app"}
}

# ------------------------------
# FastAPI setup
# ------------------------------
app = FastAPI()

# ------------------------------
# Pydantic models
# ------------------------------
class Message(BaseModel):
    type: str
    content: str

class ChatRequest(BaseModel):
    chat_id: str
    messages: List[Message]

# ------------------------------
# Load dataset
# ------------------------------

base_df = dd.read_parquet("s3://mybucket/base_products.parquet", storage_options=storage_options)[["random_key", "english_name", "persian_name", "extra_features"]].compute()

#random_key_to_names = base_df.set_index("random_key")[["english_name", "persian_name"]].to_dict(orient="index")
all_names = list(base_df.persian_name) + list(base_df.english_name)

# ------------------------------
# Endpoint
# ------------------------------
@app.post("/chat")
async def assistant(request: ChatRequest):
    last_message = request.messages[-1].content.strip()
    client = openai.OpenAI(api_key="trb-efb1e474594463bbea-27e8-4578-b6ff-b2aed5414b57", base_url="https://turbo.torob.com/v1")
    # Log incoming request
    logging.info(f"Received chat_id={request.chat_id}, message={last_message}")

    # --------
    # Scenario 1: Sanity checks
    # --------
    if last_message.lower() == "ping":
        response = {"message": "pong", "base_random_keys": None, "member_random_keys": None}
        logging.info(f"Response: {response}")
        return response

    if "return base random key" in last_message.lower():
        key = last_message.split(":")[-1].strip()
        response = {"message": None, "base_random_keys": [key], "member_random_keys": None}
        logging.info(f"Response: {response}")
        return response

    if "return member random key" in last_message.lower():
        key = last_message.split(":")[-1].strip()
        response = {"message": None, "base_random_keys": None, "member_random_keys": [key]}
        logging.info(f"Response: {response}")
        return response
    
    try:
        intent_response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You are an intent classifier."},
                {"role": "user", "content": f'''You will classify the intent of the user's query into one of these categories:
                                                1. "product_key" -> The user is asking for a product match (e.g., خرید محصول X).
                                                2. "feature" -> The user is asking about a feature of a product (e.g., عرض، وزن، سایز).
                                                3. "vendor" -> The user is asking about vendor stats like minimum, maximum, or mean price.

                                                Only answer with one word: product_key, feature, or vendor.

                                                User query: "{last_message}"
                                                '''}
            ],
            temperature=0.0
        )
        intent = intent_response.choices[0].message.content.strip().lower()
    except Exception as e:
        logging.error(f"OpenAI intent error: {e}")
        intent = "product"
    
    # --------
    # Scenario 2: Map query to a base product key using OpenAI
    # --------
    prompt = f'''
    کاربر یک محصول را توصیف کرده است.
    لطفاً فقط نام یا توضیح استاندارد همان محصول را بازگردان (مثلاً «فرشینه مخمل ترمزگیر عرض 1 متر طرح آشپزخانه کد 04»).
    
    کوئری کاربر: "{last_message}"'''

    key = None
    try:
        
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You normalize user product queries."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        predicted_name = response.choices[0].message.content.strip()
        logging.info(f"{predicted_name}")
    except Exception:
        predicted_name = ""
    
    key = None
    matched_row = None
    if predicted_name:
        match, score, idx = process.extractOne(
            predicted_name,
            all_names,
            scorer=fuzz.token_sort_ratio
        )

        if score > 70:  # accept only strong matches
            matched_row = base_df.iloc[idx]
            key = matched_row["random_key"]
    
    if intent == "feature" and matched_row is not None:
        feature = matched_row["extra_features"]
        prompt1 = f"""
    کاربر درباره‌ی ویژگی خاص یک محصول پرسیده است.
    ویژگی‌های محصول: {feature}

    فقط و فقط جواب مدنظر کاربر را برگردان (به انگلیسی).
    مثال: اگر در ویژگی‌ها نوشته شده "1.18 meter"، فقط همان "1.18 meter" را برگردان.
    """
        try:
            response = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": "You return only the requested feature."},
                    {"role": "user", "content": prompt1}
                ],
                temperature=0.0
            )
            feature_answer = response.choices[0].message.content.strip()
            return {
                "message": feature_answer,
                "base_random_keys": None,
                "member_random_keys": None
            }
        except Exception as e:
            logging.error(f"Feature extraction failed: {e}")
            return {
                "message": None,
                "base_random_keys": None,
                "member_random_keys": None
            }
    
    elif intent == "vendor" and matched_row is not None:
        members = dd.read_parquet("s3://mybucket/members.parquet", storage_options=storage_options)
        subset = members[members["base_random_key"] == key]
        min_price = subset["price"].min().compute()
        max_price = subset["price"].max().compute()
        mean_price = subset["price"].mean().compute()

        aggregates = {
                    "min_price": min_price,
                    "max_price": max_price,
                    "mean_price": round(mean_price, 2)
                }

        prompt = f'''
    کاربر سوالی درباره فروشنده ها پرسیده است.
    فقط مقدار عددی مدنظر کاربر را برگردان، هیچ توضیح اضافه نده.
    اعداد محاسبه شده برای این محصول: {aggregates}
    سوال کاربر: "{last_message}"
    '''

        # Call OpenAI LLM to select which number to return
        try:
            response = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": "You return only the numeric answer requested by the user."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0
            )
            numeric_answer = response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"Vendor info extraction failed: {e}")
            numeric_answer = None

        return {
                "message": numeric_answer,
                "base_random_keys": None,
                "member_random_keys": None
            }
    else:
        response = {
            "message": None,
            "base_random_keys": [key] if key else None,
            "member_random_keys": None
        }

        # Log outgoing response
        logging.info(f"Response: {response}")
        return response
