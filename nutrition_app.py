import os
import json
import bcrypt
import smtplib
import random
import pandas as pd
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
from email.mime.text import MIMEText
from streamlit_extras.colored_header import colored_header
from streamlit_extras.stylable_container import stylable_container
from datetime import datetime
import time
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Initialize Session State ---
def initialize_session_state():
    """Initialize all session state variables"""
    defaults = {
        'logged_in': False,
        'user_id': "",
        'profile_completed': False,
        'login_attempts': 0,
        'show_register': False,
        'show_password_change': False,
        'show_otp_verification': False,
        'num_items': 5,
        'vegetarian': False,
        'daily_calories': 2000,
        'current_meal_plan': None,
        'activity_level': "moderate",
        'show_nutrition_interface': True,
        'diet_goal': "Maintain Weight",
        'generated_meal_count': 0
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# --- Load Environment ---
def load_environment():
    """Load environment variables with error handling"""
    try:
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            st.error("❌ Google API Key not found in environment variables. Please set it in a .env file.")
            return False
            
        genai.configure(api_key=api_key)
        
        # Check SMTP environment variables
        if not os.getenv("SMTP_EMAIL") or not os.getenv("SMTP_PASSWORD"):
            st.warning("⚠️ SMTP_EMAIL or SMTP_PASSWORD not found in environment variables. Email functionality will be mocked.")

        return True
    except Exception as e:
        logger.error(f"Error loading environment: {str(e)}")
        st.error(f"❌ Error loading environment: {str(e)}")
        return False

# --- Constants ---
SMTP_EMAIL = os.getenv("SMTP_EMAIL")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
DEFAULT_PASSWORD = "Test@1234" # Consider making this more secure or randomly generated for production
USER_DB_FILE = "users.csv"
USER_PROFILE_FILE = "user_profiles.csv"
NUTRITION_DB_FILE = "nutrition_data.csv"
MEAL_HISTORY_FILE = "meal_history.csv"
FOOD_DB_FILE = "food_database.csv"

# Enhanced Medical Conditions Database
MEDICAL_CONDITIONS = {
    "Diabetes": {
        "avoid": ["sugar", "white bread", "white rice", "processed foods", "candy", "soda", "pastries"],
        "recommend": ["whole grains", "leafy greens", "berries", "nuts", "lean protein", "quinoa"],
        "description": "Foods that help manage blood sugar levels"
    },
    "High Blood Pressure": {
        "avoid": ["salt", "processed meats", "pickles", "canned soups", "fast food", "bacon"],
        "recommend": ["bananas", "spinach", "avocados", "garlic", "berries", "oats"],
        "description": "Low-sodium foods that support heart health"
    },
    "Heart Disease": {
        "avoid": ["saturated fats", "trans fats", "processed meats", "fried foods", "butter"],
        "recommend": ["fatty fish", "oats", "berries", "dark chocolate", "olive oil", "nuts"],
        "description": "Heart-healthy foods rich in omega-3s"
    },
    "Kidney Disease": {
        "avoid": ["high-potassium foods", "processed meats", "dairy", "bananas", "oranges"],
        "recommend": ["apples", "berries", "cauliflower", "olive oil", "white rice"],
        "description": "Low-potassium and low-phosphorus foods"
    },
    "Celiac Disease": {
        "avoid": ["wheat", "barley", "rye", "most processed foods", "beer", "pasta"],
        "recommend": ["quinoa", "rice", "gluten-free oats", "fruits", "vegetables"],
        "description": "Naturally gluten-free foods"
    },
    "Lactose Intolerance": {
        "avoid": ["milk", "cheese", "yogurt", "butter", "ice cream", "cream"],
        "recommend": ["almond milk", "lactose-free products", "leafy greens", "nuts"],
        "description": "Dairy-free alternatives and calcium-rich foods"
    },
    "High Cholesterol": {
        "avoid": ["fried foods", "processed meats", "full-fat dairy", "baked goods", "egg yolks"],
        "recommend": ["oats", "nuts", "fatty fish", "olive oil", "beans", "apples"],
        "description": "Foods that help lower cholesterol"
    },
    "Gout": {
        "avoid": ["red meat", "organ meats", "shellfish", "alcohol", "sugary drinks"],
        "recommend": ["low-fat dairy", "vegetables", "cherries", "whole grains", "water"],
        "description": "Low-purine foods that reduce uric acid"
    },
    "GERD": {
        "avoid": ["spicy foods", "citrus", "tomatoes", "chocolate", "coffee", "alcohol"],
        "recommend": ["oatmeal", "ginger", "lean meats", "vegetables", "melons"],
        "description": "Non-acidic foods that reduce reflux"
    },
    "IBS": {
        "avoid": ["high-fiber foods", "dairy", "artificial sweeteners", "beans", "cabbage"],
        "recommend": ["rice", "bananas", "carrots", "lean proteins", "herbal teas"],
        "description": "Gentle foods that reduce digestive symptoms"
    }
}

# --- Database Setup Functions ---
def ensure_user_db():
    """Create user database files if they don't exist"""
    try:
        columns = ["user_id", "password", "otp", "verified", "needs_password_change"]
        if not os.path.exists(USER_DB_FILE):
            pd.DataFrame(columns=columns).to_csv(USER_DB_FILE, index=False)
            logger.info(f"{USER_DB_FILE} created.")
        
        profile_columns = ["user_id", "age", "height", "weight", "gender", "medical_conditions", 
                             "diet_goal", "vegetarian", "bmi", "daily_calories", "activity_level", "created_at"]
        if not os.path.exists(USER_PROFILE_FILE):
            pd.DataFrame(columns=profile_columns).to_csv(USER_PROFILE_FILE, index=False)
            logger.info(f"{USER_PROFILE_FILE} created.")
        
        if not os.path.exists(MEAL_HISTORY_FILE):
            pd.DataFrame(columns=["user_id", "date", "meal", "rating", "feedback"]).to_csv(MEAL_HISTORY_FILE, index=False)
            logger.info(f"{MEAL_HISTORY_FILE} created.")
        
        return True
    except Exception as e:
        logger.error(f"Error creating user database: {str(e)}")
        st.error(f"❌ Error setting up user databases: {str(e)}")
        return False

def ensure_nutrition_db():
    """Create nutrition database with sample data"""
    try:
        if not os.path.exists(NUTRITION_DB_FILE):
            sample_data = {
                "food_item": ["Chicken Breast", "Brown Rice", "Broccoli", "Eggs", "Salmon", 
                              "Quinoa", "Spinach", "Almonds", "Sweet Potato", "Greek Yogurt",
                              "Oats", "Avocado", "Blueberries", "Lentils", "Tuna"],
                "category": ["Non-Veg", "Veg", "Veg", "Non-Veg", "Non-Veg", 
                              "Veg", "Veg", "Veg", "Veg", "Veg",
                              "Veg", "Veg", "Veg", "Veg", "Non-Veg"],
                "calories": [165, 111, 55, 155, 208, 120, 23, 576, 86, 59, 68, 160, 57, 116, 144],
                "protein": [31, 2.6, 3.7, 13, 20, 4.4, 2.9, 21, 1.6, 10, 2.4, 2, 0.7, 9, 30],
                "carbs": [0, 23, 11, 1.1, 0, 21, 3.6, 22, 20, 3.6, 12, 9, 14, 20, 0],
                "fats": [3.6, 0.9, 0.6, 11, 13, 1.9, 0.4, 49, 0.1, 0.4, 1.4, 15, 0.3, 0.4, 1],
                "fiber": [0, 1.8, 2.6, 0, 0, 2.8, 2.2, 12, 3, 0, 1.7, 7, 2.4, 8, 0],
                "vegetarian": [False, True, True, False, False, True, True, True, True, True, 
                               True, True, True, True, False],
                "meal_type": ["lunch,dinner", "lunch,dinner", "lunch,dinner", "breakfast", "lunch,dinner",
                               "breakfast,lunch", "lunch,dinner", "snacks", "lunch,dinner", "breakfast,snacks",
                               "breakfast", "breakfast,lunch", "breakfast,snacks", "lunch,dinner", "lunch,dinner"]
            }
            pd.DataFrame(sample_data).to_csv(NUTRITION_DB_FILE, index=False)
            logger.info(f"{NUTRITION_DB_FILE} created with sample data.")
        return True
    except Exception as e:
        logger.error(f"Error creating nutrition database: {str(e)}")
        st.error(f"❌ Error setting up nutrition database: {str(e)}")
        return False

def ensure_food_db():
    """Create comprehensive food database"""
    try:
        if not os.path.exists(FOOD_DB_FILE):
            sample_data = {
                "food_item": ["Chicken Breast", "Brown Rice", "Broccoli", "Eggs", "Salmon", 
                              "Quinoa", "Spinach", "Almonds", "Sweet Potato", "Greek Yogurt"],
                "category": ["Non-Veg", "Veg", "Veg", "Non-Veg", "Non-Veg", 
                              "Veg", "Veg", "Veg", "Veg", "Veg"],
                "calories": [165, 111, 55, 155, 208, 120, 23, 576, 86, 59],
                "protein": [31, 2.6, 3.7, 13, 20, 4.4, 2.9, 21, 1.6, 10],
                "carbs": [0, 23, 11, 1.1, 0, 21, 3.6, 22, 20, 3.6],
                "fats": [3.6, 0.9, 0.6, 11, 13, 1.9, 0.4, 49, 0.1, 0.4],
                "fiber": [0, 1.8, 2.6, 0, 0, 2.8, 2.2, 12, 3, 0],
                "vegetarian": [False, True, True, False, False, True, True, True, True, True],
                "meal_type": ["lunch,dinner", "lunch,dinner", "lunch,dinner", "breakfast", "lunch,dinner",
                               "breakfast,lunch", "lunch,dinner", "snacks", "lunch,dinner", "breakfast,snacks"],
                "is_main": [True, True, False, True, True, True, False, False, True, False],
                "ingredients": ["chicken", "rice", "broccoli", "eggs", "salmon", "quinoa", "spinach", "almonds", "sweet potato", "yogurt"]
            }
            pd.DataFrame(sample_data).to_csv(FOOD_DB_FILE, index=False)
            logger.info(f"{FOOD_DB_FILE} created with sample data.")
        return True
    except Exception as e:
        logger.error(f"Error creating food database: {str(e)}")
        st.error(f"❌ Error setting up food database: {str(e)}")
        return False

# --- Enhanced Google API Functions ---
def generate_medical_condition_diet(medical_conditions, vegetarian=False, daily_calories=2000, diet_goal="Maintain Weight"):
    """
    Generate a comprehensive diet plan using Google's Gemini API
    considering medical conditions, dietary preferences, and goals
    """
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        # Prepare medical condition information
        condition_info = []
        for condition in medical_conditions:
            if condition in MEDICAL_CONDITIONS:
                avoid_foods = ", ".join(MEDICAL_CONDITIONS[condition]["avoid"])
                recommend_foods = ", ".join(MEDICAL_CONDITIONS[condition]["recommend"])
                description = MEDICAL_CONDITIONS[condition]["description"]
                condition_info.append(f"{condition}: {description}. Avoid: {avoid_foods}. Recommend: {recommend_foods}")
        
        condition_text = "\n".join(condition_info) if condition_info else "No specific medical conditions"
        
        # Adjust calorie distribution based on meal types and diet goal
        breakfast_cal_percent = 0.25
        lunch_cal_percent = 0.35
        dinner_cal_percent = 0.30
        snacks_cal_percent = 0.10

        prompt = f"""
        As a certified nutritionist, create a comprehensive daily meal plan for someone with the following profile:

        Medical Conditions: {condition_text}
        Dietary Preference: {'Vegetarian' if vegetarian else 'Non-Vegetarian'}
        Daily Calorie Target: {daily_calories} calories
        Diet Goal: {diet_goal}

        Please provide a detailed meal plan with:
        1. Breakfast ({breakfast_cal_percent*100}% of daily calories, target: {round(daily_calories * breakfast_cal_percent)} calories)
        2. Lunch ({lunch_cal_percent*100}% of daily calories, target: {round(daily_calories * lunch_cal_percent)} calories)  
        3. Dinner ({dinner_cal_percent*100}% of daily calories, target: {round(daily_calories * dinner_cal_percent)} calories)
        4. Snacks ({snacks_cal_percent*100}% of daily calories, target: {round(daily_calories * snacks_cal_percent)} calories)

        For each meal, include 2-3 specific food items with:
        - Exact food name
        - Portion size (in grams or common units like "1 cup", "2 slices", "1 medium apple")
        - Nutritional information per serving (calories, protein, carbs, fats, fiber). **Ensure these values are realistic and accurately reflect the portion size.**
        - Why this food is beneficial for the medical conditions mentioned (if any), or general health benefits. **Provide 1-2 concise sentences for benefits.**

        **CRITICAL:**
        - Ensure the total calories for each meal (`total_calories` field) closely match the target percentage of the daily calorie target.
        - The overall daily total calories in `daily_summary` should be very close to the Daily Calorie Target provided.
        - Avoid any foods explicitly listed to be avoided for the given medical conditions.
        - Prioritize recommended foods for the given medical conditions.
        - **ALL FIELDS MUST BE POPULATED WITH REALISTIC VALUES. DO NOT LEAVE ANY FIELD AS N/A, 0, OR EMPTY.**

        Return ONLY a JSON object in this exact format:
        {{
            "breakfast": {{
                "items": [
                    {{
                        "food": "Food Name",
                        "quantity": "100g",
                        "calories": 150.0,
                        "protein": 10.0,
                        "carbs": 15.0,
                        "fats": 5.0,
                        "fiber": 3.0,
                        "benefits": "Why this food helps with the conditions"
                    }},
                    {{
                        "food": "Food Name 2",
                        "quantity": "50g",
                        "calories": 75.0,
                        "protein": 5.0,
                        "carbs": 7.5,
                        "fats": 2.5,
                        "fiber": 1.5,
                        "benefits": "Why this food helps with the conditions"
                    }}
                ],
                "total_calories": 225.0,
                "meal_benefits": "Overall benefits of this breakfast"
            }},
            "lunch": {{
                "items": [
                    {{
                        "food": "Food Name",
                        "quantity": "200g",
                        "calories": 300.0,
                        "protein": 20.0,
                        "carbs": 30.0,
                        "fats": 10.0,
                        "fiber": 6.0,
                        "benefits": "Why this food helps with the conditions"
                    }}
                ],
                "total_calories": 300.0,
                "meal_benefits": "Overall benefits of this lunch"
            }},
            "dinner": {{
                "items": [
                    {{
                        "food": "Food Name",
                        "quantity": "180g",
                        "calories": 250.0,
                        "protein": 18.0,
                        "carbs": 25.0,
                        "fats": 8.0,
                        "fiber": 5.0,
                        "benefits": "Why this food helps with the conditions"
                    }}
                ],
                "total_calories": 250.0,
                "meal_benefits": "Overall benefits of this dinner"
            }},
            "snacks": {{
                "items": [
                    {{
                        "food": "Food Name",
                        "quantity": "80g",
                        "calories": 100.0,
                        "protein": 5.0,
                        "carbs": 10.0,
                        "fats": 3.0,
                        "fiber": 2.0,
                        "benefits": "Why this food helps with the conditions"
                    }}
                ],
                "total_calories": 100.0,
                "meal_benefits": "Overall benefits of these snacks"
            }},
            "daily_summary": {{
                "total_calories": 875.0,
                "total_protein": 53.0,
                "total_carbs": 80.0,
                "total_fats": 23.5,
                "total_fiber": 17.5,
                "medical_compliance": "How this plan addresses the medical conditions and adherence to dietary goals. For example, 'This plan is designed to help manage blood sugar levels for Diabetes by focusing on whole grains and lean proteins, and is gluten-free for Celiac Disease.'"
            }}
        }}

        **Ensure all nutritional numbers (calories, protein, carbs, fats, fiber) are realistic and the food items are appropriate for the specified medical conditions and dietary preferences.**
        **The `total_calories` for each meal and the `daily_summary` should accurately reflect the sum of the `items` within them.**
        **Provide a detailed `medical_compliance` explanation in the `daily_summary`.**
        """
        
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Clean up the response to ensure it's valid JSON
        if response_text.startswith('```json'):
            response_text = response_text[7:-3].strip()
        elif response_text.startswith('```'):
            response_text = response_text[3:-3].strip()
        
        # --- CRITICAL DEBUGGING STEP ---
        logger.info(f"Raw AI Response for Meal Plan: \n{response_text}") 
        # --- END CRITICAL DEBUGGING STEP ---
        
        # Parse JSON response
        meal_plan = json.loads(response_text)
        
        # Validate the structure and convert types
        required_meals = ["breakfast", "lunch", "dinner", "snacks"]
        if not all(meal in meal_plan for meal in required_meals):
            raise ValueError("Missing required meal types in AI response")
            
        for meal_name in required_meals:
            meal = meal_plan[meal_name]
            if "items" not in meal or "total_calories" not in meal or "meal_benefits" not in meal:
                raise ValueError(f"Missing required fields in {meal_name} meal (items, total_calories, or meal_benefits)")
            
            # Ensure total_calories is a float
            meal["total_calories"] = float(meal["total_calories"])
            
            for item in meal["items"]:
                required_fields = ["food", "quantity", "calories", "protein", "carbs", "fats", "fiber", "benefits"]
                if not all(field in item for field in required_fields):
                    raise ValueError(f"Missing fields in food item: {item} within {meal_name}. Missing one of: {', '.join(required_fields)}")
                
                # Convert to proper types
                item["calories"] = float(item["calories"])
                item["protein"] = float(item["protein"])
                item["carbs"] = float(item["carbs"])
                item["fats"] = float(item["fats"])
                item["fiber"] = float(item["fiber"])
                # Ensure benefits is not empty
                if not item["benefits"]:
                    item["benefits"] = "General nutritional benefits."

            # Ensure meal_benefits is not empty
            if not meal["meal_benefits"]:
                meal["meal_benefits"] = f"A balanced {meal_name} for your dietary needs."

        # Validate daily_summary structure and convert types
        daily_summary = meal_plan.get('daily_summary')
        if not daily_summary:
            raise ValueError("Missing daily_summary in AI response")

        summary_fields = ["total_calories", "total_protein", "total_carbs", "total_fats", "total_fiber", "medical_compliance"]
        if not all(field in daily_summary for field in summary_fields):
            raise ValueError("Missing required fields in daily_summary")

        daily_summary["total_calories"] = float(daily_summary["total_calories"])
        daily_summary["total_protein"] = float(daily_summary["total_protein"])
        daily_summary["total_carbs"] = float(daily_summary["total_carbs"])
        daily_summary["total_fats"] = float(daily_summary["total_fats"])
        daily_summary["total_fiber"] = float(daily_summary["total_fiber"])
        # Ensure medical_compliance is not empty
        if not daily_summary["medical_compliance"]:
            daily_summary["medical_compliance"] = "This plan aligns with your general dietary goals."


        return meal_plan
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {str(e)}. Response text: \n{response_text}")
        st.error("❌ Failed to parse meal plan from AI response. The AI might have returned malformed JSON. Check logs for raw response.")
        return None
    except ValueError as e:
        logger.error(f"Meal plan structure validation error: {str(e)}. Response text: \n{response_text}")
        st.error(f"❌ Received malformed meal plan from AI: {str(e)}. Please try again. Check logs for raw response.")
        return None
    except Exception as e:
        logger.error(f"Error generating diet plan: {str(e)}")
        st.error(f"❌ Error generating meal plan: {str(e)}. Please check your API key and try again.")
        return None


def get_food_alternatives(food_item, medical_conditions, vegetarian=False):
    """Get alternative foods using Google API"""
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        condition_restrictions = []
        for condition in medical_conditions:
            if condition in MEDICAL_CONDITIONS:
                avoid_foods = ", ".join(MEDICAL_CONDITIONS[condition]["avoid"])
                condition_restrictions.append(f"{condition}: avoid {avoid_foods}")
        
        restrictions = "\n".join(condition_restrictions) if condition_restrictions else "No specific restrictions"
        
        prompt = f"""
        Suggest 3 healthy alternatives to "{food_item}" that are:
        - {'Vegetarian' if vegetarian else 'Non-vegetarian or vegetarian'}
        - Safe for someone with these medical conditions: {restrictions}
        - Similar in nutritional value and meal type

        Return ONLY a JSON array of food names:
        ["Alternative 1", "Alternative 2", "Alternative 3"]
        """
        
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        if response_text.startswith('```json'):
            response_text = response_text[7:-3].strip()
        elif response_text.startswith('```'):
            response_text = response_text[3:-3].strip()
        
        alternatives = json.loads(response_text)
        return alternatives if isinstance(alternatives, list) else []
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error for alternatives: {str(e)}. Response text: {response_text}")
        st.error("❌ Failed to parse food alternatives from AI response.")
        return []
    except Exception as e:
        logger.error(f"Error getting food alternatives: {str(e)}")
        st.error(f"❌ Error fetching alternatives: {str(e)}")
        return []

# --- Authentication Functions ---
def hash_password(password):
    """Hash password using bcrypt"""
    try:
        # Use a higher rounds value for better security in a real application
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt(rounds=12)).decode() 
    except Exception as e:
        logger.error(f"Error hashing password: {str(e)}")
        return None

def send_email(to_email, subject, body):
    """Send email with error handling"""
    if not SMTP_EMAIL or not SMTP_PASSWORD:
        st.warning("⚠️ Email configuration not found (SMTP_EMAIL/SMTP_PASSWORD). Using mock email sending.")
        logger.warning(f"Mock email sent to {to_email} with subject '{subject}'")
        return True # Simulate success if email credentials are not set
        
    try:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = SMTP_EMAIL
        msg["To"] = to_email
        
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(SMTP_EMAIL, SMTP_PASSWORD)
            server.sendmail(SMTP_EMAIL, to_email, msg.as_string())
        logger.info(f"Email sent successfully to {to_email}")
        return True
    except Exception as e:
        logger.error(f"Email failed: {str(e)}", exc_info=True) # Log full traceback
        st.error(f"❌ Failed to send email to {to_email}. Please check your SMTP settings or try again.")
        return False

def generate_otp():
    """Generate 6-digit OTP"""
    return str(random.randint(100000, 999999))

def register_user(user_id):
    """Register new user with enhanced error handling"""
    try:
        if not os.path.exists(USER_DB_FILE):
            if not ensure_user_db():
                st.error("❌ Failed to set up user database. Cannot register.")
                return False
                
        df = pd.read_csv(USER_DB_FILE)
        if user_id in df.user_id.values:
            st.warning("⚠️ This email is already registered. Please login or reset your password.")
            return False # Already exists
        
        otp = generate_otp()
        hashed = hash_password(DEFAULT_PASSWORD)
        
        if not hashed:
            st.error("❌ Failed to hash password during registration.")
            return False
            
        new_user = pd.DataFrame([{
            "user_id": user_id,
            "password": hashed,
            "otp": otp,
            "verified": False,
            "needs_password_change": True
        }])
        
        df = pd.concat([df, new_user], ignore_index=True)
        df.to_csv(USER_DB_FILE, index=False)
        logger.info(f"User {user_id} registered successfully.")
        
        body = f"""Welcome to Nutrition Chatbot!

Your App ID: {user_id}
Temporary Password: {DEFAULT_PASSWORD}
OTP for verification: {otp}

Please verify your account by entering the OTP when prompted. You will be required to change your password upon first login.

Best regards,
Nutrition Team
"""
        if send_email(user_id, "Nutrition Chatbot Registration", body):
            st.success("✅ Registration successful! Please check your email for the OTP and temporary password.")
            return True
        else:
            st.error("❌ Registration successful, but failed to send verification email. Please contact support.")
            # Optionally, revert registration or flag user for manual verification
            return False
            
    except Exception as e:
        logger.error(f"Error registering user: {str(e)}")
        st.error(f"❌ Registration failed: {str(e)}")
        return False

def verify_otp(user_id, otp_input):
    """Verify OTP with enhanced error handling"""
    try:
        if not os.path.exists(USER_DB_FILE):
            st.error("❌ User database not found. Cannot verify OTP.")
            return False
            
        df = pd.read_csv(USER_DB_FILE)
        user = df[df.user_id == user_id]
        
        if not user.empty and str(user.iloc[0]['otp']).strip() == str(otp_input).strip():
            df.loc[df.user_id == user_id, 'verified'] = True
            df.to_csv(USER_DB_FILE, index=False)
            logger.info(f"User {user_id} OTP verified successfully.")
            return True
        logger.warning(f"Failed OTP verification for user {user_id}")
        return False
        
    except Exception as e:
        logger.error(f"Error verifying OTP: {str(e)}")
        st.error(f"❌ Error during OTP verification: {str(e)}")
        return False

def check_login(user_id, password):
    """Check login credentials with enhanced error handling"""
    try:
        if not os.path.exists(USER_DB_FILE):
            return "not_registered"
            
        df = pd.read_csv(USER_DB_FILE)
        user = df[df.user_id == user_id]
        
        if user.empty:
            return "not_registered"
            
        if not user.iloc[0]['verified']:
            return "not_verified"
            
        try:
            if bcrypt.checkpw(password.encode(), user.iloc[0]['password'].encode()):
                # Check if password change is required (first login or reset)
                if user.iloc[0]['needs_password_change']:
                    return "change_required"
                else:
                    return "success"
            else:
                logger.warning(f"Wrong password attempt for user {user_id}")
                return "wrong_password"
        except Exception as e:
            logger.error(f"Password check error for user {user_id}: {str(e)}")
            return "wrong_password" # Treat hashing/comparison errors as wrong password for security
            
    except Exception as e:
        logger.error(f"Login check error: {str(e)}")
        st.error(f"❌ An unexpected error occurred during login: {str(e)}")
        return "error"

def update_password(user_id, new_password):
    """Update user password"""
    try:
        if not os.path.exists(USER_DB_FILE):
            st.error("❌ User database not found. Cannot update password.")
            return False

        df = pd.read_csv(USER_DB_FILE)
        hashed = hash_password(new_password)
        
        if not hashed:
            st.error("❌ Failed to hash new password.")
            return False
            
        df.loc[df.user_id == user_id, 'password'] = hashed
        df.loc[df.user_id == user_id, 'needs_password_change'] = False
        df.to_csv(USER_DB_FILE, index=False)
        logger.info(f"Password updated successfully for user {user_id}.")
        return True
        
    except Exception as e:
        logger.error(f"Error updating password: {str(e)}")
        st.error(f"❌ Error updating password: {str(e)}")
        return False

def load_user_profile(user_id):
    """Load user profile with error handling"""
    try:
        if os.path.exists(USER_PROFILE_FILE):
            profile_df = pd.read_csv(USER_PROFILE_FILE)
            user_profile = profile_df[profile_df['user_id'] == user_id]
            if not user_profile.empty:
                logger.info(f"Profile loaded for user {user_id}.")
                return user_profile.iloc[0]
        logger.info(f"No profile found for user {user_id}.")
        return None
    except Exception as e:
        logger.error(f"Error loading user profile for {user_id}: {str(e)}")
        st.error(f"❌ Error loading your profile: {str(e)}")
        return None

def save_user_profile(user_id, age, height, weight, gender, medical_conditions, diet_goal, vegetarian, activity_level):
    """Save user profile to CSV, including BMI and daily calories."""
    try:
        if not os.path.exists(USER_PROFILE_FILE):
            ensure_user_db() # Ensure the file exists
            
        profile_df = pd.read_csv(USER_PROFILE_FILE)

        bmi = calculate_bmi(weight, height)
        daily_calories = calculate_daily_calories(age, gender, weight, height, activity_level, diet_goal)
        
        # Convert list of medical conditions to a comma-separated string
        medical_conditions_str = ", ".join(medical_conditions) if medical_conditions else ""

        # Check if user profile already exists to update or create new
        if user_id in profile_df['user_id'].values:
            profile_df.loc[profile_df['user_id'] == user_id, [
                'age', 'height', 'weight', 'gender', 'medical_conditions', 'diet_goal',
                'vegetarian', 'bmi', 'daily_calories', 'activity_level', 'created_at'
            ]] = [
                age, height, weight, gender, medical_conditions_str, diet_goal,
                vegetarian, bmi, daily_calories, activity_level, datetime.now().isoformat()
            ]
            st.success("✅ Profile updated successfully!")
            logger.info(f"Profile updated for user {user_id}.")
        else:
            new_profile = pd.DataFrame([{
                'user_id': user_id,
                'age': age,
                'height': height,
                'weight': weight,
                'gender': gender,
                'medical_conditions': medical_conditions_str,
                'diet_goal': diet_goal,
                'vegetarian': vegetarian,
                'bmi': bmi,
                'daily_calories': daily_calories,
                'activity_level': activity_level,
                'created_at': datetime.now().isoformat()
            }])
            profile_df = pd.concat([profile_df, new_profile], ignore_index=True)
            st.success("✅ Profile saved successfully!")
            logger.info(f"New profile created for user {user_id}.")
            
        profile_df.to_csv(USER_PROFILE_FILE, index=False)
        return True
    except Exception as e:
        logger.error(f"Error saving user profile for {user_id}: {str(e)}")
        st.error(f"❌ Error saving profile: {str(e)}")
        return False

# --- Nutrition Calculations ---
def calculate_bmi(weight, height):
    """Calculate BMI with error handling"""
    try:
        if height <= 0 or weight <= 0:
            st.warning("⚠️ Height and weight must be positive values to calculate BMI.")
            return 0
        return round(weight / ((height/100) ** 2), 1)
    except Exception as e:
        logger.error(f"Error calculating BMI: {str(e)}")
        st.error(f"❌ Error calculating BMI: {str(e)}")
        return 0

def calculate_daily_calories(age, gender, weight, height, activity_level, diet_goal):
    """Calculate daily calories using Mifflin-St Jeor equation"""
    try:
        # BMR calculation
        if gender.lower() == "male":
            bmr = (10 * weight) + (6.25 * height) - (5 * age) + 5
        else: # Assuming female for any other input or default
            bmr = (10 * weight) + (6.25 * height) - (5 * age) - 161
        
        # Activity multipliers
        activity_multipliers = {
            "sedentary": 1.2, # little or no exercise
            "light": 1.375, # light exercise/sports 1-3 days/week
            "moderate": 1.55, # moderate exercise/sports 3-5 days/week
            "active": 1.725, # hard exercise/sports 6-7 days a week
            "very active": 1.9 # very hard exercise/physical job
        }
        
        # Goal adjustments
        goal_adjustments = {
            "weight loss": -500,
            "weight gain": 500,
            "muscle gain": 300, # A slight caloric surplus for muscle gain
            "maintain weight": 0
        }
        
        # Calculate TDEE (Total Daily Energy Expenditure)
        tdee = bmr * activity_multipliers.get(activity_level.lower(), 1.55) # Default to moderate if invalid
        goal = diet_goal.lower()
        adjustment = goal_adjustments.get(goal, 0)
        
        # Ensure minimum calories and return rounded value
        calculated_calories = round(tdee + adjustment)
        return max(1200, calculated_calories)  # Minimum 1200 calories for health
        
    except Exception as e:
        logger.error(f"Error calculating daily calories: {str(e)}")
        st.error(f"❌ Error calculating daily calorie needs: {str(e)}. Defaulting to 2000.")
        return 2000  # Default fallback

# --- Enhanced Meal Planning ---
def generate_comprehensive_meal_plan():
    """Generate meal plan using Google API with user profile"""
    try:
        profile = load_user_profile(st.session_state.user_id)
        if profile is None:
            st.error("❌ Please complete your profile first before generating a meal plan.")
            return None

        # Parse medical conditions from the profile (stored as a comma-separated string)
        medical_conditions = []
        if pd.notna(profile['medical_conditions']) and profile['medical_conditions']:
            medical_conditions = [condition.strip() for condition in str(profile['medical_conditions']).split(',') if condition.strip()]
            
        st.info("🔄 Generating your personalized meal plan based on your profile and medical conditions...")
        
        # Generate meal plan using Google API
        with st.spinner("🤖 AI is meticulously crafting your personalized meal plan... This might take a moment."):
            meal_plan = generate_medical_condition_diet(
                medical_conditions=medical_conditions,
                vegetarian=profile['vegetarian'],
                daily_calories=int(profile['daily_calories']), # Ensure integer
                diet_goal=profile['diet_goal']
            )
        
        if meal_plan:
            # Add metadata for saving/logging
            meal_plan['user_id'] = st.session_state.user_id
            meal_plan['generated_at'] = datetime.now().isoformat()
            meal_plan['medical_conditions_applied'] = medical_conditions # Store as list
            meal_plan['vegetarian_pref'] = profile['vegetarian']
            meal_plan['daily_target_calories'] = profile['daily_calories']
            
            # Calculate coverage (how close the generated plan is to the target calories)
            total_calories_generated = meal_plan.get('daily_summary', {}).get('total_calories', 0)
            meal_plan['coverage'] = round((total_calories_generated / profile['daily_calories']) * 100, 1) if profile['daily_calories'] > 0 else 0
            
            st.session_state.current_meal_plan = meal_plan # Store in session state for display
            st.session_state.generated_meal_count += 1
            st.success("✅ Your personalized meal plan has been generated successfully!")
            return meal_plan
        else:
            st.error("❌ Failed to generate meal plan. The AI could not create a suitable plan based on your criteria. Please refine your profile or try again.")
            return None
            
    except Exception as e:
        logger.error(f"Critical error generating comprehensive meal plan: {str(e)}")
        st.error(f"❌ An unexpected error occurred while trying to generate your meal plan: {str(e)}")
        return None

def save_meal_feedback(meal_plan_dict, rating, feedback):
    """Save meal feedback for future personalization"""
    try:
        if not os.path.exists(MEAL_HISTORY_FILE):
            ensure_user_db()  # Ensure the file exists

        history_df = pd.read_csv(MEAL_HISTORY_FILE)

        # Recursively convert boolean and other non-serializable values to strings
        def convert_bools(obj):
            if isinstance(obj, dict):
                return {k: convert_bools(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_bools(item) for item in obj]
            elif isinstance(obj, (bool, np.bool_)):
                return str(obj)
            return obj

        meal_plan_serializable = convert_bools(meal_plan_dict.copy())

        new_feedback = pd.DataFrame([{
            "user_id": st.session_state.user_id,
            "date": datetime.now().isoformat(),
            "meal": json.dumps(meal_plan_serializable, default=str),  # Handle any leftover unserializable types
            "rating": rating,
            "feedback": feedback
        }])

        history_df = pd.concat([history_df, new_feedback], ignore_index=True)
        history_df.to_csv(MEAL_HISTORY_FILE, index=False)

        st.success("📝 Thank you for your feedback! It helps us improve your future meal plans.")
        logger.info(f"Feedback saved for user {st.session_state.user_id} with rating {rating}.")
        return True

    except Exception as e:
        logger.error(f"Error saving meal feedback: {str(e)}")
        st.error(f"❌ Error saving your feedback: {str(e)}")
        return False
    
# --- UI Components ---
def show_auth_interface():
    """Enhanced authentication interface"""
    st.title("🥗 Nutrition Assistant")
    st.markdown("*Personalized meal planning powered by AI*")
    
    # Create tabs for login and register
    tab1, tab2 = st.tabs(["🔑 Login", "📝 Register"])
    
    with tab1:
        st.subheader("Welcome Back!")
        
        with st.form("login_form"):
            login_email = st.text_input("📧 Email", placeholder="Enter your email").strip()
            password = st.text_input("🔒 Password", type="password", placeholder="Enter your password").strip()
            
            col1, col2 = st.columns(2)
            with col1:
                login_btn = st.form_submit_button("Login", type="primary", use_container_width=True)
            with col2:
                forgot_btn = st.form_submit_button("Forgot Password?", use_container_width=True)
            
            if login_btn:
                if not login_email or not password:
                    st.error("❌ Please enter both email and password.")
                else:
                    result = check_login(login_email, password)
                    
                    if result == "not_registered":
                        st.error("❌ User not found. Please register first.")
                        st.session_state.show_register = True # Suggest registering
                    elif result == "not_verified":
                        st.warning("⚠️ Your account is not verified. Check your email for OTP.")
                        st.session_state.user_id = login_email
                        st.session_state.show_otp_verification = True
                    elif result == "wrong_password":
                        st.session_state.login_attempts += 1
                        st.error(f"❌ Incorrect password. Attempts: {st.session_state.login_attempts}/3")
                        if st.session_state.login_attempts >= 3:
                            st.warning("Too many failed attempts. Please use 'Forgot Password?' or try again later.")
                            # Optionally, disable login for a period or lock account
                    elif result == "change_required":
                        st.session_state.user_id = login_email
                        st.session_state.show_password_change = True
                        st.success("🔐 Temporary password used. Please set a new password.")
                    elif result == "success":
                        st.session_state.logged_in = True
                        st.session_state.user_id = login_email
                        st.session_state.login_attempts = 0 # Reset attempts on successful login
                        user_profile = load_user_profile(login_email)
                        if user_profile is not None:
                            st.session_state.profile_completed = True
                            st.session_state.vegetarian = user_profile['vegetarian']
                            st.session_state.daily_calories = user_profile['daily_calories']
                            st.session_state.activity_level = user_profile['activity_level']
                            st.session_state.diet_goal = user_profile['diet_goal']
                        else:
                            st.session_state.profile_completed = False
                        st.success("✅ Logged in successfully!")
                        st.rerun() # Rerun to switch to main app interface
                    elif result == "error":
                        st.error("❌ An internal error occurred. Please try again later.")

            if forgot_btn:
                if login_email:
                    # In a real app, you'd send a password reset link/OTP to the email
                    st.info("ℹ️ If this email is registered, a password reset link/OTP has been sent. (Functionality not fully implemented here)")
                    # For this example, we can trigger OTP verification if user exists
                    df = pd.read_csv(USER_DB_FILE)
                    if login_email in df['user_id'].values:
                        user_record = df[df['user_id'] == login_email].iloc[0]
                        new_otp = generate_otp()
                        df.loc[df['user_id'] == login_email, 'otp'] = new_otp
                        df.loc[df['user_id'] == login_email, 'verified'] = False # Re-verify for password reset
                        df.loc[df['user_id'] == login_email, 'needs_password_change'] = True # Force password change
                        df.to_csv(USER_DB_FILE, index=False)
                        if send_email(login_email, "Nutrition Chatbot Password Reset OTP", f"Your OTP for password reset is: {new_otp}"):
                            st.session_state.user_id = login_email
                            st.session_state.show_otp_verification = True
                            st.session_state.show_password_change = False # Ensure password change is only after OTP
                            st.success("✉️ An OTP has been sent to your email for password reset.")
                        else:
                            st.error("❌ Failed to send password reset OTP. Please try again.")
                else:
                    st.warning("Please enter your email to reset your password.")


    with tab2:
        st.subheader("Join Us Today!")
        with st.form("register_form"):
            register_email = st.text_input("📧 Your Email", placeholder="Enter your email to register").strip()
            register_btn = st.form_submit_button("Register", type="primary", use_container_width=True)

            if register_btn:
                if register_email:
                    if register_user(register_email):
                        st.session_state.user_id = register_email
                        st.session_state.show_otp_verification = True
                        # The success message is handled inside register_user
                    # Error messages are also handled inside register_user
                else:
                    st.error("❌ Please enter an email to register.")

    if st.session_state.show_otp_verification:
        st.info(f"Please enter the OTP sent to **{st.session_state.user_id}**")
        with st.form("otp_form"):
            otp_input = st.text_input("Enter OTP", max_chars=6).strip()
            otp_verify_btn = st.form_submit_button("Verify OTP", type="secondary", use_container_width=True)

            if otp_verify_btn:
                if verify_otp(st.session_state.user_id, otp_input):
                    st.success("✅ Account verified successfully!")
                    st.session_state.show_otp_verification = False
                    st.session_state.show_password_change = True # After OTP, if it's a new user or reset
                else:
                    st.error("❌ Invalid OTP. Please try again.")
    
    if st.session_state.show_password_change:
        st.subheader("Change Your Password")
        st.warning("Please set a new, strong password.")
        with st.form("password_change_form"):
            new_password = st.text_input("New Password", type="password", placeholder="Enter your new password").strip()
            confirm_password = st.text_input("Confirm New Password", type="password", placeholder="Confirm your new password").strip()
            change_password_btn = st.form_submit_button("Set New Password", type="primary", use_container_width=True)

            if change_password_btn:
                if new_password and confirm_password:
                    if new_password == DEFAULT_PASSWORD:
                        st.error("❌ New password cannot be the same as the temporary password.")
                    elif new_password == confirm_password:
                        if len(new_password) < 8 or not any(char.isdigit() for char in new_password) or \
                           not any(char.isupper() for char in new_password) or \
                           not any(char.islower() for char in new_password) or \
                           not any(char in "!@#$%^&*()-_+=" for char in new_password):
                            st.error("❌ Password must be at least 8 characters long and contain at least one uppercase letter, one lowercase letter, one digit, and one special character (!@#$%^&*()-_+=).")
                        else:
                            if update_password(st.session_state.user_id, new_password):
                                st.success("✅ Password updated successfully! Please login with your new password.")
                                st.session_state.show_password_change = False
                                st.session_state.show_otp_verification = False # Ensure OTP is reset
                                st.session_state.logged_in = False # Force re-login with new password
                                st.rerun() # Rerun to show login form
                            else:
                                st.error("❌ Failed to update password. Please try again.")
                    else:
                        st.error("❌ Passwords do not match.")
                else:
                    st.error("❌ Please fill in both password fields.")

def show_profile_completion_interface():
    """Display and handle user profile completion."""
    st.header("👤 Complete Your Profile")
    st.write("Please provide some information to help us create a personalized meal plan for you.")

    user_profile = load_user_profile(st.session_state.user_id)
    
    # Pre-fill with existing data if available
    initial_age = user_profile['age'] if user_profile is not None and pd.notna(user_profile['age']) else 25
    initial_height = user_profile['height'] if user_profile is not None and pd.notna(user_profile['height']) else 170
    initial_weight = user_profile['weight'] if user_profile is not None and pd.notna(user_profile['weight']) else 70
    initial_gender = user_profile['gender'] if user_profile is not None and pd.notna(user_profile['gender']) else "Male"
    
    # Handle medical conditions (stored as comma-separated string)
    initial_medical_conditions_str = user_profile['medical_conditions'] if user_profile is not None and pd.notna(user_profile['medical_conditions']) else ""
    initial_medical_conditions = [cond.strip() for cond in initial_medical_conditions_str.split(',') if cond.strip()]

    initial_diet_goal = user_profile['diet_goal'] if user_profile is not None and pd.notna(user_profile['diet_goal']) else "Maintain Weight"
    initial_vegetarian = user_profile['vegetarian'] if user_profile is not None and pd.notna(user_profile['vegetarian']) else False
    
    # Fix for activity level - ensure case matches the options
    activity_options = ["Sedentary", "Light", "Moderate", "Active", "Very Active"]
    initial_activity_level = user_profile['activity_level'] if user_profile is not None and pd.notna(user_profile['activity_level']) else "Moderate"
    # Convert to proper case if needed
    if initial_activity_level.lower() == "moderate":
        initial_activity_level = "Moderate"

    with st.form("profile_form", clear_on_submit=False):
        age = st.slider("Age (Years)", 18, 99, int(initial_age))
        height = st.slider("Height (cm)", 100, 250, int(initial_height))
        weight = st.slider("Weight (kg)", 30, 200, int(initial_weight))
        gender = st.radio("Gender", ["Male", "Female", "Other"], index=["Male", "Female", "Other"].index(initial_gender))
        
        # Multiselect for medical conditions
        all_medical_conditions = list(MEDICAL_CONDITIONS.keys())
        medical_conditions = st.multiselect(
            "Select any applicable medical conditions:",
            options=all_medical_conditions,
            default=initial_medical_conditions,
            help="This will help us tailor your meal plan to your health needs."
        )

        diet_goal = st.selectbox(
            "Your Diet Goal",
            ["Maintain Weight", "Weight Loss", "Weight Gain", "Muscle Gain"],
            index=["Maintain Weight", "Weight Loss", "Weight Gain", "Muscle Gain"].index(initial_diet_goal)
        )
        vegetarian = st.checkbox("Are you Vegetarian?", value=initial_vegetarian)
        activity_level = st.selectbox(
            "Activity Level",
            activity_options,
            index=activity_options.index(initial_activity_level)
        )

        # Proper submit button using st.form_submit_button()
        submit_profile_btn = st.form_submit_button("Save Profile", type="primary", use_container_width=True)

        if submit_profile_btn:
            if age and height and weight and gender and diet_goal and activity_level:
                if save_user_profile(st.session_state.user_id, age, height, weight, gender, medical_conditions, diet_goal, vegetarian, activity_level):
                    st.session_state.profile_completed = True
                    # Update session state with new profile values
                    st.session_state.vegetarian = vegetarian
                    st.session_state.daily_calories = calculate_daily_calories(age, gender, weight, height, activity_level, diet_goal)
                    st.session_state.activity_level = activity_level
                    st.session_state.diet_goal = diet_goal
                    st.rerun() # Rerun to transition to main app
            else:
                st.error("❌ Please fill in all required profile fields.")

def display_meal_plan(meal_plan):
    """Displays the generated meal plan in a structured way."""
    if not meal_plan:
        st.warning("No meal plan available to display.")
        return

    st.subheader(f"🍽️ Your Personalized Meal Plan")
    st.write(f"Generated for: **{st.session_state.user_id}** on **{datetime.now().strftime('%Y-%m-%d')}**")

    daily_summary = meal_plan.get("daily_summary", {})
    st.markdown("---")
    colored_header(
        label=f"Daily Summary: {round(daily_summary.get('total_calories', 0))} / {st.session_state.daily_calories} Calories ({meal_plan.get('coverage', 0)}% Coverage)",
        description=f"Protein: {round(daily_summary.get('total_protein', 0))}g | Carbs: {round(daily_summary.get('total_carbs', 0))}g | Fats: {round(daily_summary.get('total_fats', 0))}g | Fiber: {round(daily_summary.get('total_fiber', 0))}g",
        color_name="green-70",
    )
    st.info(f"**Medical Compliance:** {daily_summary.get('medical_compliance', 'N/A')}")
    st.markdown("---")

    meal_order = ["breakfast", "lunch", "dinner", "snacks"]
    for meal_type in meal_order:
        meal_data = meal_plan.get(meal_type)
        if meal_data:
            st.markdown(f"#### 🥣 {meal_type.capitalize()} (Target: {round(st.session_state.daily_calories * (0.25 if meal_type == 'breakfast' else (0.35 if meal_type == 'lunch' else (0.30 if meal_type == 'dinner' else 0.10))))} kcal)")
            st.markdown(f"**Total Calories:** {round(meal_data.get('total_calories', 0))} kcal")
            st.markdown(f"**Benefits:** {meal_data.get('meal_benefits', 'N/A')}")
            
            for item in meal_data.get("items", []):
                with st.expander(f"**{item.get('food', 'N/A')}** - {item.get('quantity', 'N/A')}"):
                    st.write(f"Calories: {item.get('calories', 0)} kcal")
                    st.write(f"Protein: {item.get('protein', 0)}g")
                    st.write(f"Carbs: {item.get('carbs', 0)}g")
                    st.write(f"Fats: {item.get('fats', 0)}g")
                    st.write(f"Fiber: {item.get('fiber', 0)}g")
                    st.write(f"**Benefits:** {item.get('benefits', 'N/A')}")

                    # Option to find alternatives
                    # Get user's current medical conditions and vegetarian preference
                    user_profile_data = load_user_profile(st.session_state.user_id)
                    current_medical_conditions = []
                    if user_profile_data is not None and pd.notna(user_profile_data['medical_conditions']):
                        current_medical_conditions = [cond.strip() for cond in str(user_profile_data['medical_conditions']).split(',') if cond.strip()]
                    current_vegetarian = user_profile_data['vegetarian'] if user_profile_data is not None else False

                    if st.button(f"Find alternatives for {item.get('food', 'this item')}", key=f"alt_{meal_type}_{item.get('food')}"):
                        alternatives = get_food_alternatives(item['food'], current_medical_conditions, current_vegetarian)
                        if alternatives:
                            st.success(f"Here are some alternatives for {item['food']}:")
                            for alt in alternatives:
                                st.write(f"- {alt}")
                        else:
                            st.warning(f"Could not find alternatives for {item['food']} at the moment.")
            st.markdown("---")

    # Feedback Section
    st.markdown("### 👍 Rate Your Meal Plan")
    with st.form("meal_feedback_form"):
        rating = st.slider("How would you rate this meal plan?", 1, 5, 3)
        feedback = st.text_area("Your feedback (optional)", "What did you like? What could be improved?")
        feedback_submit_btn = st.form_submit_button("Submit Feedback", type="primary", use_container_width=True)

        if feedback_submit_btn:
            if save_meal_feedback(meal_plan, rating, feedback):
                st.session_state.current_meal_plan = None # Clear after feedback
                st.rerun() # Rerun to refresh the meal plan section
            else:
                st.error("❌ Failed to save feedback. Please try again.")

def show_main_app_interface():
    """Main application interface after login and profile completion."""
    st.sidebar.title(f"Welcome, {st.session_state.user_id.split('@')[0]}!")

    if st.sidebar.button("⚙️ Edit Profile"):
        st.session_state.profile_completed = False # Go back to profile editing
        st.session_state.current_meal_plan = None # Clear meal plan if profile is edited
        st.rerun()

    if st.sidebar.button("🔄 Generate New Meal Plan"):
        st.session_state.current_meal_plan = generate_comprehensive_meal_plan()
        st.rerun()
        
    if st.sidebar.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.session_state.user_id = ""
        st.session_state.profile_completed = False
        st.session_state.login_attempts = 0
        st.session_state.show_register = False
        st.session_state.show_password_change = False
        st.session_state.show_otp_verification = False
        st.session_state.current_meal_plan = None
        st.session_state.generated_meal_count = 0
        st.success("👋 You have been logged out.")
        st.rerun()

    # Main content area
    st.header("📊 Your Daily Nutrition Dashboard")

    user_profile = load_user_profile(st.session_state.user_id)
    if user_profile is not None:
        bmi = user_profile['bmi']
        daily_calories = user_profile['daily_calories']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("BMI", f"{bmi}")
            st.caption(get_bmi_category(bmi))
        with col2:
            st.metric("Target Calories", f"{int(daily_calories)} kcal")
        with col3:
            st.metric("Diet Goal", user_profile['diet_goal'])

        st.markdown("---")
        st.markdown("### 📋 Current Meal Plan")

        if st.session_state.current_meal_plan:
            display_meal_plan(st.session_state.current_meal_plan)
        else:
            st.info("No meal plan generated yet for today. Click 'Generate New Meal Plan' in the sidebar to get started!")

    else:
        st.error("❌ User profile not found. Please complete your profile.")
        st.session_state.profile_completed = False # Force profile completion
        st.rerun()


def get_bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif 18.5 <= bmi < 24.9:
        return "Normal weight"
    elif 25 <= bmi < 29.9:
        return "Overweight"
    else:
        return "Obese"

# --- Main Application Flow ---
def main():
    """Main function to run the Streamlit application"""
    st.set_page_config(
        page_title="Nutrition Assistant",
        page_icon="🥗",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    initialize_session_state()

    # Ensure databases exist before anything else
    if not ensure_user_db() or not ensure_nutrition_db() or not ensure_food_db():
        st.stop() # Stop execution if database setup fails

    # Load environment variables and configure Google API
    if not load_environment():
        st.stop() # Stop if API key is missing

    # Apply custom CSS
    st.markdown("""
        <style>
        .stButton>button {
            border-radius: 20px;
            border: 1px solid #4CAF50;
            color: #FFFFFF;
            background-color: #4CAF50;
            font-size: 16px;
            padding: 10px 24px;
            cursor: pointer;
        }
        .stButton>button:hover {
            background-color: #45a049;
            border: 1px solid #45a049;
        }
        .stTextInput>div>div>input {
            border-radius: 20px;
            padding: 10px 15px;
        }
        .stTextArea>div>div>textarea {
            border-radius: 20px;
            padding: 10px 15px;
        }
        .stSlider .stSliderHandle {
            background-color: #4CAF50;
        }
        .stSlider .stSliderTrack {
            background-color: #E6E6E6;
        }
        .stSlider [data-baseweb="slider"] {
            background-color: #4CAF50;
        }
        .stAlert {
            border-radius: 10px;
        }
        .css-1d391kg e16zcsfj9 { /* This targets the main content area */
            padding-top: 1rem;
        }
        </style>
        """, unsafe_allow_html=True)

    if not st.session_state.logged_in:
        show_auth_interface()
    elif not st.session_state.profile_completed:
        show_profile_completion_interface()
    else:
        show_main_app_interface()

if __name__ == "__main__":
    main()